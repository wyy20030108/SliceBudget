"""Reassess frozen slice policies by detector, anatomy, phase, and area."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score

from acdc_data import _load_case, make_protocol
from audit_decision_curves import OPERATING_POINTS, interval, matched_recall, patient_histograms
from audit_external_transfer import frozen_protocol, replay_manifest
from experiment import load_json, log, save_json, scores
from run_matched_slice_budget import calibrate, rates
from segment import configure, predict_many


CORE = ("interval", "uncertain", "targeted_unbalanced", "targeted")
EXTRAS = ("uniform_random", "pooled_image_farthest", "mixed_image")
ARMS = CORE + EXTRAS
LABELS = (1, 2, 3)
SCORES = ("top1", "top0_1", "component_area")
NEGATIVES = ("absent", "near_absent")
COMPARANDS = ("interval", "uncertain", "targeted_unbalanced", *EXTRAS)


def connected_area(predicted, foreground):
    components, count = connected_components(predicted == foreground)
    return int(np.bincount(components.ravel())[1:].max()) if count else 0


def infer(cases, teacher, device):
    patients = {}
    for (patient, case), probabilities in zip(cases.items(), predict_many(
            SimpleNamespace(teacher=teacher), cases.values(), device), strict=True):
        predicted = probabilities.argmax(axis=1)
        flat = probabilities[:, 1:].reshape(len(probabilities), 3, -1)
        pixels = flat.shape[-1]
        top = {}
        for name, fraction in (("top1", 0.01), ("top0_1", 0.001)):
            k = max(1, round(pixels * fraction))
            top[name] = np.partition(flat, -k, axis=-1)[..., -k:].mean(axis=-1)
        outcome = scores(case, predicted)
        rows = []
        for column, foreground in enumerate(LABELS):
            truth = case.masks == foreground
            predicted_mask = predicted == foreground
            area = np.count_nonzero(truth, axis=(1, 2))
            pred_area = np.count_nonzero(predicted_mask, axis=(1, 2))
            overlap = np.count_nonzero(truth & predicted_mask, axis=(1, 2))
            for phase in (0, 1):
                indices = np.flatnonzero(case.phase == phase)
                peak = int(area[indices].max())
                small = (area[indices] > 0) & (4 * area[indices] <= peak)
                positions = case.slice_index[indices].astype(int)
                marked = positions[small]
                near = np.isin(positions - 1, marked) | np.isin(positions + 1, marked)
                for local, index in enumerate(indices):
                    rows.append({"label": foreground, "phase": phase, "small": bool(small[local]),
                                 "absent": bool(area[index] == 0),
                                 "near_absent": bool(area[index] == 0 and near[local]),
                                 "area": int(area[index]), "peak": peak, "top1_pixels": round(pixels / 100),
                                 "dice": float(2 * overlap[index] / (area[index] + pred_area[index])) if area[index] else None,
                                 "top1": float(top["top1"][index, column]),
                                 "top0_1": float(top["top0_1"][index, column]),
                                 "component_area": connected_area(predicted[index], foreground) / pixels})
        patients[patient] = {"rows": rows, "global_dice": outcome["global_dice"],
                             "hard_dice": outcome["hard_dice"]}
    return patients


def registration_records(data, score):
    return {
        patient: ({label: {key: np.array([row[score] for row in item["rows"]
                                         if row["label"] == label and row[key]]) for key in ("small", *NEGATIVES)}
                   for label in LABELS}, item["global_dice"], item["hard_dice"])
        for patient, item in data.items()}


def verify_replay(data, thresholds, reference):
    calculated = rates(registration_records(data, "top1"), thresholds)
    recorded = reference["assessment"]
    if calculated.keys() != recorded.keys():
        raise ValueError("Original official ACDC roster differs")
    differences = []
    for patient, row in calculated.items():
        for key, value in row.items():
            expected = recorded[patient][key]
            if (value is None) != (expected is None):
                raise ValueError("Replay changes metric eligibility")
            if value is not None:
                differences.append(abs(value - expected))
    return max(differences)


def source_calibration(data, target):
    report = {}
    for score in SCORES:
        result, support = calibrate(registration_records(data, score), target)
        report[score] = {label: {"cutoff": result[label], "patients": support[label],
                                 "slices": sum(row["label"] == label and row["absent"]
                                               for item in data.values() for row in item["rows"]),
                                 "fallback": support[label] < 5} for label in LABELS}
    return report


def gather(args):
    settings, cache, _, source = frozen_protocol(args)
    replay_manifest(args.core_weights, source)
    calibration = {patient: _load_case(cache[patient]) for patient in source["calibration"]}
    reference_run = load_json(args.official_original / "run.json")
    if not reference_run["held_out_test_used"]:
        raise ValueError("Expected the original official ACDC test roster")
    official_paths = {path.stem: path for path in args.official.glob("*.npz")}
    if len(official_paths) != 50 or official_paths.keys() != set(reference_run["assessment"]):
        raise ValueError("Official test cache must match the recorded assessment")
    if official_paths.keys() & cache.keys():
        raise ValueError("Training and official evaluation caches overlap")
    evaluation = {"official": {patient: _load_case(official_paths[patient]) for patient in sorted(official_paths)}}
    if args.sealed:
        holdout = sorted(make_protocol(cache, settings["holdout_protocol_seed"])["test"])
        if len(holdout) != 20 or set(holdout) & set(source["training"] + source["calibration"]):
            raise ValueError("Invalid originally reserved ACDC roster")
        evaluation["reserved"] = {patient: _load_case(cache[patient]) for patient in holdout}
    result = {"fold": args.fold, "arms": {}, "cohorts": list(evaluation)}
    device = configure("cuda")
    for arm in ARMS:
        weights = args.core_weights if arm in CORE else args.extra_weights
        if arm in CORE:
            reference_path = args.official_original / f"{arm}.json"
        elif arm == "mixed_image":
            reference_path = args.mixed_runs / "mixed_image-official.json"
        else:
            reference_path = args.baseline_runs / f"{arm}.json"
        reference = load_json(reference_path)
        teacher = torch.load(weights / f"{arm}.pt", map_location="cpu", weights_only=True)
        cuts = source_calibration(infer(calibration, teacher, device), settings["calibration_false_positive_rate"])
        archived = {int(key): value for key, value in reference["thresholds"].items()}
        max_cut_delta = max(abs(cuts["top1"][key]["cutoff"] - value) for key, value in archived.items())
        predictions = {name: infer(cases, teacher, device) for name, cases in evaluation.items()}
        max_metric_delta = verify_replay(predictions["official"], archived, reference)
        if max_cut_delta > 1e-6 or max_metric_delta > 1e-6:
            raise ValueError(f"Source replay failed for {arm}: {max_cut_delta}, {max_metric_delta}")
        result["arms"][arm] = {"calibration": cuts, "cohorts": predictions,
                               "source_thr_max_delta": max_cut_delta, "official_metric_max_delta": max_metric_delta}
        log("stratified_inference", arm=arm, fold=args.fold, threshold_delta=max_cut_delta, metric_delta=max_metric_delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


def filter_rows(item, structure, phase, size):
    rows = [row for row in item["rows"]
            if (structure is None or row["label"] == structure) and (phase is None or row["phase"] == phase)]
    positives = [row for row in rows if row["small"]]
    if size == "rel_le_10":
        positives = [row for row in positives if 10 * row["area"] <= row["peak"]]
    if size == "rel_10_to_25":
        positives = [row for row in positives if 10 * row["area"] > row["peak"]]
    if size == "abs_below_top1":
        positives = [row for row in positives if row["area"] < row["top1_pixels"]]
    if size == "abs_at_least_top1":
        positives = [row for row in positives if row["area"] >= row["top1_pixels"]]
    return {"small": positives, **{subset: [row for row in rows if row[subset]] for subset in NEGATIVES}}


def scores_by_patient(groups, cutoff, score):
    return {subset: [[row[score] - cutoff[str(row["label"])]["cutoff"] for row in group[subset]]
                     for group in groups] for subset in ("small", *NEGATIVES)}


def partial_auc(positive, negative):
    outcome, value, weight = [], [], []
    for truth, groups in ((1, positive), (0, negative)):
        for row in groups:
            if row:
                outcome.extend([truth] * len(row))
                value.extend(row)
                weight.extend([1 / len(row)] * len(row))
    if len(set(outcome)) < 2:
        return None
    return float(roc_auc_score(outcome, value, sample_weight=weight, max_fpr=0.4))


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def evaluate_group(folds, cohort, patients, assignments, score, structure, phase, size, points):
    curves = {}
    methods = {}
    support = None
    for arm in ARMS:
        estimates = {name: [] for name in ("small_dice", "recall", "fpr", "near_fpr")}
        arm_curves = {key: [] for key in NEGATIVES}
        p_auc = {key: [] for key in NEGATIVES}
        for fold in folds:
            run = fold["arms"][arm]
            groups = [filter_rows(run["cohorts"][cohort][patient], structure, phase, size) for patient in patients]
            margins = scores_by_patient(groups, run["calibration"][score], score)
            counts = {key: {"patients": sum(bool(row) for row in rows), "slice_classes": sum(map(len, rows))}
                      for key, rows in margins.items()}
            if support is None:
                support = counts
            if support != counts:
                raise ValueError("Contrasts must match the image, structure, phase, and reference support")
            estimates["small_dice"].append(mean_or_none([np.mean([row["dice"] for row in g["small"]])
                                                         for g in groups if g["small"]]))
            for subset, metric in (("small", "recall"), ("absent", "fpr"), ("near_absent", "near_fpr")):
                estimates[metric].append(mean_or_none([np.mean(np.asarray(row) > 0)
                                                        for row in margins[subset] if row]))
            for key in NEGATIVES:
                p_auc[key].append(partial_auc(margins["small"], margins[key]))
                if support["small"]["patients"] and support[key]["patients"]:
                    h = patient_histograms(margins["small"], margins[key])
                    arm_curves[key].append(matched_recall(h, assignments, points))
        curves[arm] = {key: np.mean(x, axis=0) for key, x in arm_curves.items() if x}
        methods[arm] = {key: mean_or_none([x for x in v if x is not None]) for key, v in estimates.items()}
        methods[arm]["matched_recall"] = {key: data[0].tolist() for key, data in curves[arm].items()}
        methods[arm]["standardized_pauc"] = {key: mean_or_none([x for x in v if x is not None])
                                             for key, v in p_auc.items()}
    comparisons = {}
    for arm in COMPARANDS:
        contrasts = {}
        for key, value in curves["targeted"].items():
            differences = value - curves[arm][key]
            contrasts[key] = {"difference": differences[0].tolist(),
                              "intervals": [interval(column) for column in differences[1:].T] if len(differences) > 1 else None}
        comparisons[arm] = contrasts
    return {"fpr_points": list(points), "support": support, "methods": methods, "targeted_minus": comparisons}


def summarize(args):
    folds = [load_json(path) for path in args.folds]
    if {row["fold"] for row in folds} != {0, 1} or any(tuple(row["arms"]) != ARMS for row in folds):
        raise ValueError("Expected two distinct training folds with all seven policies")
    if folds[0]["cohorts"] != folds[1]["cohorts"]:
        raise ValueError("Both folds must share cohort names")
    result = {"score_definitions": {"top1": "mean of top 1% class probabilities",
                                     "top0_1": "mean of top 0.1% class probabilities",
                                     "component_area": "largest 4-connected argmax component, as fraction of pixels"},
              "bootstrap_draws": args.draws, "pointwise": True, "cohorts": {},
              "calibration": {row["fold"]: {arm: value["calibration"] for arm, value in row["arms"].items()} for row in folds},
              "replay_max_deltas": {row["fold"]: {arm: {"thresholds": v["source_thr_max_delta"],
                                                         "patient_metrics": v["official_metric_max_delta"]}
                                                   for arm, v in row["arms"].items()} for row in folds}}
    for cohort in folds[0]["cohorts"]:
        patients = sorted(folds[0]["arms"]["interval"]["cohorts"][cohort])
        for row in folds:
            for record in row["arms"].values():
                if sorted(record["cohorts"][cohort]) != patients:
                    raise ValueError("Evaluation rosters do not match")
        count = len(patients)
        rng = np.random.default_rng(20261116)
        assignments = np.vstack([np.ones(count), rng.multinomial(count, np.full(count, 1 / count), size=args.draws)])
        strata = [("pooled", None, None, None)] + [(f"label{lab}", lab, None, None) for lab in LABELS]
        if cohort == "official":
            strata += [(f"label{lab}_phase{phase}", lab, phase, None) for lab in LABELS for phase in (0, 1)]
            strata += [(size, None, None, size) for size in ("rel_le_10", "rel_10_to_25", "abs_below_top1", "abs_at_least_top1")]
        report = {}
        for score in SCORES:
            report[score] = {}
            for key, structure, phase, size in strata:
                if score != "top1" and (phase is not None or size is not None):
                    continue
                primary = key in ("pooled", "label1", "label2", "label3")
                selected_assignments = assignments if primary else assignments[:1]
                points = OPERATING_POINTS if key == "pooled" else (0.2,)
                report[score][key] = evaluate_group(folds, cohort, patients, selected_assignments, score,
                                                    structure, phase, size, points)
            log("stratified_summary", cohort=cohort, score=score, patients=count)
        result["cohorts"][cohort] = {"patients": count, "scores": report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    for field in ("data", "official", "registration", "original", "official-original", "core-weights",
                  "extra-weights", "baseline-runs", "mixed-runs", "output"):
        score.add_argument(f"--{field}", required=True, type=Path)
    score.add_argument("--fold", required=True, type=int, choices=(0, 1))
    score.add_argument("--sealed", action="store_true")
    pool = commands.add_parser("aggregate")
    pool.add_argument("--folds", nargs=2, required=True, type=Path)
    pool.add_argument("--output", required=True, type=Path)
    pool.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    if args.command == "score":
        gather(args)
    else:
        if args.draws < 1:
            parser.error("Use at least one bootstrap draw")
        summarize(args)


if __name__ == "__main__":
    main()
