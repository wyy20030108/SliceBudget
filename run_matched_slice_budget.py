"""Cross-fitted study of sparse supervision with independently calibrated presence scores."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from acdc_data import _load_case, make_protocol
from experiment import load_json, log, save_json, scores
from matched_slice_budget import allocate_cohort
from run_sparse_annotation import paired_difference
from segment import configure, predict_many
from sparse_train import train_sparse


def split(cache, settings, fold):
    sealed = set(make_protocol(cache, settings["holdout_protocol_seed"])["test"])
    roster = np.random.default_rng(settings["development_seed"]).permutation(sorted(cache.keys() - sealed)).tolist()
    blocks = np.array_split(roster, settings["folds"])
    opposite = np.concatenate([block for index, block in enumerate(blocks) if index != fold]).tolist()
    size = settings["calibration_patients_per_fold"]
    return blocks[fold].tolist(), opposite[:size], opposite[size:]


def collect(cases, state, device):
    patients = {}
    for (patient, case), probabilities in zip(cases.items(), predict_many(state, cases.values(), device), strict=True):
        flat = probabilities[:, 1:].reshape(len(probabilities), 3, -1)
        top = max(1, round(flat.shape[-1] / 100))
        evidence = np.partition(flat, -top, axis=-1)[..., -top:].mean(axis=-1)
        records = {}
        for label in (1, 2, 3):
            area = np.count_nonzero(case.masks == label, axis=(1, 2))
            small = np.zeros(len(area), dtype=bool)
            neighboring = np.zeros(len(area), dtype=bool)
            for phase in (0, 1):
                in_phase = case.phase == phase
                small |= in_phase & (area > 0) & (4 * area <= area[in_phase].max())
                positions = case.slice_index[in_phase].astype(int)
                marked = case.slice_index[in_phase & small].astype(int)
                neighboring[in_phase] = np.isin(positions - 1, marked) | np.isin(positions + 1, marked)
            s = evidence[:, label - 1]
            records[label] = {"small": s[small], "absent": s[area == 0], "near_absent": s[(area == 0) & neighboring]}
        masks = scores(case, probabilities.argmax(axis=1))
        patients[patient] = (records, masks["global_dice"], masks["hard_dice"])
    return patients


def choose_threshold(groups, target):
    nonempty = [row for row in groups if len(row)]
    if not nonempty:
        raise ValueError("Calibration cohort lacks truly absent examples")
    values = np.concatenate(nonempty)
    weights = np.concatenate([np.full(len(row), 1 / len(row)) for row in nonempty])
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cutoff = np.searchsorted(cumulative, (1 - target) * cumulative[-1], side="left")
    return float(values[order][cutoff])


def calibrate(records, target):
    by_class = {label: [items[label]["absent"] for items, _, _ in records.values() if len(items[label]["absent"])]
                for label in (1, 2, 3)}
    pooled = [np.concatenate([group["absent"] for group in items.values()]) for items, _, _ in records.values()]
    thresholds = {label: choose_threshold(values if len(values) >= 5 else pooled, target)
                  for label, values in by_class.items()}
    return thresholds, {label: len(values) for label, values in by_class.items()}


def rates(records, thresholds):
    outcomes = {}
    for patient, (items, overall, small_dice) in records.items():
        row = {"global_dice": overall, "small_dice": small_dice}
        for subset, metric in (("small", "recall"), ("absent", "fpr"), ("near_absent", "near_fpr")):
            decisions = np.concatenate([group[subset] > thresholds[label] for label, group in items.items()])
            row[metric] = float(decisions.mean()) if len(decisions) else None
            row[f"count_{subset}"] = int(len(decisions))
        outcomes[patient] = row
    return outcomes


def aggregate(records):
    result = {}
    for key in next(iter(records.values())):
        values = [row[key] for row in records.values() if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def partial_auc(records, absent="absent"):
    labels, scores_, weights = [], [], []
    for groups, _, _ in records.values():
        for outcome, subset in ((1, "small"), (0, absent)):
            values = np.concatenate([group[subset] for group in groups.values()])
            if len(values):
                labels.extend([outcome] * len(values))
                scores_.extend(values)
                weights.extend([1 / len(values)] * len(values))
    return float(roc_auc_score(labels, scores_, sample_weight=weights, max_fpr=0.4))


def pool_seeds(seed_results, arms):
    pooled = {}
    first = next(iter(seed_results.values()))
    for arm in arms:
        pooled[arm] = {}
        for patient, row in first[arm].items():
            pooled[arm][patient] = {}
            for key in row:
                observed = [seed[arm][patient][key] for seed in seed_results.values() if seed[arm][patient][key] is not None]
                pooled[arm][patient][key] = float(np.mean(observed)) if observed else None
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output", "registration"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 1))
    parser.add_argument("--official-audit", type=Path, help="Optional, distinct official ACDC challenge test cache")
    args = parser.parse_args()
    settings = load_json(args.registration)
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    if len(cache) != 100:
        parser.error("Use the official 100-patient training set")
    training, calibration, assessment = split(cache, settings, args.fold)
    assessment_cache = cache
    if args.official_audit is not None:
        assessment_cache = {item.stem: item for item in args.official_audit.glob("patient*.npz")}
        if len(assessment_cache) != 50 or set(assessment_cache).intersection(cache):
            parser.error("Official audit requires 50 test patients disjoint from the 100 training cases")
        assessment = sorted(assessment_cache)
    args.output.mkdir(parents=True)
    device = configure("cuda")
    train_cases, calibration_cases = (
        {patient: _load_case(cache[patient]) for patient in group} for group in (training, calibration))
    assessment_cases = {patient: _load_case(assessment_cache[patient]) for patient in assessment}
    save_json(args.output / "run.json", {"fold": args.fold, "training": training, "calibration": calibration,
                                          "assessment": assessment, "held_out_test_used": args.official_audit is not None,
                                          "assessment_release": "official challenge test" if args.official_audit else "development",
                                          "pytorch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                                          "registration_sha256": hashlib.sha256(args.registration.read_bytes()).hexdigest()})
    method_results = {}
    for seed in settings["training_seeds"]:
        allocations, ledger = allocate_cohort(train_cases, seed)
        save_json(args.output / f"allocation-{seed}.json", {"arms": allocations, "ledger": ledger})
        results = {}
        for arm in settings["methods"]:
            selection = allocations[arm]
            if any(len(set(indices)) != 2 * settings["dense_slices_per_phase"] for indices in selection.values()):
                raise ValueError("Annotation budget mismatch")
            model = train_sparse(train_cases, selection, [], steps=settings["updates"], seed=seed,
                                 device=device, batch_size=settings["batch_size"])
            calibrated = collect(calibration_cases, model, device)
            thresholds, support = calibrate(calibrated, settings["target_calibration_fpr"])
            observed = collect(assessment_cases, model, device)
            results[arm] = rates(observed, thresholds)
            save_json(args.output / f"{seed}-{arm}.json", {"assessment": results[arm], "thresholds": thresholds,
                                                            "absent_calibration_patients_per_class": support,
                                                            "calibration": aggregate(rates(calibrated, thresholds)),
                                                            "assessment_pauc": {"all_absent": partial_auc(observed),
                                                                                "near_absent": partial_auc(observed, "near_absent")},
                                                            "assessment_means": aggregate(results[arm])})
            log("matched_report_complete", fold=args.fold, seed=seed, arm=arm, results=aggregate(results[arm]))
        method_results[seed] = results
    pooled = pool_seeds(method_results, settings["methods"])
    draws = np.random.default_rng(args.fold + settings["development_seed"]).integers(0, len(assessment), (4000, len(assessment)))
    comparisons = {}
    for arm in ("interval", "positive"):
        differences = {metric: paired_difference(pooled["balanced"], pooled[arm], metric, draws)
                       for metric in ("recall", "fpr", "near_fpr", "global_dice", "small_dice")}
        passes = (differences["recall"]["difference"] >= settings["minimum_small_recall_increase"] and
                  differences["fpr"]["difference"] <= settings["maximum_evaluation_fpr_increase"] and
                  differences["global_dice"]["difference"] >= -settings["maximum_global_dice_drop"])
        comparisons[arm] = {"passed": passes, "paired": differences}
    save_json(args.output / "summary.json", {"methods": {arm: aggregate(data) for arm, data in pooled.items()},
                                             "comparisons": comparisons, "held_out_test_used": args.official_audit is not None})


if __name__ == "__main__":
    main()
