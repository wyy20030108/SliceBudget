"""Audit held-out ACDC recall at retrospectively matched false-positive rates."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import load_cases
from audit_external_transfer import frozen_protocol, replay_manifest
from experiment import load_json, log, save_json
from run_matched_slice_budget import collect, rates
from segment import configure, predict_many
from slice_metrics import patient_metrics


OPERATING_POINTS = (0.05, 0.10, 0.20, 0.30, 0.40)
NEGATIVE_GROUPS = ("absent", "near_absent")
PRESENCE_METRICS = ("recall", "fpr", "near_fpr")


def score(args):
    settings, cache, _, source = frozen_protocol(args)
    manifest = replay_manifest(args.weights, source)
    cases = load_cases(args.assessment)
    official = load_json(args.official_original / "run.json")
    if (len(cases) != 50 or set(cases) != set(official["assessment"]) or set(cases) & cache.keys()
            or not official["held_out_test_used"]):
        raise ValueError("Use the disjoint 50-patient official ACDC test cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    device = configure("cuda")
    for arm in settings["methods"]:
        checkpoint = args.weights / f"{arm}.pt"
        replay = manifest["threshold_replay"][arm]
        if replay["max_abs_delta_from_original"] > 0.001:
            raise ValueError(f"Checkpoint failed threshold replay: {arm}")
        teacher = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = SimpleNamespace(teacher=teacher)
        if args.command == "argmax":
            outcomes = {}
            for (patient, case), probabilities in zip(cases.items(), predict_many(model, cases.values(), device),
                                                      strict=True):
                result = patient_metrics(case, probabilities.argmax(axis=1))
                miss = result["small_miss_rate"]
                outcomes[patient] = {
                    "recall": None if miss is None else 1 - miss,
                    "fpr": result["all_absent_false_positive_rate"],
                    "near_fpr": result["bordering_absent_false_positive_rate"],
                }
            save_json(args.output / f"{arm}.argmax.json", outcomes)
            log("argmax_presence_ready", arm=arm, fold=args.fold)
            continue
        records = collect(cases, model, device)
        thresholds = source["source_thresholds"][arm]
        observed = rates(records, thresholds)
        original = load_json(args.official_original / f"{arm}.json")["assessment"]
        for patient, outcomes in observed.items():
            for metric, value in outcomes.items():
                expected = original[patient][metric]
                if value is None and expected is None:
                    continue
                if value is None or expected is None or abs(value - expected) > 1e-6:
                    raise ValueError(f"Replay differs from the original at {arm}, {patient}, {metric}")
        rows = {
            patient: {
                subset: [
                    float(value) - thresholds[label]
                    for label, group in groups.items() for value in group[subset]
                ] for subset in ("small", *NEGATIVE_GROUPS)
            } for patient, (groups, _, _) in records.items()
        }
        save_json(args.output / f"{arm}.json", rows)
        log("decision_scores_ready", arm=arm, fold=args.fold)
    if args.command == "score":
        save_json(args.output / "roster.json",
                  {"fold": args.fold, "methods": settings["methods"], "patients": list(cases)})


def patient_histograms(positive, negative):
    """Create patient-normalized event histograms on a shared descending score grid."""
    values = np.unique(np.concatenate(positive + negative))[::-1]
    histograms = []
    for group in (positive, negative):
        histogram = np.zeros((len(group), len(values)))
        for index, scores in enumerate(group):
            if scores:
                ranks = np.searchsorted(-values, -np.asarray(scores))
                histogram[index] = np.bincount(ranks, minlength=len(values)) / len(scores)
        histograms.append(histogram)
    return tuple(histograms)


def matched_recall(histograms, patient_weights, fprs):
    """Vary a common score offset and interpolate only between adjacent ROC points."""
    numerators = tuple(
        (patient_weights @ histogram).cumsum(axis=1)
        for histogram in histograms
    )
    curves = []
    for positive, negative in zip(*numerators, strict=True):
        if positive[-1] == 0 or negative[-1] == 0:
            curves.append(np.full(len(fprs), np.nan))
            continue
        recall = np.r_[0.0, positive / positive[-1]]
        fpr = np.r_[0.0, negative / negative[-1]]
        result = []
        for target in fprs:
            # Keep the best recall attainable without exceeding target when
            # multiple consecutive thresholds realize the same empirical FPR.
            upper = np.searchsorted(fpr, target + 1e-12, side="right")
            lower = upper - 1
            if fpr[lower] >= target - 1e-12:
                result.append(recall[lower])
            else:
                fraction = (target - fpr[lower]) / (fpr[upper] - fpr[lower])
                result.append(recall[lower] + fraction * (recall[upper] - recall[lower]))
        curves.append(result)
    return np.asarray(curves)


def interval(values):
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None


def positive_rate(values):
    return float(np.mean(np.asarray(values) > 0)) if values else None


def summarize_presence(arm_folds, patients, bootstrap):
    results = {}
    for arm, folds in arm_folds.items():
        data = np.array([[[fold[patient][metric] for metric in PRESENCE_METRICS]
                          for patient in patients] for fold in folds], dtype=float)
        # Support for a metric is defined by a patient's reference mask, so it
        # agrees across folds and methods.
        if not np.array_equal(np.isfinite(data[0]), np.isfinite(data[1])):
            raise ValueError(f"Presence support differs by fold: {arm}")
        results[arm] = data.mean(axis=0)
    methods = {arm: {metric: float(np.nanmean(values[:, index]))
                     for index, metric in enumerate(PRESENCE_METRICS)} for arm, values in results.items()}
    contrasts = {}
    for arm, values in results.items():
        if arm == "targeted":
            continue
        contrasts[arm] = {}
        for index, metric in enumerate(PRESENCE_METRICS):
            differences = results["targeted"][:, index] - values[:, index]
            present = np.isfinite(differences)
            weights = bootstrap[:, present]
            support = weights.sum(axis=1)
            simulated = ((weights @ differences[present])[support > 0] / support[support > 0])
            contrasts[arm][metric] = {"difference": float(np.mean(differences[present])),
                                      "descriptive_95_interval": interval(simulated)}
    return {"methods": methods, "comparisons": contrasts}


def aggregate(folds, output, draws):
    manifests = [load_json(path / "roster.json") for path in folds]
    if sorted(item["fold"] for item in manifests) != [0, 1]:
        raise ValueError("Provide the two source training folds")
    first = manifests[0]
    patients = first["patients"]
    methods = first["methods"]
    if len(patients) != 50 or any(
            item["patients"] != patients or item["methods"] != methods for item in manifests):
        raise ValueError("Folds must share 50 consistently ordered patients and all methods")
    random = np.random.default_rng(20261119)
    bootstrap = random.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=draws)
    assignments = np.vstack([np.ones(len(patients)), bootstrap])
    scores = {arm: [load_json(path / f"{arm}.json") for path in folds] for arm in methods}
    zero = {
        arm: [{patient: {metric: positive_rate(row[subset]) for metric, subset in
                        zip(PRESENCE_METRICS, ("small", *NEGATIVE_GROUPS), strict=True)}
               for patient, row in records.items()} for records in records_by_fold]
        for arm, records_by_fold in scores.items()
    }
    results = {
        "cohort": "official ACDC test",
        "n_patients": len(patients),
        "score": "mean top 1 percent in-plane class probability minus fold/method/class ACDC calibration cutoff",
        "interpretation": "Retrospective discrimination audit. Test labels determine matched operating points.",
        "pooling": "Each patient contributes equal weight within each outcome. Calculate ROC within each "
                   "fixed source fold and average recall across folds at the same FPR. Re-estimate both "
                   "curves in every shared paired patient bootstrap draw.",
        "bootstrap_draws": draws,
        "false_positive_rates": list(OPERATING_POINTS),
        "original_source_operating_point": summarize_presence(zero, patients, bootstrap),
        "groups": {},
    }
    argmax_files = [path / f"{arm}.argmax.json" for path in folds for arm in methods]
    if all(path.exists() for path in argmax_files):
        argmax = {arm: [load_json(path / f"{arm}.argmax.json") for path in folds] for arm in methods}
        results["any_argmax_pixel_operating_point"] = summarize_presence(argmax, patients, bootstrap)
    for group in NEGATIVE_GROUPS:
        arm_curves = {}
        arm_rates = {}
        support = {}
        for arm in methods:
            curves = []
            by_fold = {}
            for path, manifest, rows in zip(folds, manifests, scores[arm], strict=True):
                if set(rows) != set(patients):
                    raise ValueError(f"Score output does not match the official patient roster: {path}/{arm}")
                positives = [rows[patient]["small"] for patient in patients]
                negatives = [rows[patient][group] for patient in patients]
                support[manifest["fold"]] = {
                    "small_patients": sum(map(bool, positives)),
                    "absent_patients": sum(map(bool, negatives)),
                    "small_slice_classes": sum(map(len, positives)),
                    "absent_slice_classes": sum(map(len, negatives)),
                }
                curve = matched_recall(patient_histograms(positives, negatives), assignments, OPERATING_POINTS)
                by_fold[manifest["fold"]] = curve[0].tolist()
                curves.append(curve)
            arm_curves[arm] = np.mean(curves, axis=0)
            arm_rates[arm] = {"recall": arm_curves[arm][0].tolist(), "recall_by_fold": by_fold}
        contrasts = {}
        for arm, recall in arm_curves.items():
            if arm == "targeted":
                continue
            differences = arm_curves["targeted"] - recall
            contrasts[arm] = {
                "targeted_minus_baseline": differences[0].tolist(),
                "descriptive_95_intervals": [interval(values) for values in differences[1:].T],
            }
        results["groups"][group] = {"support": support, "methods": arm_rates, "comparisons": contrasts}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("score", "argmax"):
        scoring = commands.add_parser(command)
        for path in ("data", "registration", "original", "official-original", "assessment", "weights", "output"):
            scoring.add_argument(f"--{path}", type=Path, required=True)
        scoring.add_argument("--fold", type=int, choices=(0, 1), required=True)
    analysis = commands.add_parser("aggregate")
    analysis.add_argument("--folds", type=Path, nargs=2, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    if args.command in ("score", "argmax"):
        score(args)
    else:
        if args.draws < 1:
            parser.error("Supply at least one bootstrap draw")
        aggregate(args.folds, args.output, args.draws)


if __name__ == "__main__":
    main()
