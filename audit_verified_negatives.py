"""Evaluate frozen ACDC models on externally verified slice-class absences."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_decision_curves import interval, matched_recall, patient_histograms
from audit_external_transfer import digest, frozen_protocol, replay_manifest
from experiment import load_json, log, save_json
from segment import configure, predict_many


LABELS = (1, 2, 3)
ARMS = ("targeted", "uncertain", "pooled_image_farthest")
SUBSETS = ("small", "absent", "near_absent")
METRICS = ("small_recall", "fpr", "near_fpr")


def reference(case, native_areas, verified_absent, labels):
    """Keep native label evidence independent of image interpolation."""
    shape = (len(case.images), len(LABELS))
    if (native_areas.shape != shape or native_areas.dtype.kind not in "ui"
            or (native_areas < 0).any() or verified_absent.shape != shape
            or verified_absent.dtype.kind != "b"):
        raise ValueError("Expected native integer areas and boolean absence eligibility per slice/class")
    if (verified_absent & (native_areas > 0)).any():
        raise ValueError("A verified absence has a positive native reference area")
    if verified_absent[:, [label - 1 for label in LABELS if label not in labels]].any():
        raise ValueError("A class outside the verified reference scope was flagged as absent")
    small = np.zeros(shape, dtype=bool)
    near = np.zeros_like(small)
    for phase in np.unique(case.phase):
        mask = case.phase == phase
        areas = native_areas[mask]
        positions = case.slice_index[mask].astype(np.int32)
        for label in labels:
            column = label - 1
            area = areas[:, column]
            small[mask, column] = (area > 0) & (4 * area.astype(np.int64) <= area.max())
            marked = positions[small[mask, column]]
            near[mask, column] = verified_absent[mask, column] & (
                np.isin(positions - 1, marked) | np.isin(positions + 1, marked))
    return dict(zip(SUBSETS, (small, verified_absent, near), strict=True))


def infer(args):
    _, cache, _, source = frozen_protocol(args)
    replay_manifest(args.core_weights, source)
    control = load_json(args.extra_weights / "source_replay.json")
    if (control["fold"] != args.fold or control["registration_sha256"] != source["registration_sha256"]
            or control["calibration"] != source["calibration"]):
        raise ValueError("Supplemental weights differ from the frozen source protocol")
    reviewed = load_json(args.source_audit)
    if reviewed["fold"] != args.fold:
        raise ValueError("Source replay and requested fold differ")
    manifest = load_json(args.coverage)
    labels = tuple(manifest["labels"])
    if not labels or labels != tuple(sorted(set(labels))) or not set(labels) <= set(LABELS):
        raise ValueError("Coverage must declare the original reference labels in ACDC class order")
    cases, annotations, checksums = {}, {}, {}
    official = load_json(args.official_original / "run.json")
    forbidden = set(cache) | set(official["assessment"])
    for path in sorted(args.external.glob("*.npz")):
        if path.stem in forbidden:
            raise ValueError("External patient identifiers must differ from ACDC")
        case = _load_case(path, require_both_phases=False)
        with np.load(path, allow_pickle=False) as data:
            annotations[path.stem] = reference(case, data["native_areas"], data["verified_absent"], labels)
        if case.images.shape[1:] != (192, 192):
            raise ValueError("The frozen ACDC models require 192×192 slice inputs")
        cases[path.stem] = case
        checksums[path.stem] = digest(path)
    if len(cases) != manifest["patients"] or not any(x["absent"].any() for x in annotations.values()):
        raise ValueError("External roster differs from coverage or has no verified absences")
    device = configure("cuda")
    results = {}
    for arm in ARMS:
        archived_path = (args.baseline_runs if arm == "pooled_image_farthest" else args.official_original) / f"{arm}.json"
        cutoffs = load_json(archived_path)["thresholds"]
        replay = reviewed["arms"][arm]
        if (replay["source_thr_max_delta"] > 1e-6 or replay["official_metric_max_delta"] > 1e-6
                or any(abs(value - replay["calibration"]["top1"][label]["cutoff"]) > 1e-6
                       for label, value in cutoffs.items())):
            raise ValueError(f"Exact source replay was not verified for {arm}")
        weight_path = (args.extra_weights if arm == "pooled_image_farthest" else args.core_weights) / f"{arm}.pt"
        teacher = torch.load(weight_path, map_location="cpu", weights_only=True)
        patients = {}
        margins = np.array([cutoffs[str(label)] for label in LABELS], dtype=np.float32)
        for (patient, case), probabilities in zip(cases.items(), predict_many(
                SimpleNamespace(teacher=teacher), cases.values(), device), strict=True):
            flat = probabilities[:, 1:].reshape(len(probabilities), len(LABELS), -1)
            top = max(1, round(flat.shape[-1] * 0.01))
            scores = np.partition(flat, -top, axis=-1)[..., -top:].mean(axis=-1) - margins
            patients[patient] = {
                str(phase): {str(label): {subset: scores[(case.phase == phase) & annotations[patient][subset][:, label - 1],
                                                         label - 1].tolist() for subset in SUBSETS}
                             for label in labels} for phase in np.unique(case.phase)
            }
        results[arm] = {"source_thresholds": cutoffs, "weight_sha256": digest(weight_path), "patients": patients}
        log("verified_absence_scored", fold=args.fold, method=arm, patients=len(cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"fold": args.fold, "coverage": manifest, "cache_sha256": checksums, "methods": results})


def patient_margins(rows, phase, label):
    cells = [cell for p, classes in rows.items() if phase is None or int(p) == phase
             for c, cell in classes.items() if label is None or int(c) == label]
    return {subset: [value for cell in cells for value in cell[subset]] for subset in SUBSETS}


def macro_estimates(groups, weights):
    n = np.array([len(row) for row in groups])
    fractions = np.array([np.count_nonzero(np.asarray(row) > 0) / len(row) if row else 0.0 for row in groups])
    count = weights @ (n > 0)
    return np.divide(weights @ fractions, count, out=np.full(count.shape, np.nan), where=count > 0)


def measurement(values):
    return {"estimate": float(values[0]) if np.isfinite(values[0]) else None, "ci95": interval(values[1:])}


def estimate(folds, patients, weights, phase=None, label=None):
    support = None
    results = {}
    for arm in ARMS:
        fits = []
        for fold in folds:
            patients_by_arm = fold["methods"][arm]["patients"]
            rows = [patient_margins(patients_by_arm[patient], phase, label) for patient in patients]
            groups = {subset: [row[subset] for row in rows] for subset in SUBSETS}
            counts = {subset: [len(row) for row in group] for subset, group in groups.items()}
            if support is not None and counts != support:
                raise ValueError("Reference eligibility differs across models or source folds")
            support = counts
            measures = {metric: macro_estimates(groups[subset], weights)
                        for metric, subset in zip(METRICS, SUBSETS, strict=True)}
            for subset in SUBSETS[1:]:
                if sum(counts["small"]) and sum(counts[subset]):
                    histograms = patient_histograms(groups["small"], groups[subset])
                    measures[f"matched_recall_{subset}_at_20pct"] = matched_recall(histograms, weights, (0.20,))[:, 0]
            fits.append(measures)
        results[arm] = {key: (fits[0][key] + fits[1][key]) / 2 for key in fits[0]}
    return {"support": {subset: {"patients": sum(n > 0 for n in count), "slice_classes": sum(count)}
                        for subset, count in support.items()},
            "methods": {arm: {metric: measurement(value) for metric, value in measures.items()}
                        for arm, measures in results.items()},
            "targeted_minus": {arm: {metric: measurement(results["targeted"][metric] - value)
                                     for metric, value in results[arm].items()} for arm in ARMS[1:]}}


def aggregate(args):
    folds = [load_json(path) for path in args.folds]
    if ({run["fold"] for run in folds} != {0, 1} or folds[0]["coverage"] != folds[1]["coverage"]
            or folds[0]["cache_sha256"] != folds[1]["cache_sha256"]):
        raise ValueError("Scoring must use two distinct frozen folds and the identical external cache")
    patients = sorted(folds[0]["cache_sha256"])
    if any(set(run["methods"][arm]["patients"]) != set(patients) for run in folds for arm in ARMS):
        raise ValueError("Assessment rows must align for paired resampling")
    rng = np.random.default_rng(20261117)

    def resample(roster):
        n = len(roster)
        return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), size=args.draws)])

    weights = resample(patients)
    labels = folds[0]["coverage"]["labels"]
    strata = {"pooled": estimate(folds, patients, weights)}
    for label in labels:
        strata[f"label{label}"] = estimate(folds, patients, weights, label=label)
        if folds[0]["coverage"].get("phase_scope") != "static":
            for phase in (0, 1):
                strata[f"label{label}_phase{phase}"] = estimate(folds, patients, weights, phase=phase, label=label)
    by_reference_method = {}
    if args.reference_groups:
        groups = load_json(args.reference_groups)
        if set(groups) != set(patients):
            raise ValueError("Reference-method groups must cover exactly the evaluated patients")
        for name in sorted(set(groups.values())):
            members = [patient for patient in patients if groups[patient] == name]
            sampled = resample(members)
            by_reference_method[name] = {
                "patients": len(members),
                "strata": {"pooled": estimate(folds, members, sampled),
                           **{f"label{label}": estimate(folds, members, sampled, label=label) for label in labels}},
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {
        "coverage": folds[0]["coverage"], "bootstrap_draws": args.draws,
        "source": "Frozen original ACDC weights and independently archived class-specific thresholds",
        "estimator": "Per-patient rates, averaged over source fits; percentile patient bootstrap. Matched FPR is retrospective.",
        "strata": strata,
        "by_reference_method": by_reference_method,
        "source_weights_and_thresholds": {run["fold"]: {arm: {
            key: run["methods"][arm][key] for key in ("weight_sha256", "source_thresholds")} for arm in ARMS} for run in folds},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("score")
    for field in ("data", "external", "coverage", "registration", "original", "official-original",
                  "core-weights", "extra-weights", "source-audit", "baseline-runs", "output"):
        run.add_argument(f"--{field}", type=Path, required=True)
    run.add_argument("--fold", type=int, choices=(0, 1), required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--folds", nargs=2, required=True, type=Path)
    combine.add_argument("--output", required=True, type=Path)
    combine.add_argument("--draws", type=int, default=4000)
    combine.add_argument("--reference-groups", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        infer(args)
    elif args.draws < 1:
        parser.error("--draws must be positive")
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
