"""Score frozen ACDC policies on CHLA's released chamber-positive reference slices."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_external_transfer import digest, frozen_protocol, replay_manifest
from audit_verified_negatives import ARMS, reference
from experiment import load_json, log, save_json
from run_sparse_annotation import paired_difference
from segment import configure, predict_many


METRICS = ("recall", "small_dice", "global_dice")
STRATA = {"pooled": (0, 1), "ED": (0,), "ES": (1,)}


def evaluate(args) -> None:
    _, _, _, origin = frozen_protocol(args)
    replay_manifest(args.core_weights, origin)
    other = load_json(args.extra_weights / "source_replay.json")
    if (other["fold"] != args.fold or other["registration_sha256"] != origin["registration_sha256"]
            or other["calibration"] != origin["calibration"]):
        raise ValueError("Supplemental weights differ from frozen source protocol")
    source_audit = load_json(args.source_audit)
    if source_audit["fold"] != args.fold:
        raise ValueError("Source audit belongs to a different fold")

    coverage = load_json(args.external / "coverage.json")
    labels = coverage["labels"]
    if labels not in ([1], [3]):
        raise ValueError("Each published CHLA stack belongs to one ventricle")
    label = labels[0]
    cases, evidence, hashes = {}, {}, {}
    for file in sorted(args.external.glob("*.npz")):
        case = _load_case(file)
        with np.load(file, allow_pickle=False) as stored:
            support = reference(case, stored["native_areas"], stored["verified_absent"], labels)
            if (support["absent"].any() or np.any(stored["native_areas"][:, label - 1] <= 0)):
                raise ValueError(f"A contour-positive cache contains an unverified slice: {file}")
            evidence[file.stem] = support["small"][:, label - 1]
        cases[file.stem] = case
        hashes[file.stem] = digest(file)
    if len(cases) != coverage["patients"] or not all(c.images.shape[1:] == (192, 192) for c in cases.values()):
        raise ValueError("Cache differs from the released cohort or the frozen model input grid")

    result = {}
    device = configure("cuda")
    for arm in ARMS:
        augmented = arm == "pooled_image_farthest"
        archived = (args.baseline_runs if augmented else args.official_original) / f"{arm}.json"
        thresholds = load_json(archived)["thresholds"]
        proof = source_audit["arms"][arm]
        if (proof["source_thr_max_delta"] > 1e-6 or proof["official_metric_max_delta"] > 1e-6
                or any(abs(value - proof["calibration"]["top1"][class_id]["cutoff"]) > 1e-6
                       for class_id, value in thresholds.items())):
            raise ValueError(f"Exact source replay was not verified: {arm}")
        checkpoint = (args.extra_weights if augmented else args.core_weights) / f"{arm}.pt"
        state = SimpleNamespace(teacher=torch.load(checkpoint, map_location="cpu", weights_only=True))
        patient_rows = {}
        for (patient, case), probabilities in zip(cases.items(), predict_many(state, cases.values(), device), strict=True):
            truth = case.masks == label
            predicted = probabilities.argmax(axis=1) == label
            sizes = truth.sum(axis=(1, 2)) + predicted.sum(axis=(1, 2))
            overlap = 2 * (truth & predicted).sum(axis=(1, 2))
            flat = probabilities[:, label].reshape(len(probabilities), -1)
            top = max(1, round(flat.shape[-1] * 0.01))
            detected = np.partition(flat, -top, axis=-1)[:, -top:].mean(axis=1) > thresholds[str(label)]

            rows = {}
            for stratum, phases in STRATA.items():
                within = np.isin(case.phase, phases)
                small = within & evidence[patient]
                rows[stratum] = {"recall": float(detected[small].mean()) if small.any() else None,
                                 "small_dice": float((overlap[small] / sizes[small]).mean()) if small.any() else None,
                                 "global_dice": float(np.mean([overlap[case.phase == p].sum() / sizes[case.phase == p].sum()
                                                                for p in phases])),
                                 "small_count": int(small.sum()), "positive_count": int(within.sum())}
            patient_rows[patient] = rows
        result[arm] = {"source_threshold": thresholds[str(label)], "weight_sha256": digest(checkpoint),
                       "patients": patient_rows}
        log("chla_positive_scored", fold=args.fold, label=label, arm=arm, patients=len(cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"fold": args.fold, "coverage": coverage, "cache_sha256": hashes, "methods": result})


def aggregate(args) -> None:
    folds = [load_json(path) for path in args.folds]
    coverage = folds[0]["coverage"]
    hashes = folds[0]["cache_sha256"]
    if {fold["fold"] for fold in folds} != {0, 1} or any(
            fold["coverage"] != coverage or fold["cache_sha256"] != hashes for fold in folds):
        raise ValueError("Two distinct source folds must use the identical evaluation cache")
    patients = sorted(hashes)
    rng = np.random.default_rng(20261116)
    draws = rng.integers(0, len(patients), size=(4000, len(patients)))
    output = {}
    for stratum in STRATA:
        rows = {}
        reference_support = None
        for arm in ARMS:
            rows[arm] = {}
            for patient in patients:
                observed = [fold["methods"][arm]["patients"][patient][stratum] for fold in folds]
                rows[arm][patient] = {metric: float(np.mean([x[metric] for x in observed]))
                                      if all(x[metric] is not None for x in observed) else None for metric in METRICS}
            support = [(fold["methods"][arm]["patients"][p][stratum]["small_count"],
                        fold["methods"][arm]["patients"][p][stratum]["positive_count"]) for fold in folds for p in patients]
            if support[:len(patients)] != support[len(patients):] or (reference_support is not None and support != reference_support):
                raise ValueError("Reference support varies across folds or policies")
            reference_support = support
        counts = reference_support[:len(patients)]
        methods = {}
        for arm, outcomes in rows.items():
            methods[arm] = {}
            for metric in METRICS:
                values = np.array([outcome[metric] if outcome[metric] is not None else np.nan for outcome in outcomes.values()])
                valid = values[np.isfinite(values)]
                sample = values[draws]
                sample = sample[np.isfinite(sample).any(axis=1)]
                methods[arm][metric] = {"estimate": float(valid.mean()) if len(valid) else None,
                                        "ci95": np.quantile(np.nanmean(sample, axis=1), (.025, .975)).tolist() if len(sample) else None}
        output[stratum] = {"support": {"positive_slices": sum(total for _, total in counts),
                                       "small_slices": sum(small for small, _ in counts),
                                       "small_patients": sum(small > 0 for small, _ in counts)},
                           "methods": methods,
                           "targeted_minus": {arm: {metric: paired_difference(rows["targeted"], rows[arm], metric, draws)
                                                   for metric in METRICS} for arm in ARMS[1:]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"coverage": coverage, "strata": output,
                           "source": {str(fold["fold"]): {arm: {key: fold["methods"][arm][key]
                                       for key in ("source_threshold", "weight_sha256")} for arm in ARMS} for fold in folds},
                           "uncertainty": "Two fixed source fits averaged per patient. Paired bootstrap over CHLA patients only."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("score")
    for field in ("data", "external", "registration", "original", "official-original",
                  "core-weights", "extra-weights", "source-audit", "baseline-runs", "output"):
        run.add_argument(f"--{field}", type=Path, required=True)
    run.add_argument("--fold", type=int, choices=(0, 1), required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--folds", nargs=2, required=True, type=Path)
    combine.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    (evaluate if opts.command == "score" else aggregate)(opts)
