"""Factorial sensitivity audit of frozen slice acquisition and the DiffUnder calibration code path."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from acdc_data import _load_case
from experiment import load_json, log, save_json
from run_image_slice_budget import patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from run_sparse_annotation import paired_difference
from segment import configure
from sparse_train import train_sparse


POLICIES = ("uncertain", "targeted")
MODES = ("off", "on")
METRICS = ("global_dice", "small_dice", "recall", "fpr", "near_fpr")


def mask_occupancy(training, selection):
    """Describe class occupancy in acquired masks and the entire retrospective training pool."""
    groups = {}
    for phase in (0, 1):
        for label in (1, 2, 3):
            result = {population: {"slices": 0, "absent": 0, "small": 0, "mean_foreground_area_fraction": 0.0}
                      for population in ("acquired", "pool")}
            for patient, case in training.items():
                indices = np.flatnonzero(case.phase == phase)
                masks = case.masks[indices]
                area = np.count_nonzero(masks == label, axis=(1, 2))
                small = (area > 0) & (4 * area <= area.max())
                chosen = np.isin(indices, selection[patient])
                for population, subset in (("pool", np.ones(len(indices), dtype=bool)), ("acquired", chosen)):
                    row = result[population]
                    row["slices"] += int(subset.sum())
                    row["absent"] += int((area[subset] == 0).sum())
                    row["small"] += int(small[subset].sum())
                    row["mean_foreground_area_fraction"] += float(area[subset].sum() / np.prod(masks.shape[1:]))
            for row in result.values():
                row["mean_foreground_area_fraction"] /= row["slices"]
            groups[f"phase{phase}_class{label}"] = result
    return groups


def verify_reference(origin, expected, train_ids, cal_ids, audit_ids, sha):
    if origin["training"] != train_ids or origin["calibration"] != cal_ids or origin["held_out_test_used"]:
        raise ValueError("Provide the corresponding original development run")
    if expected["training"] != train_ids or expected["calibration"] != cal_ids or expected["assessment"] != audit_ids:
        raise ValueError("Official reference must have the identical training, calibration and audit rosters")
    if not expected["held_out_test_used"] or any(run["registration_sha256"] != sha for run in (origin, expected)):
        raise ValueError("Use the frozen registration and official reference run")


def maximum_replay_delta(observed, expected):
    if observed.keys() != expected.keys():
        raise ValueError("Replayed subjects differ from reference")
    deltas = []
    for patient, row in observed.items():
        for metric in METRICS:
            before, after = expected[patient][metric], row[metric]
            if before is None and after is None:
                continue
            if before is None or after is None:
                raise ValueError(f"Missing outcome mismatch for {metric}")
            deltas.append(abs(before - after))
    return max(deltas)


def study_summary(runs):
    first = runs[0]
    pooled = {}
    for cell, patients in first.items():
        if any(run[cell].keys() != patients.keys() for run in runs):
            raise ValueError("Pair the same official test patients across folds")
        pooled[cell] = {}
        for patient in sorted(patients):
            row = pooled[cell][patient] = {}
            for metric in METRICS:
                values = [run[cell][patient][metric] for run in runs]
                if any(value is None for value in values) and not all(value is None for value in values):
                    raise ValueError("Inconsistent metric eligibility across folds")
                row[metric] = None if values[0] is None else float(np.mean(values))
    size = len(pooled["uncertain_off"])
    draws = np.random.default_rng(20261120).integers(0, size, (4000, size))

    def contrast(lhs, rhs):
        return {metric: paired_difference(lhs, rhs, metric, draws) for metric in METRICS}

    def change(lhs, rhs):
        return {p: {k: None if row[k] is None or rhs[p][k] is None else row[k] - rhs[p][k] for k in METRICS}
                for p, row in lhs.items()}

    off_policy = contrast(pooled["targeted_off"], pooled["uncertain_off"])
    on_policy = contrast(pooled["targeted_on"], pooled["uncertain_on"])
    uncertain_change = change(pooled["uncertain_on"], pooled["uncertain_off"])
    targeted_change = change(pooled["targeted_on"], pooled["targeted_off"])
    return {"patients": size, "means": {cell: aggregate(data) for cell, data in pooled.items()},
            "targeted_minus_uncertain_off": off_policy, "targeted_minus_uncertain_on": on_policy,
            "uot_effect_uncertain": contrast(pooled["uncertain_on"], pooled["uncertain_off"]),
            "uot_effect_targeted": contrast(pooled["targeted_on"], pooled["targeted_off"]),
            "interaction_targeted_minus_uncertain": contrast(targeted_change, uncertain_change)}


def run(args):
    settings = load_json(args.registration)
    registration_hash = sha256(args.registration.read_bytes()).hexdigest()
    train_cache = {p.stem: p for p in args.data.glob("patient*.npz")}
    audit_cache = {p.stem: p for p in args.official_audit.glob("patient*.npz")}
    if len(train_cache) != 100 or len(audit_cache) != 50 or train_cache.keys() & audit_cache.keys():
        raise ValueError("Use the distinct official train and test ACDC releases")
    train_ids, cal_ids, _ = patient_roster(train_cache, settings, args.fold)
    roster = sorted(audit_cache)
    origin = load_json(args.original / "run.json")
    expected = load_json(args.official_original / "run.json")
    verify_reference(origin, expected, train_ids, cal_ids, roster, registration_hash)
    frozen_path = args.original / "frozen_acquisitions.json"
    frozen = load_json(frozen_path)
    if frozen != load_json(args.official_original / frozen_path.name):
        raise ValueError("Development and official acquisition ledgers differ")
    initial = {patient: frozen["warmup"][patient] for patient in train_ids}
    training = {patient: _load_case(train_cache[patient]) for patient in train_ids}
    calibration = {patient: _load_case(train_cache[patient]) for patient in cal_ids}
    assessment = {patient: _load_case(audit_cache[patient]) for patient in roster}
    args.output.mkdir(parents=True)
    device = configure("cuda")
    parameters = {"seed": settings["training_seed"], "device": device, "batch_size": settings["batch_size"]}
    common = train_sparse(training, initial, [], steps=settings["warmup_updates"], **parameters)
    record = {"fold": args.fold, "training": train_ids, "calibration": cal_ids, "assessment": roster,
              "registration_sha256": registration_hash, "acquisition_sha256": sha256(frozen_path.read_bytes()).hexdigest(),
              "pytorch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
              "note": "Post-hoc fixed official ACDC audit, with the existing dense DiffUnder calibration code path", "cells": {}}
    for policy in POLICIES:
        # Preserve the original saved insertion order for the deterministic training sampler.
        selection = frozen["arms"][policy]
        if set(selection) != set(initial) or any(len(set(indices)) != 4 or not set(initial[p]) <= set(indices)
                                                for p, indices in selection.items()):
            raise ValueError("A policy must preserve its shared warmup and four-mask budget")
        record["cells"][policy] = {"mask_occupancy": mask_occupancy(training, selection)}
        for mode in MODES:
            branch = train_sparse(training, selection, [], steps=settings["additional_updates"], start=common,
                                  uot=mode == "on", **parameters)
            thresholds, support = calibrate(collect(calibration, branch, device), settings["calibration_false_positive_rate"])
            scores = collect(assessment, branch, device)
            evaluated = rates(scores, thresholds)
            evaluation = {"assessment": evaluated, "means": aggregate(evaluated), "thresholds": thresholds,
                          "calibration_support": support, "prior_end": branch.prior.tolist(),
                          "pseudo_fraction": branch.pseudo_fraction, "continuation_seconds": branch.seconds,
                          "assessment_pauc": {"all_absent": partial_auc(scores),
                                              "near_absent": partial_auc(scores, "near_absent")}}
            if mode == "off":
                reference = load_json(args.official_original / f"{policy}.json")
                evaluation["reference_max_abs_delta"] = maximum_replay_delta(evaluated, reference["assessment"])
                evaluation["reference_threshold_max_abs_delta"] = max(abs(thresholds[label] - reference["thresholds"][str(label)])
                                                                      for label in thresholds)
            cell = f"{policy}_{mode}"
            record["cells"][policy][mode] = {k: v for k, v in evaluation.items() if k != "assessment"}
            save_json(args.output / f"{cell}.json", evaluation)
            log("prcv_bridge_completed", fold=args.fold, policy=policy, calibration=mode, means=evaluation["means"],
                pseudo_fraction=evaluation["pseudo_fraction"], reference_max_abs_delta=evaluation.get("reference_max_abs_delta"))
    save_json(args.output / "summary.json", record)


def combine(args):
    records = [load_json(path / "summary.json") for path in args.folds]
    if {row["fold"] for row in records} != {0, 1} or len({row["registration_sha256"] for row in records}) != 1:
        raise ValueError("Aggregate the two preregistered folds")
    outcomes = [{f"{policy}_{mode}": load_json(path / f"{policy}_{mode}.json")["assessment"]
                 for policy in POLICIES for mode in MODES} for path in args.folds]
    results = study_summary(outcomes)
    results["folds"] = [{"fold": row["fold"], "cells": row["cells"]} for row in records]
    save_json(args.output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    for name in ("data", "official-audit", "registration", "original", "official-original", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--fold", type=int, required=True, choices=(0, 1))
    merge = commands.add_parser("aggregate")
    merge.add_argument("--folds", required=True, type=Path, nargs=2)
    merge.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        combine(args)


if __name__ == "__main__":
    main()
