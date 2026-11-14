"""Factorial audit of selection information and allocation on one shared protocol."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from acdc_data import _load_case
from experiment import load_json, log, save_json
from image_slice_budget import balanced_assign
from run_image_slice_budget import image_view, patient_roster, selection_quality
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from segment import configure
from sparse_train import train_sparse


def truth_options(case, indices, available):
    order = case.slice_index.astype(int)
    options = {}
    for label in (1, 2, 3):
        area = (case.masks == label).sum(axis=(1, 2))
        positive = indices[area[indices] > 0]
        if not len(positive):
            continue
        peak = area[positive].max()
        small = [index for index in available if 0 < 4 * area[index] <= peak]
        absent = [index for index in available if area[index] == 0]
        if small:
            highest = min(small, key=lambda row: (-area[row], order[row]))
            options[(label, "small")] = (int(highest), float(4 * area[highest] / peak))
        if absent:
            distance = {row: np.abs(order[row] - order[positive]).min() for row in absent}
            closest = min(absent, key=lambda row: (distance[row], order[row]))
            options[(label, "absent")] = (int(closest), float(1 / np.sqrt(distance[closest])))
    return options


def make_choices(initial, original, training, seed):
    patients = sorted(initial)
    rng = np.random.default_rng(seed)
    methods = {name: {p: list(initial[p]) for p in patients} for name in ("truth_balanced", "mixed_image")}
    strata = {}
    for phase in (0, 1):
        options = []
        use_targeting = set(rng.choice(patients, len(patients) // 2, replace=False))
        for patient in patients:
            case = training[patient]
            indices = np.flatnonzero(case.phase == phase)
            available = [index for index in indices if index not in initial[patient]]
            options.append(truth_options(case, indices, available))
            inherited = original["targeted" if patient in use_targeting else "interval"][patient]
            mixed = next(index for index in inherited if case.phase[index] == phase and index not in initial[patient])
            methods["mixed_image"][patient].append(mixed)
        strata[str(phase)] = {}
        for patient, key, available in zip(patients, balanced_assign(options), options, strict=True):
            case = training[patient]
            interval = next(index for index in original["interval"][patient]
                            if case.phase[index] == phase and index not in initial[patient])
            methods["truth_balanced"][patient].append(available[key][0] if key is not None else interval)
            strata[str(phase)][patient] = key
    return {arm: {p: sorted(values) for p, values in roster.items()} for arm, roster in methods.items()}, strata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "official-audit", "registration", "prior", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    settings = load_json(args.registration)
    training_cache = {p.stem: p for p in args.data.glob("patient*.npz")}
    audit_cache = {p.stem: p for p in args.official_audit.glob("patient*.npz")}
    if len(audit_cache) != 50 or len(training_cache) != 100 or set(training_cache).intersection(audit_cache):
        parser.error("Use the official, separate 100 and 50 patient ACDC caches")
    train_ids, cal_ids, assessment_ids = patient_roster(training_cache, settings, args.fold)
    reference = load_json(args.prior / "run.json")
    frozen = load_json(args.prior / "frozen_acquisitions.json")
    if reference["training"] != train_ids or reference["calibration"] != cal_ids or reference["held_out_test_used"]:
        parser.error("The original development assignment must match")
    args.output.mkdir(parents=True)
    truth = {patient: _load_case(training_cache[patient]) for patient in train_ids}
    initial = {patient: frozen["warmup"][patient] for patient in train_ids}
    alternatives, ledger = make_choices(initial, frozen["arms"], truth, settings["training_seed"] + args.fold)
    for arm in alternatives:
        for p, indices in alternatives[arm].items():
            if len(set(indices)) != 4 or set(initial[p]) - set(indices):
                raise ValueError("Budget or initial assignment mismatch")
    save_json(args.output / "audit_plan.json", {"choices": alternatives, "truth_strata": ledger,
                                               "registration_sha256": hashlib.sha256(args.registration.read_bytes()).hexdigest(),
                                               "note": "Freeze both arms before reading official references"})
    rng_seed = settings["training_seed"]
    device = configure("cuda")
    options = {"seed": rng_seed, "device": device, "batch_size": settings["batch_size"]}
    common = train_sparse(truth, initial, [], steps=settings["warmup_updates"], **options)
    calibration = {p: _load_case(training_cache[p]) for p in cal_ids}
    assessments = {"development": {p: _load_case(training_cache[p]) for p in assessment_ids},
                   "official": {p: _load_case(audit_cache[p]) for p in sorted(audit_cache)}}
    results = {}
    for method, selection in alternatives.items():
        # The base runner constructs dictionaries in sorted order for these targeted arms.
        trained = train_sparse(truth, selection, [], steps=settings["additional_updates"], start=common, **options)
        threshold, support = calibrate(collect(calibration, trained, device), settings["calibration_false_positive_rate"])
        selection_audit = selection_quality(truth, selection, initial)
        results[method] = {}
        for cohort, cases in assessments.items():
            raw = collect(cases, trained, device)
            patient_data = rates(raw, threshold)
            means = aggregate(patient_data)
            results[method][cohort] = means
            save_json(args.output / f"{method}-{cohort}.json", {"assessment": patient_data,
                                                            "assessment_pauc": {"all_absent": partial_auc(raw),
                                                                                "near_absent": partial_auc(raw, "near_absent")},
                                                            "thresholds": threshold, "support": support,
                                                            "acquisition_audit": selection_audit})
            log("allocation_audit_complete", fold=args.fold, method=method, cohort=cohort, means=means)
    save_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
