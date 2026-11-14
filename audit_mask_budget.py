"""Test whether the frozen acquisition comparisons persist at 240 training masks."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from acdc_data import _load_case
from experiment import load_json, log, save_json
from run_image_slice_budget import image_view, patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from segment import configure
from sparse_train import train_sparse


ARMS = ("interval", "uncertain", "targeted")


def expand(cases, initial, frozen, budget):
    """Add one common spatially distributed slice per patient and phase."""
    masks_per_patient, extra = {160: (4, False), 240: (6, True)}[budget]
    if len(cases) * masks_per_patient != budget:
        raise ValueError("Expected the frozen 40-patient training cohort")
    if any(set(frozen[arm]) != set(cases) for arm in ARMS) or set(initial) != set(cases):
        raise ValueError("Frozen acquisition policies must match the original training patients")

    plans = {arm: {p: list(indices) for p, indices in frozen[arm].items()} for arm in ARMS}
    extra_slices = {p: [] for p in cases}
    for patient, case in cases.items():
        for arm in ARMS:
            positions = plans[arm][patient]
            if (len(set(positions)) != 4 or set(initial[patient]) - set(positions)
                or np.bincount(case.phase[positions], minlength=2).tolist() != [2, 2]):
                raise ValueError(f"Invalid frozen four-slice policy for {patient}")
        if not extra:
            continue
        union = set().union(*(plans[arm][patient] for arm in ARMS))
        for phase in (0, 1):
            available = sorted(set(np.flatnonzero(case.phase == phase)) - union)
            if not available:
                raise ValueError(f"No common additional slice remains for {patient}, phase {phase}")
            chosen = [i for i in union if case.phase[i] == phase]
            taken = case.slice_index[chosen].astype(int)
            extra_slices[patient].append(int(max(available, key=lambda i: (
                int(np.abs(int(case.slice_index[i]) - taken).min()), -int(case.slice_index[i])))))
    return ({arm: {p: sorted([*positions, *extra_slices[p]]) for p, positions in plan.items()}
             for arm, plan in plans.items()}, extra_slices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "official-audit", "registration", "original", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--budget", type=int, default=240, choices=(160, 240))
    args = parser.parse_args()
    settings = load_json(args.registration)
    cache = {p.stem: p for p in args.data.glob("patient*.npz")}
    audit = {p.stem: p for p in args.official_audit.glob("patient*.npz")}
    if len(cache) != 100 or len(audit) != 50 or cache.keys() & audit.keys():
        parser.error("Use the distinct ACDC challenge releases")
    train_ids, cal_ids, _ = patient_roster(cache, settings, args.fold)
    origin = load_json(args.original / "run.json")
    if origin["training"] != train_ids or origin["calibration"] != cal_ids or origin["held_out_test_used"]:
        parser.error("Provide the corresponding development run")
    registration_hash = sha256(args.registration.read_bytes()).hexdigest()
    if origin["registration_sha256"] != registration_hash:
        parser.error("Development run used a different registration")
    frozen = load_json(args.original / "frozen_acquisitions.json")
    images = {p: image_view(cache[p]) for p in train_ids}
    initial = {p: frozen["warmup"][p] for p in train_ids}
    plans, added = expand(images, initial, frozen["arms"], args.budget)
    args.output.mkdir(parents=True)
    save_json(args.output / "audit_plan.json", {"selections": plans, "common_additions": added,
                                               "training_budget": args.budget, "registration_sha256": registration_hash})

    training = {p: _load_case(cache[p]) for p in train_ids}
    calibration = {p: _load_case(cache[p]) for p in cal_ids}
    assessment = {p: _load_case(audit[p]) for p in sorted(audit)}
    device = configure("cuda")
    params = {"seed": settings["training_seed"], "device": device, "batch_size": settings["batch_size"]}
    common = train_sparse(training, initial, [], steps=settings["warmup_updates"], **params)
    means = {}
    for arm, selection in plans.items():
        trained = train_sparse(training, selection, [], steps=settings["additional_updates"], start=common, **params)
        threshold, support = calibrate(collect(calibration, trained, device), settings["calibration_false_positive_rate"])
        raw = collect(assessment, trained, device)
        data = rates(raw, threshold)
        means[arm] = aggregate(data)
        save_json(args.output / f"{arm}.json", {"assessment": data, "thresholds": threshold, "support": support,
                                               "assessment_pauc": {"all_absent": partial_auc(raw),
                                                                   "near_absent": partial_auc(raw, "near_absent")}})
        log("mask_budget_robustness", fold=args.fold, method=arm, training_budget=args.budget, metrics=means[arm])
    calibration_budget = sum(len(case.images) for case in calibration.values())
    save_json(args.output / "summary.json", {"means": means, "training_budget": args.budget,
                                           "calibration_budget": calibration_budget,
                                           "analysis": "post-hoc mask-budget sensitivity on the fixed official audit"})


if __name__ == "__main__":
    main()
