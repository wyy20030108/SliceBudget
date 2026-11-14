"""Exploratory robustness check on frozen acquisition sets under supervised training."""

import argparse
from pathlib import Path

from acdc_data import _load_case
from experiment import load_json, log, save_json
from run_image_slice_budget import patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from segment import configure
from sparse_train import train_sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "official-audit", "registration", "original", "distribution", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
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
    frozen = load_json(args.original / "frozen_acquisitions.json")
    supplemental = load_json(args.distribution / "audit_plan.json")["choices"]
    args.output.mkdir(parents=True)
    training = {p: _load_case(cache[p]) for p in train_ids}
    calibration = {p: _load_case(cache[p]) for p in cal_ids}
    assessment = {p: _load_case(audit[p]) for p in sorted(audit)}
    initial = {p: frozen["warmup"][p] for p in train_ids}
    device = configure("cuda")
    params = {"seed": settings["training_seed"], "device": device, "batch_size": settings["batch_size"],
              "consistency": False}
    common = train_sparse(training, initial, [], steps=settings["warmup_updates"], **params)
    means = {}
    for arm in ("interval", "uncertain", "targeted", "mixed_image"):
        selection = supplemental[arm] if arm == "mixed_image" else frozen["arms"][arm]
        if set(selection) != set(initial) or any(len(set(indices)) != 4 or set(initial[p]) - set(indices)
                                               for p, indices in selection.items()):
            raise ValueError("Original set must preserve the shared four-mask budget")
        trained = train_sparse(training, selection, [], steps=settings["additional_updates"], start=common, **params)
        threshold, support = calibrate(collect(calibration, trained, device), settings["calibration_false_positive_rate"])
        raw = collect(assessment, trained, device)
        data = rates(raw, threshold)
        means[arm] = aggregate(data)
        save_json(args.output / f"{arm}.json", {"assessment": data, "thresholds": threshold, "support": support,
                                               "assessment_pauc": {"all_absent": partial_auc(raw),
                                                                   "near_absent": partial_auc(raw, "near_absent")}})
        log("supervised_robustness", fold=args.fold, method=arm, metrics=means[arm])
    save_json(args.output / "summary.json", {"means": means, "analysis": "post-hoc supervised robustness on fixed official audit"})


if __name__ == "__main__":
    main()
