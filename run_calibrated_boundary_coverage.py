"""Fit calibrated boundary coverage while keeping the original dense label budget."""

import argparse
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import torch

from acdc_data import _load_case
from audit_external_transfer import digest
from calibrated_boundary_coverage import build_groups, fit as fit_state_model, select
from experiment import load_json, log, save_json
from replay_audit_baselines import save_teacher
from run_boundary_coverage import acquisition_counts, audit_nominations, score_cases, view_summary
from run_image_slice_budget import centers, image_view, patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect
from segment import configure
from sparse_train import train_sparse


def anchors_for(images, truth, initial):
    if set(images) != set(initial) or set(images) != set(truth):
        raise ValueError("Anchor roster does not match images and reference")
    return {patient: {index: truth[patient].masks[index] for index in indices} for patient, indices in initial.items()}


def fit(args):
    design = load_json(args.design)
    settings = load_json(args.registration)
    prior = load_json(args.prior / "run.json")
    original = load_json(args.prior / "frozen_acquisitions.json")
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    training, calibration, development = patient_roster(cache, settings, args.fold)
    if (len(cache) != 100 or (len(training), len(calibration), len(development)) != (40, 12, 20)
            or prior["fold"] != args.fold or prior["seed"] != args.seed
            or prior["training"] != training or prior["calibration"] != calibration
            or prior["development"] != development or prior["registration_sha256"] != digest(args.registration)):
        raise ValueError("Source roles and archived baseline run must match")
    if set(training) & set(calibration + development) or set(calibration) & set(development):
        raise ValueError("Patient roles must be disjoint per fit")
    reliability_ids, cutoff_ids = calibration[::2], calibration[1::2]
    initial = original["initial"]
    train_cases = {p: _load_case(cache[p]) for p in training}
    reliability_cases = {p: _load_case(cache[p]) for p in reliability_ids}
    train_images = {p: image_view(cache[p]) for p in training}
    reliability_images = {p: image_view(cache[p]) for p in reliability_ids}
    reliability_initial = {p: centers(case) for p, case in reliability_images.items()}
    device = configure("cuda")
    options = {"seed": args.seed, "device": device, "batch_size": settings["batch_size"]}
    args.output.mkdir(parents=True)
    metadata = {**prior, "design_sha256": digest(args.design), "code_sha256": digest(Path(__file__)),
                "selection_sha256": digest(Path(__file__).with_name("calibrated_boundary_coverage.py")),
                "reliability_fit": reliability_ids, "detection_calibration": cutoff_ids,
                "reference_masks": {
                    "dense_training": 160,
                    "selection_calibration": sum(len(case.images) for case in reliability_cases.values()),
                }}
    save_json(args.output / "run.json", metadata)
    warmup = train_sparse(train_cases, initial, [], steps=settings["warmup_updates"], **options)
    original_views, views = view_summary({**train_images, **reliability_images}, warmup, device, design)
    del original_views
    sample = build_groups(reliability_images, reliability_initial,
                          anchors_for(reliability_images, reliability_cases, reliability_initial),
                          {p: views[p] for p in reliability_ids}, truth=reliability_cases)
    candidates = build_groups(train_images, initial, anchors_for(train_images, train_cases, initial),
                              {p: views[p] for p in training})
    selections = {arm: original["arms"][arm] for arm in ("anchored", "pooled_image_farthest", "targeted")}
    ledgers, nominations = {}, {}
    for arm, contiguous in (("calibrated", True), ("calibrated_local", False)):
        model = fit_state_model(sample, contiguous=contiguous)
        selections[arm], ledgers[arm], nominations[arm] = select(candidates, initial, model, contiguous=contiguous,
                                                                strength=design["strength"])
        if set(selections[arm]) != set(training) or any(
                len(set(indices)) != 4 or not set(initial[p]).issubset(indices)
                or sorted(train_images[p].phase[indices].tolist()) != [0, 0, 1, 1]
                for p, indices in selections[arm].items()):
            raise ValueError("The primary acquisition must retain exactly 160 masks")
    save_json(args.output / "frozen_acquisitions.json",
              {"initial": initial, "arms": selections, "credits": ledgers, "nominations": nominations})
    save_json(args.output / "nomination_audit.json", {
        arm: audit_nominations(train_cases, rows, design["boundary_precision_k"]) for arm, rows in nominations.items()})
    detection_cases = {p: _load_case(cache[p]) for p in cutoff_ids}
    metadata["reference_masks"]["detection_calibration"] = sum(len(case.images) for case in detection_cases.values())
    save_json(args.output / "run.json", metadata)
    assessments = {p: _load_case(cache[p]) for p in development}
    for arm in design["arms"]:
        weight_path = args.output / f"{arm}.pt"
        if arm in ledgers:
            model = train_sparse(train_cases, selections[arm], [], steps=settings["additional_updates"],
                                 start=warmup, **options)
            weight_hash = save_teacher(weight_path, model)
        else:
            source = args.prior / f"{arm}.pt"
            if digest(source) != load_json(args.prior / f"{arm}.json")["weight_sha256"]:
                raise ValueError("Previously trained comparator has changed")
            try:
                weight_path.hardlink_to(source.resolve())
            except OSError:
                copyfile(source, weight_path)
            model = SimpleNamespace(teacher=torch.load(weight_path, map_location="cpu", weights_only=True))
            weight_hash = digest(weight_path)
        cutoffs, support = calibrate(collect(detection_cases, model, device), settings["calibration_false_positive_rate"])
        result = score_cases(assessments, model, device, cutoffs)
        save_json(args.output / f"{arm}.json", {
            **result, "cutoffs": cutoffs, "calibration_support": support, "weight_sha256": weight_hash,
            "training_selection": acquisition_counts(train_cases, initial, selections[arm])})
        log("calibrated_boundary_arm_complete", fold=args.fold, seed=args.seed, arm=arm,
            metrics=aggregate(result["outcomes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("data", "registration", "design", "prior", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, choices=(2027, 2028, 2029), required=True)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
