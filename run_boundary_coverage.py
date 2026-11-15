"""Fit and assess four fixed-budget acquisition arms on the existing ACDC protocol."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_acquisition_baselines import acquisition_rosters
from audit_external_transfer import digest
from boundary_coverage import scores_from_views, select_boundaries
from experiment import load_json, log, save_json
from image_slice_budget import CLASSES, allocate_image
from replay_audit_baselines import save_teacher
from run_image_slice_budget import centers, image_view, patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, rates
from segment import configure, predict_many
from sparse_train import train_sparse


def view_summary(cases, teacher, device, design):
    all_views = {patient: [] for patient in cases}
    original = {}
    for gain, bias in design["views"]:
        stacks = [SimpleNamespace(images=case.images * gain + bias) for case in cases.values()]
        for patient, probability in zip(cases, predict_many(teacher, stacks, device), strict=True):
            if (gain, bias) == (1.0, 0.0):
                original[patient] = probability
            all_views[patient].append(scores_from_views(probability, design["tail_fraction"], design["area_cutoff"]))
    summaries = {}
    for patient, views in all_views.items():
        tails, areas = zip(*views, strict=True)
        summaries[patient] = (np.clip(np.median(tails, axis=0), 0.01, 0.99), np.median(areas, axis=0))
    return original, summaries


def acquisition_counts(truth, initial, selection):
    results = {}
    for patient, case in truth.items():
        eligible = set(selection[patient]) - set(initial[patient])
        counts = {str(label): {"small": 0, "absent": 0, "near_absent": 0, "adjacent_positive": 0}
                  for label in CLASSES}
        for label in CLASSES:
            area = np.count_nonzero(case.masks == label, axis=(1, 2))
            by_label = counts[str(label)]
            for index in eligible:
                phase = case.phase == case.phase[index]
                peak = int(area[phase].max())
                near = phase & (np.abs(case.slice_index - case.slice_index[index]) == 1)
                by_label["small"] += int(0 < 4 * area[index] <= peak)
                by_label["absent"] += int(area[index] == 0)
                by_label["near_absent"] += int(area[index] == 0 and ((area[near] > 0) & (4 * area[near] <= peak)).any())
                by_label["adjacent_positive"] += int(area[index] == 0 and (area[near] > 0).any())
        results[patient] = counts
    return results


def audit_nominations(truth, nominations, count):
    rows = []
    for patient, phase, label, index, _ in sorted(nominations, key=lambda row: -row[-1])[:count]:
        case = truth[patient]
        area = np.count_nonzero(case.masks == label, axis=(1, 2))
        in_phase = case.phase == phase
        neighboring = in_phase & (np.abs(case.slice_index - case.slice_index[index]) == 1)
        peak = int(area[in_phase].max())
        rows.append({
            "patient": patient, "index": index, "label": label, "phase": phase,
            "absent": bool(area[index] == 0),
            "near_absent": bool(area[index] == 0 and ((area[neighboring] > 0) & (4 * area[neighboring] <= peak)).any()),
            "adjacent_positive": bool(area[index] == 0 and (area[neighboring] > 0).any()),
            "small": bool(0 < 4 * area[index] <= peak),
        })
    return {"available": len(nominations), "nominations": rows}


def score_cases(cases, model, device, cutoffs):
    records = collect(cases, model, device)
    scores = {patient: {key: [float(value) - cutoffs[label] for label, rows in groups.items() for value in rows[key]]
                        for key in ("small", "absent", "near_absent")}
              for patient, (groups, _, _) in records.items()}
    return {"scores": scores, "outcomes": rates(records, cutoffs)}


def fit(args):
    design = load_json(args.design)
    settings = load_json(args.registration)
    if args.seed not in design["seeds"] or args.fold not in design["folds"]:
        raise ValueError("Unexpected fold/seed for locked design")
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    if len(cache) != 100:
        raise ValueError("Use the official 100-case ACDC training release")
    training_ids, calibration_ids, development_ids = patient_roster(cache, settings, args.fold)
    if not (len(training_ids) == 40 and len(calibration_ids) == 12 and len(development_ids) == 20):
        raise ValueError("Patient roles disagree with source protocol")
    device = configure("cuda")
    images = {patient: image_view(cache[patient]) for patient in training_ids}
    truth = {patient: _load_case(cache[patient]) for patient in training_ids}
    initial = {patient: centers(case) for patient, case in images.items()}
    anchors = {patient: {index: truth[patient].masks[index] for index in indices}
               for patient, indices in initial.items()}
    options = {"seed": args.seed, "device": device, "batch_size": settings["batch_size"]}
    source_archive = args.original / f"fold-{args.fold}"
    fit_args = {"seed": args.seed, "fold": args.fold, "training": training_ids, "calibration": calibration_ids,
                "development": development_ids, "design_sha256": digest(args.design),
                "registration_sha256": digest(args.registration), "code_sha256": digest(Path(__file__)),
                "selection_sha256": digest(Path(__file__).with_name("boundary_coverage.py")),
                "torch": torch.__version__}
    args.output.mkdir(parents=True)
    save_json(args.output / "run.json", fit_args)
    warmup = train_sparse(truth, initial, [], steps=settings["warmup_updates"], **options)
    probabilities, views = view_summary(images, warmup, device, design)
    legacy, legacy_ledger = allocate_image(images, initial, probabilities, args.seed)
    if args.seed == settings["training_seed"]:
        archive = load_json(source_archive / "frozen_acquisitions.json")
        if not all(legacy[arm] == archive["arms"][arm] for arm in ("targeted", "uncertain")):
            raise ValueError("Original seed acquisition lists did not replay")
    del probabilities
    selections = {"targeted": legacy["targeted"],
                  "pooled_image_farthest": acquisition_rosters(images, initial, args.fold)["pooled_image_farthest"]}
    credits = {"targeted": legacy_ledger}
    nominations = {}
    for arm, contiguous in (("anchored", True), ("local", False)):
        selections[arm], credits[arm], nominations[arm] = select_boundaries(
            images, initial, anchors, views, design["strength"], contiguous)
    for arm, roster in selections.items():
        if set(roster) != set(initial):
            raise ValueError(f"Patient roster mismatch for {arm}")
        for patient, indices in roster.items():
            if (len(set(indices)) != 4 or not set(initial[patient]).issubset(indices)
                    or sorted(images[patient].phase[indices].tolist()) != [0, 0, 1, 1]):
                raise ValueError(f"Mask budget mismatch for {arm}")
    save_json(args.output / "frozen_acquisitions.json",
              {"initial": initial, "arms": selections, "credits": credits, "nominations": nominations})
    save_json(args.output / "nomination_audit.json", {
        arm: audit_nominations(truth, rows, design["boundary_precision_k"]) for arm, rows in nominations.items()})
    log("boundary_choices_frozen", fold=args.fold, seed=args.seed)
    calibration = {patient: _load_case(cache[patient]) for patient in calibration_ids}
    development = {patient: _load_case(cache[patient]) for patient in development_ids}
    for arm in design["arms"]:
        model = train_sparse(truth, selections[arm], [], steps=settings["additional_updates"], start=warmup, **options)
        cutoffs, support = calibrate(collect(calibration, model, device), settings["calibration_false_positive_rate"])
        result = score_cases(development, model, device, cutoffs)
        save_json(args.output / f"{arm}.json", {**result, "cutoffs": cutoffs, "calibration_support": support,
                  "weight_sha256": save_teacher(args.output / f"{arm}.pt", model),
                  "training_selection": acquisition_counts(truth, initial, selections[arm])})
        log("boundary_arm_complete", fold=args.fold, seed=args.seed, arm=arm, metrics=aggregate(result["outcomes"]))


def official(args):
    design = load_json(args.design)
    metadata = load_json(args.fit / "run.json")
    if metadata["design_sha256"] != digest(args.design):
        raise ValueError("Official assessment requires the same frozen acquisition design")
    cache = {path.stem: path for path in args.assessment.glob("patient*.npz")}
    if len(cache) != 50 or set(cache) & set(metadata["training"] + metadata["calibration"] + metadata["development"]):
        raise ValueError("Official cache must contain 50 disjoint test patients")
    args.output.mkdir(parents=True)
    save_json(args.output / "run.json", metadata)
    cases = {patient: _load_case(path) for patient, path in sorted(cache.items())}
    device = configure("cuda")
    for arm in design["arms"]:
        fitted = load_json(args.fit / f"{arm}.json")
        weights = args.fit / f"{arm}.pt"
        if digest(weights) != fitted["weight_sha256"]:
            raise ValueError("Frozen weight digest changed")
        model = SimpleNamespace(teacher=torch.load(weights, map_location="cpu", weights_only=True))
        cutoffs = {int(key): value for key, value in fitted["cutoffs"].items()}
        result = score_cases(cases, model, device, cutoffs)
        save_json(args.output / f"{arm}.json", result)
        log("boundary_official_arm", fold=metadata["fold"], seed=metadata["seed"], arm=arm, metrics=aggregate(result["outcomes"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fields in (("fit", ("data", "registration", "original", "design", "output")),
                         ("official", ("fit", "assessment", "design", "output"))):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "fit":
            command.add_argument("--fold", type=int, choices=(0, 1), required=True)
            command.add_argument("--seed", type=int, choices=(2027, 2028, 2029), required=True)
    args = parser.parse_args()
    (fit if args.command == "fit" else official)(args)


if __name__ == "__main__":
    main()
