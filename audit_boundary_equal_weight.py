"""Compare acquired states with V2's calibrated and matched-weight raw credits."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from boundary_coverage import select_boundaries
from calibrated_boundary_coverage import build_groups, fit, select
from experiment import load_json, log, save_json
from image_slice_budget import CLASSES
from run_boundary_coverage import acquisition_counts, view_summary
from run_calibrated_boundary_coverage import anchors_for
from run_image_slice_budget import centers, image_view
from segment import configure
from sparse_train import train_sparse


SEEDS = (2027, 2028, 2029)
METRICS = ("small", "absent", "near_absent")
ARMS = ("calibrated", "raw_weight_1", "raw_weight_0_5")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def replay_fold(args):
    settings, design = load_json(args.registration), load_json(args.design)
    if design["strength"] != 1.0 or tuple(design["seeds"]) != SEEDS:
        raise ValueError("Unexpected calibrated boundary design")
    paths = {path.stem: path for path in args.data.glob("patient*.npz")}
    if len(paths) != 100:
        raise ValueError("Expected only the original 100-case ACDC training cache")
    device = configure("cuda")
    outputs, roster, receipts = {}, None, {}
    for seed in SEEDS:
        directory = args.v2 / f"seed-{seed}-fold-{args.fold}"
        metadata = load_json(directory / "run.json")
        if (metadata["seed"] != seed or metadata["fold"] != args.fold
                or metadata["design_sha256"] != digest(args.design)
                or metadata["registration_sha256"] != digest(args.registration)):
            raise ValueError("V2 metadata differs from the locked design")
        training, calibration = metadata["training"], metadata["reliability_fit"]
        if len(training) != 40 or len(calibration) != 6 or len(set(training + calibration)) != 46:
            raise ValueError("Unexpected overlap or size in acquisition and calibration roles")
        patients = sorted(training)
        if roster is not None and patients != roster:
            raise ValueError("The training roster must be invariant across seeds")
        roster = patients
        frozen_file = directory / "frozen_acquisitions.json"
        frozen = load_json(frozen_file)
        initial = frozen["initial"]
        cases = {patient: _load_case(paths[patient]) for patient in training}
        images = {patient: image_view(paths[patient]) for patient in training + calibration}
        source_images = {patient: images[patient] for patient in training}
        calibration_images = {patient: images[patient] for patient in calibration}
        calibration_cases = {patient: _load_case(paths[patient]) for patient in calibration}
        anchors = anchors_for(source_images, cases, initial)
        calibr_initial = {patient: centers(case) for patient, case in calibration_images.items()}
        calibr_anchors = anchors_for(calibration_images, calibration_cases, calibr_initial)

        def replay_with(warmup):
            raw_probabilities, views = view_summary(images, warmup, device, design)
            del raw_probabilities
            source_views = {patient: views[patient] for patient in training}
            raw_original, _, _ = select_boundaries(source_images, initial, anchors, source_views,
                                                     strength=0.5, contiguous=True)
            if raw_original != frozen["arms"]["anchored"]:
                return None
            samples = build_groups(calibration_images, calibr_initial, calibr_anchors,
                                   {patient: views[patient] for patient in calibration}, truth=calibration_cases)
            options = build_groups(source_images, initial, anchors, source_views)
            model = fit(samples, contiguous=True)
            calibrated, _, _ = select(options, initial, model, contiguous=True, strength=1.0)
            if calibrated != frozen["arms"]["calibrated"]:
                return None
            matched, _, _ = select_boundaries(source_images, initial, anchors, source_views,
                                              strength=1.0, contiguous=True)
            return dict(zip(ARMS, (calibrated, matched, raw_original), strict=True))

        control = None
        backup = args.warmups / f"fold-{args.fold}" / "warmup.pt"
        source_mode = "deterministic warm-up replay"
        if seed == 2027 and backup.is_file():
            teacher = torch.load(backup, map_location="cpu", weights_only=True)
            control = replay_with(SimpleNamespace(teacher=teacher))
            if control is not None:
                source_mode = "retained warm-up checkpoint"
        if control is None:
            warmup = train_sparse(cases, initial, [], steps=settings["warmup_updates"], seed=seed,
                                  device=device, batch_size=settings["batch_size"])
            control = replay_with(warmup)
        if control is None:
            raise ValueError(f"Acquisition replay disagrees with V1 or V2 in seed {seed}, fold {args.fold}")
        for arm, chosen in control.items():
            if any(len(set(rows)) != 4 or not set(initial[p]).issubset(rows)
                   or sorted(images[p].phase[rows].tolist()) != [0, 0, 1, 1] for p, rows in chosen.items()):
                raise ValueError(f"{arm} violates the locked 160-mask budget")
        counts = {arm: acquisition_counts(cases, initial, queries) for arm, queries in control.items()}
        if counts["calibrated"] != load_json(directory / "calibrated.json")["training_selection"]:
            raise ValueError("Recomputed V2 outcome counts differ from the archived counts")
        outputs[seed] = [
            [[counts[arm][patient][str(label)][metric] for metric in METRICS] for label in CLASSES]
            for arm in ARMS for patient in patients
        ]
        receipts[seed] = {"frozen_acquisitions_sha256": digest(frozen_file), "warm_up": source_mode,
                          "exact_replay": ["calibrated", "raw_weight_0_5"]}
        log("matched_weight_fold_seed_completed", fold=args.fold, seed=seed, warmup=source_mode)
    save_json(args.output, {"fold": args.fold, "patients": len(roster), "counts": outputs, "receipt": receipts})


def aggregate(args):
    folds = sorted((load_json(path) for path in args.inputs), key=lambda value: value["fold"])
    if [run["fold"] for run in folds] != [0, 1] or any(run["patients"] != 40 for run in folds):
        raise ValueError("Expected two original forty-patient fold outputs")
    # Fold, seed, arm, patient, class, metric. Patient order is stable across seeds.
    values = np.asarray([[run["counts"][str(seed)] for seed in SEEDS] for run in folds]).reshape(
        2, len(SEEDS), len(ARMS), 40, len(CLASSES), len(METRICS))
    rng = np.random.default_rng(20261117)
    weights = [np.vstack([np.ones(40), rng.multinomial(40, np.full(40, 1 / 40), size=4000)]) for _ in folds]
    draws = np.mean([np.einsum("bp,sapcm->bsacm", options, block)
                     for options, block in zip(weights, values, strict=True)], axis=0)

    def metrics(block):
        return {name: {"estimate": float(block[0, i]), "ci95": np.quantile(block[1:, i], [0.025, 0.975]).tolist()}
                for i, name in enumerate(METRICS)}

    def stratify(block):
        return {"all_classes": metrics(block.sum(axis=1)),
                **{name: metrics(block[:, i]) for i, name in enumerate(("RV", "MYO", "LV"))}}

    means = draws.mean(axis=1)
    with_intervals = {arm: stratify(means[:, i]) for i, arm in enumerate(ARMS)}
    contrasts = {
        "calibrated_minus_raw_weight_1": stratify(means[:, 0] - means[:, 1]),
        "raw_weight_1_minus_raw_weight_0_5": stratify(means[:, 1] - means[:, 2]),
    }
    seed_estimates = {seed: {arm: dict(zip(METRICS, samples[i].sum(axis=0).tolist(), strict=True))
                             for i, arm in enumerate(ARMS)} for seed, samples in zip(SEEDS, draws[0], strict=True)}
    result = {
        "scope": "Exploratory acquisition-only control on the original two ACDC development training blocks.",
        "unit": "Mean reference slice-class counts per 80 new training masks, averaged over two folds and three seeds.",
        "analysis": "Four thousand paired patient bootstrap draws within folds, shared across arms and seeds. Intervals condition on warm-up models and frozen choices.",
        "metrics": {"small": "Positive reference area no greater than 25% of the class and phase peak",
                    "absent": "Zero reference area for the class",
                    "near_absent": "Absent with a native neighboring small-positive slice of the same class and phase"},
        "training_patients_per_fold": 40,
        "new_masks_per_fold": 80,
        "variants": with_intervals,
        "paired_contrasts": contrasts,
        "seed_estimates_all_classes": seed_estimates,
        "archive_replay": {run["fold"]: run["receipt"] for run in folds},
    }
    save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    score = commands.add_parser("score")
    for field in ("registration", "design", "data", "v2", "warmups", "output"):
        score.add_argument(f"--{field}", type=Path, required=True)
    score.add_argument("--fold", type=int, choices=(0, 1), required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--inputs", nargs=2, type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "score":
        replay_fold(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
