"""Repeat the three central acquisition policies on the same official ACDC cohort."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from acdc_data import _load_case
from audit_acquisition_baselines import acquisition_rosters
from audit_decision_curves import matched_recall, patient_histograms
from experiment import load_json, log, save_json
from image_slice_budget import allocate_image
from run_image_slice_budget import centers, image_view, patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from segment import configure, predict_many
from sparse_train import train_sparse
from summarize_annotation_evidence import summary


ARMS = ("uncertain", "pooled_image_farthest", "targeted")
SEEDS = (2028, 2029)
FOLDS = (0, 1)
MATCHED_FPR = 0.2
NEGATIVES = ("absent", "near_absent")
ROLES = ("training", "calibration", "assessment")


def replay_agreement(outcomes, thresholds, archived):
    original = archived["assessment"]
    if set(original) != set(outcomes):
        raise ValueError("Archived assessment patients differ")
    discrepancies = []
    for patient, measures in outcomes.items():
        for metric, value in measures.items():
            expected = original[patient][metric]
            if (value is None) != (expected is None):
                raise ValueError(f"Archived metric support differs for {patient}/{metric}")
            if value is not None:
                discrepancies.append(abs(value - expected))
    return {
        "max_abs_patient_delta": max(discrepancies),
        "max_abs_threshold_delta": max(abs(value - archived["thresholds"][str(label)])
                                       for label, value in thresholds.items()),
    }


def matched_operating_points(records, thresholds):
    def margins(subset):
        return [[float(value) - thresholds[label] for label, group in groups.items() for value in group[subset]]
                for groups, _, _ in records.values()]

    positives = margins("small")
    weights = np.ones((1, len(positives)))
    points = {}
    for subset in NEGATIVES:
        negatives = margins(subset)
        histograms = patient_histograms(positives, negatives)
        points[subset] = {
            "recall_at_20pct_fpr": float(matched_recall(histograms, weights, (MATCHED_FPR,))[0, 0]),
            "small_patients": sum(map(bool, positives)),
            "negative_patients": sum(map(bool, negatives)),
            "small_pairs": sum(map(len, positives)),
            "negative_pairs": sum(map(len, negatives)),
        }
    return points


def train_and_score(args):
    settings = load_json(args.registration)
    cache = {p.stem: p for p in args.data.glob("patient*.npz")}
    official = {p.stem: p for p in args.official_audit.glob("patient*.npz")}
    if len(cache) != 100 or len(official) != 50 or cache.keys() & official.keys():
        raise ValueError("Use the distinct official ACDC 100/50 training/test caches")
    training_ids, calibration_ids, _ = patient_roster(cache, settings, args.fold)
    testing_ids = sorted(official)
    if (len(training_ids), len(calibration_ids)) != (40, 12) or set(training_ids) & set(calibration_ids):
        raise ValueError("Expected disjoint 40/12 source patients")
    roles = dict(zip(ROLES, (training_ids, calibration_ids, testing_ids), strict=True))
    original = load_json(args.official_original / "run.json")
    if (original["fold"] != args.fold or not original["held_out_test_used"]
            or any(original[key] != value for key, value in roles.items())):
        raise ValueError("Patient roles must match the archived official assessment")
    if (settings["warmup_updates"], settings["additional_updates"], settings["batch_size"],
        settings["calibration_false_positive_rate"]) != (1800, 2000, 4, MATCHED_FPR):
        raise ValueError("Use the original training schedule and calibration target")

    device = configure("cuda")
    manifest = {**roles, "fold": args.fold, "seed": args.seed, "pytorch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device), "held_out_test_used": True,
                "registration_sha256": sha256(args.registration.read_bytes()).hexdigest(),
                "reference_protocol_sha256": sha256((args.official_original / "run.json").read_bytes()).hexdigest(),
                "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    images = {patient: image_view(cache[patient]) for patient in training_ids}
    initial = {patient: centers(case) for patient, case in images.items()}
    baseline_plan = load_json(args.acquisition_controls / "audit_plan.json")
    image_difference = acquisition_rosters(images, initial, args.fold)["pooled_image_farthest"]
    if (baseline_plan["initial"] != initial or baseline_plan["fold"] != args.fold
            or baseline_plan["choices"]["pooled_image_farthest"] != image_difference):
        raise ValueError("Image-difference allocations must match the original image-only rule")

    options = {"seed": args.seed, "device": device, "batch_size": settings["batch_size"]}
    args.output.mkdir(parents=True)
    save_json(args.output / "run.json", manifest)
    training = {patient: _load_case(cache[patient]) for patient in training_ids}
    common = train_sparse(training, initial, [], steps=settings["warmup_updates"], **options)
    probabilities = dict(zip(images, predict_many(common, images.values(), device), strict=True))
    choices, _ = allocate_image(images, initial, probabilities, args.seed)
    del probabilities
    choices["pooled_image_farthest"] = image_difference
    selected = {arm: choices[arm] for arm in ARMS}
    for roster in selected.values():
        if set(roster) != set(training_ids) or any(
            len(set(idx)) != 4 or not set(initial[p]).issubset(idx)
            or np.bincount(images[p].phase[idx], minlength=2).tolist() != [2, 2]
            for p, idx in roster.items()
        ):
            raise ValueError("Every policy must use the same initial and training mask budget")

    archived_run = None
    if args.seed == SEEDS[0]:
        archived_run = load_json(args.archived_2028 / "run.json")
        old = load_json(args.archived_2028 / "frozen_acquisitions.json")
        if (any(archived_run[key] != manifest[key] for key in (*ROLES, "fold", "seed"))
                or old["warmup"] != initial
                or any(old["arms"][arm] != selected[arm] for arm in ("targeted", "uncertain"))):
            raise ValueError("The seed-2028 replay differs from the archived patient or acquisition lists")
    save_json(args.output / "frozen_acquisitions.json", {"initial": initial, "arms": selected})
    log("strong_seed_choices_frozen", seed=args.seed, fold=args.fold, warmup_seconds=common.seconds)

    calibration = {patient: _load_case(cache[patient]) for patient in calibration_ids}
    assessment = {patient: _load_case(official[patient]) for patient in testing_ids}
    for arm, masks in selected.items():
        model = train_sparse(training, masks, [], steps=settings["additional_updates"], start=common, **options)
        cal = collect(calibration, model, device)
        thresholds, support = calibrate(cal, MATCHED_FPR)
        scores = collect(assessment, model, device)
        outcomes = rates(scores, thresholds)
        replay = None
        if archived_run and arm != "pooled_image_farthest":
            replay = replay_agreement(outcomes, thresholds, load_json(args.archived_2028 / f"{arm}.json"))
            replay["same_environment"] = all(manifest[k] == archived_run[k] for k in ("pytorch", "gpu"))
            if replay["same_environment"] and max(replay["max_abs_patient_delta"],
                                                  replay["max_abs_threshold_delta"]) > 1e-6:
                raise ValueError(f"Same-environment seed-2028 outcomes did not replay for {arm}: {replay}")
        save_json(args.output / f"{arm}.json", {
            "assessment": outcomes,
            "assessment_pauc": {"all_absent": partial_auc(scores), "near_absent": partial_auc(scores, "near_absent")},
            "calibration": aggregate(rates(cal, thresholds)),
            "thresholds": thresholds,
            "calibration_negative_patient_support": support,
            "matched_operating_points": matched_operating_points(scores, thresholds),
            "archived_2028_replay": replay,
            "training_seconds": model.seconds,
        })
        log("strong_seed_arm_complete", fold=args.fold, seed=args.seed, arm=arm, metrics=aggregate(outcomes))


def summarize(runs, output):
    results = {
        "scope": "Post-hoc robustness, same official ACDC patients and source folds; training repeats are not new patients",
        "matched_fpr": MATCHED_FPR,
        "matched_estimator": "Retrospective patient-macro ROC by fitted fold using class-cutoff-centered scores; mean of fold recalls",
        "contrast_reference": "balanced targeting minus comparator",
        "results_by_seed": {},
    }
    for seed in SEEDS:
        paths = [runs / f"seed-{seed}-fold-{fold}" for fold in FOLDS]
        manifests = [load_json(path / "run.json") for path in paths]
        for index, manifest in enumerate(manifests):
            if (manifest["seed"] != seed or manifest["fold"] != index
                    or manifest["assessment"] != manifests[0]["assessment"]
                    or len(manifest["assessment"]) != 50):
                raise ValueError("Expected two distinct original training folds and a shared official assessment")
        arms = {arm: [load_json(path / f"{arm}.json") for path in paths] for arm in ARMS}
        estimates = summary(arms, ARMS, ARMS[:-1])
        estimates["by_fold"] = {arm: [{
            "matched_operating_points": fold["matched_operating_points"],
            "calibration": fold["calibration"],
            "calibration_thresholds": fold["thresholds"],
            "calibration_negative_patient_support": fold["calibration_negative_patient_support"],
            "archived_2028_replay": fold["archived_2028_replay"],
            "training_seconds": fold["training_seconds"],
        } for fold in folds] for arm, folds in arms.items()}
        matched = {arm: {subset: float(np.mean([fold["matched_operating_points"][subset]["recall_at_20pct_fpr"]
                                               for fold in folds])) for subset in NEGATIVES} for arm, folds in arms.items()}
        estimates["matched_recall"] = matched
        estimates["matched_contrasts"] = {
            arm: {subset: matched["targeted"][subset] - matched[arm][subset] for subset in NEGATIVES}
            for arm in ARMS[:-1]
        }
        results["results_by_seed"][str(seed)] = estimates
    save_json(output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("run")
    for option in ("data", "official-audit", "official-original", "acquisition-controls", "registration", "output"):
        train.add_argument(f"--{option}", type=Path, required=True)
    train.add_argument("--archived-2028", type=Path)
    train.add_argument("--fold", type=int, choices=FOLDS, required=True)
    train.add_argument("--seed", type=int, choices=SEEDS, required=True)
    merge = commands.add_parser("aggregate")
    merge.add_argument("--runs", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.seed == SEEDS[0] and args.archived_2028 is None:
            parser.error("Seed 2028 requires --archived-2028 for replay validation")
        train_and_score(args)
    else:
        summarize(args.runs, args.output)


if __name__ == "__main__":
    main()
