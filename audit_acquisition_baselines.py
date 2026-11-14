"""Compare two image-only controls at the frozen 160-mask ACDC budget."""

import argparse
from hashlib import blake2s, sha256
from pathlib import Path

import numpy as np

from acdc_data import _load_case
from experiment import load_json, log, save_json
from run_image_slice_budget import image_view, patient_roster, selection_quality
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from run_sparse_annotation import paired_difference
from segment import configure
from sparse_train import train_sparse


ARMS = ("uniform_random", "pooled_image_farthest")
RANDOM_SEED = 20261126
METRICS = ("recall", "fpr", "near_fpr", "global_dice", "small_dice")


def acquisition_rosters(images, initial, fold):
    selections = {arm: {} for arm in ARMS}
    for patient, case in sorted(images.items()):
        if len(initial[patient]) != 2 or sorted(case.phase[initial[patient]].tolist()) != [0, 1]:
            raise ValueError(f"Expected one initial mask per phase for {patient}")
        count, height, width = case.images.shape
        if height % 16 or width % 16:
            raise ValueError("The image grid must be evenly divisible into 16×16 cells")
        features = case.images.reshape(count, 16, height // 16, 16, width // 16).mean(axis=(2, 4))
        features = features.reshape(count, 256).astype(np.float64)
        features -= features.mean(axis=1, keepdims=True)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        choices = {arm: list(initial[patient]) for arm in ARMS}
        patient_seed = int.from_bytes(blake2s(patient.encode(), digest_size=8).digest(), "little")
        for phase in (0, 1):
            start = next(index for index in initial[patient] if case.phase[index] == phase)
            candidates = np.flatnonzero((case.phase == phase) & (np.arange(count) != start))
            if not len(candidates):
                raise ValueError(f"No new phase-{phase} candidate for {patient}")
            candidates = candidates[np.argsort(case.slice_index[candidates], kind="stable")]
            rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, fold, phase, patient_seed]))
            choices[ARMS[0]].append(int(rng.choice(candidates)))
            distances = np.square(features[candidates] - features[start]).sum(axis=1)
            choices[ARMS[1]].append(int(candidates[distances.argmax()]))
        for arm in ARMS:
            selections[arm][patient] = sorted(choices[arm])
    return selections


def freeze_prior(args, settings):
    training_cache = {p.stem: p for p in args.data.glob("patient*.npz")}
    official_cache = {p.stem: p for p in args.official_audit.glob("patient*.npz")}
    if len(training_cache) != 100 or len(official_cache) != 50 or training_cache.keys() & official_cache.keys():
        raise ValueError("Expected 100 ACDC training and 50 disjoint official test patients")
    train_ids, calibration_ids, development_ids = patient_roster(training_cache, settings, args.fold)
    registration_hash = sha256(args.registration.read_bytes()).hexdigest()
    fields = {"fold": args.fold, "seed": settings["training_seed"], "training": train_ids,
              "calibration": calibration_ids, "registration_sha256": registration_hash}
    for prior, ids, is_test in ((args.prior, development_ids, False),
                               (args.official_prior, sorted(official_cache), True)):
        run = load_json(prior / "run.json")
        expected = {**fields, "assessment": ids, "held_out_test_used": is_test}
        if any(run[key] != value for key, value in expected.items()):
            raise ValueError(f"Prior run does not match the frozen experiment: {prior}")
    acquisition = (args.prior / "frozen_acquisitions.json").read_bytes()
    if acquisition != (args.official_prior / "frozen_acquisitions.json").read_bytes():
        raise ValueError("The official and development plans differ")
    plan = load_json(args.prior / "frozen_acquisitions.json")
    images = {patient: image_view(training_cache[patient]) for patient in train_ids}
    initial = plan["warmup"]
    if list(initial) != train_ids or len(initial) * 4 != 160:
        raise ValueError("The frozen initial roster does not match the 160-mask protocol")
    choices = acquisition_rosters(images, initial, args.fold)
    for roster in choices.values():
        for patient, indexes in roster.items():
            if (not set(initial[patient]).issubset(indexes) or len(set(indexes)) != 4
                    or sorted(images[patient].phase[indexes].tolist()) != [0, 0, 1, 1]):
                raise ValueError(f"Budget or warmup mask mismatch for {patient}")
    manifest = {**fields, "random_seed": RANDOM_SEED, "official": sorted(official_cache),
                "development": development_ids, "source_acquisitions_sha256": sha256(acquisition).hexdigest(),
                "masks_per_arm": 160, "initial": initial, "choices": choices,
                "note": "Supplemental controls frozen from training images before loading assessment labels"}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "audit_plan.json"
    if path.exists() and load_json(path) != manifest:
        raise ValueError(f"Existing output uses a different acquisition manifest: {path}")
    save_json(path, manifest)
    return manifest, training_cache, official_cache


def evaluate(args, settings, plan, training_cache, official_cache):
    device = configure("cuda")
    options = {"seed": plan["seed"], "device": device, "batch_size": settings["batch_size"]}
    training = {patient: _load_case(training_cache[patient]) for patient in plan["training"]}
    calibration = {patient: _load_case(training_cache[patient]) for patient in plan["calibration"]}
    official = {patient: _load_case(official_cache[patient]) for patient in plan["official"]}
    comparisons = {}
    original = load_json(args.official_prior / "targeted.json")["assessment"]
    draws = np.random.default_rng(RANDOM_SEED + args.fold).integers(0, len(original), size=(4000, len(original)))
    warmup = train_sparse(training, plan["initial"], [], steps=settings["warmup_updates"], **options)
    for arm, choices in plan["choices"].items():
        model = train_sparse(training, choices, [], steps=settings["additional_updates"], start=warmup, **options)
        threshold, support = calibrate(collect(calibration, model, device), settings["calibration_false_positive_rate"])
        raw = collect(official, model, device)
        patient_data = rates(raw, threshold)
        if set(patient_data) != set(original):
            raise ValueError("New controls and frozen targeted results must assess the same patients")
        summaries = {metric: paired_difference(original, patient_data, metric, draws) for metric in METRICS}
        audit = selection_quality(training, choices, plan["initial"])
        save_json(args.output / f"{arm}.json", {"assessment": patient_data, "thresholds": threshold, "support": support,
                                                 "assessment_pauc": {"all_absent": partial_auc(raw),
                                                                     "near_absent": partial_auc(raw, "near_absent")},
                                                 "acquisition_audit": audit})
        comparisons[arm] = {"metrics": aggregate(patient_data), "targeted_minus_baseline": summaries}
        log("acquisition_control", fold=args.fold, arm=arm, metrics=comparisons[arm]["metrics"])
    save_json(args.output / "summary.json", {"fold": args.fold, "comparisons": comparisons})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "official-audit", "registration", "prior", "official-prior", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--freeze-only", action="store_true", help="Write image-only rosters without loading masks or using CUDA")
    args = parser.parse_args()
    settings = load_json(args.registration)
    plan, training_cache, official_cache = freeze_prior(args, settings)
    if args.freeze_only:
        log("acquisition_controls_frozen", fold=args.fold, output=str(args.output), masks_per_arm=160)
        return
    evaluate(args, settings, plan, training_cache, official_cache)


if __name__ == "__main__":
    main()
