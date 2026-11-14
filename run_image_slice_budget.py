"""Train and evaluate image-guided sparse acquisition at one common mask budget."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case, make_protocol
from experiment import load_json, log, save_json
from image_slice_budget import allocate_image
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from run_sparse_annotation import paired_difference
from segment import configure, predict_many
from sparse_train import train_sparse


def patient_roster(cohort, settings, fold):
    hidden = set(make_protocol(cohort, settings["holdout_protocol_seed"])["test"])
    shuffled = np.random.default_rng(settings["development_seed"]).permutation(sorted(cohort.keys() - hidden))
    blocks = np.array_split(shuffled, settings["blocks"])
    train = blocks[fold].tolist()
    evaluation = blocks[1 - fold].tolist()
    calibration_size = settings["calibration_patients"]
    assess_size = settings["assessment_patients"]
    return train, evaluation[:calibration_size], evaluation[calibration_size:calibration_size + assess_size]


def image_view(path):
    with np.load(path, allow_pickle=False) as stored:
        return SimpleNamespace(patient_id=path.stem, images=stored["images"], phase=stored["phase"],
                               slice_index=stored["slice_index"])


def centers(case):
    result = []
    for phase in (0, 1):
        members = np.flatnonzero(case.phase == phase)
        ordered = members[np.argsort(case.slice_index[members])]
        if len(ordered) < 2:
            raise ValueError("Acquisition needs two or more slices per phase")
        result.append(int(ordered[(len(ordered) - 1) // 2]))
    return result


def selection_quality(truth, selected, initial):
    tally = [[{"small": 0, "absent": 0, "all": 0} for _ in range(3)] for _ in range(2)]
    for patient, indexes in selected.items():
        case = truth[patient]
        added = sorted(set(indexes) - set(initial[patient]))
        for index in added:
            phase = int(case.phase[index])
            for label in (1, 2, 3):
                areas = np.count_nonzero(case.masks == label, axis=(1, 2))
                total = areas[case.phase == phase].max()
                row = tally[phase][label - 1]
                row["absent"] += int(areas[index] == 0)
                row["small"] += int(0 < 4 * areas[index] <= total)
                row["all"] += 1
    return tally


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "output", "registration"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1))
    parser.add_argument("--official-audit", type=Path, help="Optional, distinct official ACDC challenge test cache")
    args = parser.parse_args()
    settings = load_json(args.registration)
    cache = {item.stem: item for item in args.data.glob("patient*.npz")}
    if len(cache) != 100:
        parser.error("Provide the official ACDC training cache")
    train_ids, cal_ids, evaluation_ids = patient_roster(cache, settings, args.fold)
    evaluation_cache = cache
    if args.official_audit is not None:
        evaluation_cache = {item.stem: item for item in args.official_audit.glob("patient*.npz")}
        if len(evaluation_cache) != 50 or set(evaluation_cache).intersection(cache):
            parser.error("Official audit requires 50 test patients disjoint from the 100 training cases")
        evaluation_ids = sorted(evaluation_cache)
    args.output.mkdir(parents=True)
    device = configure("cuda")
    seed = settings["training_seed"]
    options = {"seed": seed, "device": device, "batch_size": settings["batch_size"]}
    save_json(args.output / "run.json", {"fold": args.fold, "seed": seed, "training": train_ids,
                                        "calibration": cal_ids, "assessment": evaluation_ids,
                                        "registration_sha256": sha256(args.registration.read_bytes()).hexdigest(),
                                        "pytorch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                                        "held_out_test_used": args.official_audit is not None,
                                        "assessment_release": "official challenge test" if args.official_audit else "development"})
    images = {patient: image_view(cache[patient]) for patient in train_ids}
    initial = {patient: centers(case) for patient, case in images.items()}
    truth = {patient: _load_case(cache[patient]) for patient in train_ids}
    common = train_sparse(truth, initial, [], steps=settings["warmup_updates"], **options)
    probabilities = dict(zip(images, predict_many(common, images.values(), device), strict=True))
    selected, ledger = allocate_image(images, initial, probabilities, seed)
    del probabilities
    for arm in settings["methods"]:
        if set(selected[arm]) != set(train_ids):
            raise ValueError("Acquisition patient mismatch")
        for patient, indices in selected[arm].items():
            if len(set(indices)) != 4 or set(initial[patient]) - set(indices):
                raise ValueError("Each arm must keep its two initial masks and add exactly two")
            if np.bincount(images[patient].phase[indices], minlength=2).tolist() != [2, 2]:
                raise ValueError("Phase annotation budget mismatch")
    save_json(args.output / "frozen_acquisitions.json", {"warmup": initial, "arms": selected, "ledger": ledger})
    log("acquisition_frozen", fold=args.fold, masks_each_arm=4 * len(train_ids))
    calibration = {patient: _load_case(cache[patient]) for patient in cal_ids}
    evaluation = {patient: _load_case(evaluation_cache[patient]) for patient in evaluation_ids}
    results = {}
    for arm in settings["methods"]:
        branch = train_sparse(truth, selected[arm], [], steps=settings["additional_updates"], start=common, **options)
        cal = collect(calibration, branch, device)
        thresholds, support = calibrate(cal, settings["calibration_false_positive_rate"])
        raw = collect(evaluation, branch, device)
        results[arm] = rates(raw, thresholds)
        metrics = aggregate(results[arm])
        save_json(args.output / f"{arm}.json", {"assessment": results[arm], "thresholds": thresholds,
                                               "calibration": aggregate(rates(cal, thresholds)), "support": support,
                                               "assessment_pauc": {"all_absent": partial_auc(raw),
                                                                   "near_absent": partial_auc(raw, "near_absent")},
                                               "acquisition_audit": selection_quality(truth, selected[arm], initial)})
        log("image_acquisition_completed", fold=args.fold, arm=arm, metrics=metrics)
    rng = np.random.default_rng(seed + args.fold)
    draws = rng.integers(0, len(evaluation_ids), size=(4000, len(evaluation_ids)))
    comparisons = {arm: {metric: paired_difference(results["targeted"], results[arm], metric, draws)
                         for metric in ("recall", "fpr", "near_fpr", "global_dice", "small_dice")}
                   for arm in settings["methods"] if arm != "targeted"}
    save_json(args.output / "summary.json", {"methods": {arm: aggregate(data) for arm, data in results.items()},
                                             "comparisons": comparisons, "held_out_test_used": args.official_audit is not None})


if __name__ == "__main__":
    main()
