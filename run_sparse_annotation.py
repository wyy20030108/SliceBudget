"""Development-only matched-budget experiment on within-patient annotation placement."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from acdc_data import _load_case, make_protocol
from experiment import load_json, log, save_json
from segment import configure, predict_many
from slice_budget import allocate
from slice_metrics import patient_metrics
from sparse_train import train_sparse


def implementation_digest():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("acdc_data.py", "segment.py", "sparse_train.py", "slice_budget.py", "slice_metrics.py",
                 "experiment.py", "run_sparse_annotation.py"):
        digest.update(name.encode())
        digest.update(hashlib.sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()


def split_development(cache, settings):
    sealed = set(make_protocol(cache, settings["holdout_protocol_seed"])["test"])
    eligible = sorted(cache.keys() - sealed)
    rng = np.random.default_rng(settings["split_seed"])
    order = rng.permutation(eligible).tolist()
    blocks = []
    offset = 0
    for design in settings["development_blocks"]:
        split = offset + design["train_patients"]
        end = split + design["evaluation_patients"]
        blocks.append((design["seed"], order[offset:split], order[split:end]))
        offset = end
    if offset != len(order):
        raise ValueError("Development blocks must partition the available patients")
    return blocks


def implementation_cache_digest(cache, blocks):
    patients = sorted(patient for _, train_ids, assessment_ids in blocks for patient in train_ids + assessment_ids)
    digest = hashlib.sha256()
    for patient in patients:
        digest.update(patient.encode())
        digest.update(hashlib.sha256(cache[patient].read_bytes()).digest())
    return digest.hexdigest()


def means(results):
    metrics = {}
    for metric in next(iter(results.values())):
        values = [record[metric] for record in results.values() if record[metric] is not None]
        metrics[metric] = float(np.mean(values)) if values else None
    return metrics


def paired_difference(target, reference, metric, draws):
    a = np.array([np.nan if target[p][metric] is None else target[p][metric] for p in target])
    b = np.array([np.nan if reference[p][metric] is None else reference[p][metric] for p in target])
    difference = a - b
    valid = np.isfinite(difference)
    if not valid.any():
        return {"difference": None, "patients": 0, "descriptive_95_interval": None}
    samples = difference[draws]
    with_value = np.isfinite(samples).any(axis=1)
    interval = np.quantile(np.nanmean(samples[with_value], axis=1), [0.025, 0.975]) if with_value.any() else None
    return {"difference": float(difference[valid].mean()), "patients": int(valid.sum()),
            "descriptive_95_interval": interval.tolist() if interval is not None else None}


def compare(results, settings, seed):
    target = results["paired_adjacent"]
    rng = np.random.default_rng(seed)
    bootstrap = rng.integers(0, len(target), size=(4000, len(target)))
    metrics = ("small_miss_rate", "bordering_absent_false_positive_rate", "small_dice", "global_dice")
    contrasts = {}
    for name in settings["reference_methods"]:
        effect = {key: paired_difference(target, results[name], key, bootstrap) for key in metrics}
        miss, fp, _, overall = (effect[key]["difference"] for key in metrics)
        minimum = settings["minimum_rate_reduction"]
        guard = settings["maximum_global_dice_drop"]
        passes = (None not in (miss, fp, overall) and miss <= -minimum and fp <= -minimum and overall >= -guard)
        contrasts[name] = {"passes": passes, "effects": effect}
    return contrasts


def run_block(args, settings, cache, block, device):
    seed, train_ids, assessment_ids = block
    dest = args.output / f"block-{seed}"
    dest.mkdir(parents=True)
    training = {patient: _load_case(cache[patient]) for patient in train_ids}
    assessment = {patient: _load_case(cache[patient]) for patient in assessment_ids}
    selections = {name: {} for name in settings["methods"]}
    allocations = {}
    for patient, case in training.items():
        chosen, detail = allocate(case, seed, settings["slices_per_patient"])
        allocations[patient] = detail
        for name, method in selections.items():
            method[patient] = [int(index) for index in chosen[name]]
            if len(set(method[patient])) != settings["slices_per_patient"]:
                raise ValueError(f"Incomplete {name} annotation budget")
    save_json(dest / "assigned_slices.json", {"selections": selections, "allocations": allocations})
    save_json(dest / "patient_roles.json", {"training": train_ids, "development_assessment": assessment_ids})
    results = {}
    for name, selected in selections.items():
        state = train_sparse(training, selected, [], steps=settings["steps"], seed=seed, device=device,
                             batch_size=settings["batch_size"])
        outcomes = {}
        for (patient, case), probabilities in zip(assessment.items(),
                                                predict_many(state, assessment.values(), device), strict=True):
            outcomes[patient] = patient_metrics(case, probabilities.argmax(axis=1))
        results[name] = outcomes
        save_json(dest / f"{name}.json", outcomes)
        log("method_complete", block=seed, method=name, labeled_slices=sum(map(len, selected.values())),
            training_seconds=round(state.seconds, 2), metrics=means(outcomes))

    summary = {"seed": seed, "train_patients": len(train_ids), "development_assessment_patients": len(assessment_ids),
               "annotated_slices_per_arm": len(train_ids) * settings["slices_per_patient"],
               "methods": {name: means(result) for name, result in results.items()},
               "contrasts": compare(results, settings, seed), "held_out_test_used": False}
    save_json(dest / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--block", type=int, choices=(2025, 2026), help="Run one block on this device")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    settings = load_json(args.registration)
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    if len(cache) != 100:
        parser.error("Use the 100-patient official ACDC training cache")
    blocks = split_development(cache, settings)
    args.output.mkdir(parents=True)
    device = configure(args.device)
    save_json(args.output / "run.json", {"registration": settings,
                                        "registration_sha256": hashlib.sha256(args.registration.read_bytes()).hexdigest(),
                                        "implementation_sha256": implementation_digest(),
                                        "cache_sha256": implementation_cache_digest(cache, blocks),
                                        "pytorch": torch.__version__,
                                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                        "held_out_test_used": False})
    for block in blocks:
        if args.block is None or block[0] == args.block:
            run_block(args, settings, cache, block, device)


if __name__ == "__main__":
    main()
