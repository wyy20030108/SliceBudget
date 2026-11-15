"""Rebuild supplemental ACDC checkpoints from the archived acquisition plans."""

import argparse
from pathlib import Path

import torch

from acdc_data import _load_case
from audit_external_transfer import digest, frozen_protocol
from experiment import load_json, log, save_json
from run_matched_slice_budget import calibrate, collect, rates
from segment import configure
from sparse_train import train_sparse


IMAGE_ARMS = ("pooled_image_farthest", "uniform_random")


def save_teacher(path, model):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({key: tensor.cpu() for key, tensor in model.teacher.items()}, temporary)
    temporary.replace(path)
    return digest(path)


def replay(args):
    settings, cache, plan, source = frozen_protocol(args)
    official_run = load_json(args.official_original / "run.json")
    test = {path.stem: path for path in args.assessment.glob("patient*.npz")}
    case_ids = sorted(test)
    if len(test) != 50 or case_ids != official_run["assessment"] or set(test) & cache.keys():
        raise ValueError("Replay requires the same disjoint 50-patient official test cohort")
    baseline_path = args.baselines / "audit_plan.json"
    mixed_path = args.factorial / "audit_plan.json"
    baseline_plan = load_json(baseline_path)
    factorial_plan = load_json(mixed_path)
    expected = {"fold": args.fold, "seed": source["seed"], "training": source["training"],
                "calibration": source["calibration"], "registration_sha256": source["registration_sha256"],
                "source_acquisitions_sha256": source["frozen_acquisitions_sha256"], "official": case_ids,
                "masks_per_arm": 160, "initial": plan["warmup"]}
    if any(baseline_plan[key] != value for key, value in expected.items()):
        raise ValueError("Archival image-only control plan does not match the source")
    if factorial_plan["registration_sha256"] != source["registration_sha256"]:
        raise ValueError("Archival mixed control plan does not match the source registration")

    training = {patient: _load_case(cache[patient]) for patient in source["training"]}
    calibration = {patient: _load_case(cache[patient]) for patient in source["calibration"]}
    official = {patient: _load_case(test[patient]) for patient in case_ids}
    arms = {
        arm: (baseline_plan["choices"][arm], args.baselines / f"{arm}.json")
        for arm in IMAGE_ARMS
    }
    arms["mixed_image"] = (factorial_plan["choices"]["mixed_image"], args.factorial / "mixed_image-official.json")
    for arm, (selection, _) in arms.items():
        if list(selection) != sorted(training):
            raise ValueError(f"Unexpected archived patient order for {arm}")
        for patient, indexes in selection.items():
            if (len(set(indexes)) != 4 or not set(plan["warmup"][patient]).issubset(indexes)
                    or sorted(training[patient].phase[indexes].tolist()) != [0, 0, 1, 1]):
                raise ValueError(f"Unexpected archived mask assignments for {arm}/{patient}")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source_replay.json"
    fixed = {**source, "pytorch": torch.__version__, "assessment_patients": len(test),
             "baseline_plan_sha256": digest(baseline_path), "factorial_plan_sha256": digest(mixed_path)}
    manifest = load_json(manifest_path) if manifest_path.exists() else dict(fixed, verification={})
    if any(manifest[key] != value for key, value in fixed.items() if key != "source_thresholds"):
        raise ValueError("Existing replay output belongs to a different source protocol")
    pending = {arm: info for arm, info in arms.items() if arm not in manifest["verification"]
               or not (args.output / f"{arm}.pt").is_file()}
    if not pending and (args.output / "warmup.pt").is_file():
        return

    device = configure("cuda")
    options = {"seed": source["seed"], "device": device, "batch_size": settings["batch_size"]}
    common = train_sparse(training, plan["warmup"], [], steps=settings["warmup_updates"], **options)
    manifest["warmup_checkpoint_sha256"] = save_teacher(args.output / "warmup.pt", common)
    save_json(manifest_path, manifest)
    log("reviewer_control_warmup_complete", fold=args.fold)
    for arm, (selection, archived) in pending.items():
        target = load_json(archived)
        model = train_sparse(training, selection, [], steps=settings["additional_updates"], start=common, **options)
        threshold, support = calibrate(collect(calibration, model, device), settings["calibration_false_positive_rate"])
        archived_threshold = {int(key): value for key, value in target["thresholds"].items()}
        maximum_threshold_delta = max(abs(threshold[key] - value) for key, value in archived_threshold.items())
        if maximum_threshold_delta > 0.001 or support != {int(k): v for k, v in target["support"].items()}:
            raise ValueError(f"Calibration replay differs for {arm}, maximum threshold delta {maximum_threshold_delta}")
        observed = rates(collect(official, model, device), archived_threshold)
        expected = target["assessment"]
        if list(observed) != list(expected):
            raise ValueError(f"Assessment patient order differs for {arm}")
        maximum_outcome_delta = 0.0
        for patient, row in observed.items():
            for metric, value in row.items():
                reference = expected[patient][metric]
                if value is None and reference is None:
                    continue
                if value is None or reference is None:
                    raise ValueError(f"Assessment support differs for {arm}/{patient}/{metric}")
                maximum_outcome_delta = max(maximum_outcome_delta, abs(value - reference))
        if maximum_outcome_delta > 1e-6:
            raise ValueError(f"Official ACDC replay differs for {arm}, maximum outcome delta {maximum_outcome_delta}")
        manifest["verification"][arm] = {
            "thresholds": archived_threshold,
            "calibration_negative_patient_support": support,
            "max_abs_threshold_delta": maximum_threshold_delta,
            "max_abs_patient_outcome_delta": maximum_outcome_delta,
            "archived_assessment_sha256": digest(archived),
            "checkpoint_sha256": save_teacher(args.output / f"{arm}.pt", model),
        }
        save_json(manifest_path, manifest)
        log("reviewer_control_verified", fold=args.fold, arm=arm, threshold_delta=maximum_threshold_delta,
            outcome_delta=maximum_outcome_delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for path in ("data", "registration", "original", "official-original", "assessment", "baselines",
                 "factorial", "output"):
        parser.add_argument(f"--{path}", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
