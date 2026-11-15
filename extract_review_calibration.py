"""Archive frozen source calibration scores for patient resampling."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from acdc_data import _load_case
from audit_external_transfer import frozen_protocol, replay_manifest
from experiment import load_json, save_json
from run_matched_slice_budget import calibrate, collect
from segment import configure


ARMS = ("targeted", "uncertain", "pooled_image_farthest")


def extract(args):
    settings, cache, _, source = frozen_protocol(args)
    replay_manifest(args.core_weights, source)
    extra = load_json(args.extra_weights / "source_replay.json")
    if (extra["fold"] != args.fold or extra["calibration"] != source["calibration"]
            or extra["registration_sha256"] != source["registration_sha256"]):
        raise ValueError("Baseline checkpoints belong to a different source protocol")
    patients = source["calibration"]
    cohort = {patient: _load_case(cache[patient]) for patient in patients}
    target = settings["calibration_false_positive_rate"]
    output = {"fold": args.fold, "patients": patients, "threshold_target": target, "arms": {}}
    device = configure("cuda")
    for arm in ARMS:
        is_baseline = arm == "pooled_image_farthest"
        folder = args.extra_weights if is_baseline else args.core_weights
        reference = (args.baseline_runs if is_baseline else args.official_original) / f"{arm}.json"
        expected = load_json(reference)["thresholds"]
        state = SimpleNamespace(teacher=torch.load(folder / f"{arm}.pt", map_location="cpu", weights_only=True))
        records = collect(cohort, state, device)
        cutoffs, support = calibrate(records, target)
        delta = max(abs(cutoffs[int(label)] - value) for label, value in expected.items())
        if delta > 1e-6:
            raise ValueError(f"Frozen calibration scores do not reproduce the archived {arm} cutoffs: {delta}")
        output["arms"][arm] = {"thresholds": expected, "support": support, "max_abs_delta": delta,
                                "by_patient": {patient: {label: row["absent"].tolist() for label, row in groups.items()}
                                               for patient, (groups, _, _) in records.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "registration", "original", "official-original", "baseline-runs",
                 "core-weights", "extra-weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--fold", required=True, type=int, choices=(0, 1))
    extract(parser.parse_args())


if __name__ == "__main__":
    main()
