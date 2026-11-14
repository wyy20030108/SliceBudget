"""Check whether frozen probability-based acquisitions actually hit requested strata."""

import argparse
import json
from pathlib import Path

import numpy as np

from acdc_data import _load_case
from experiment import load_json, save_json


def row_for(rows, key):
    return rows.setdefault(key, {"nominated": 0, "hit": 0, "typical_positive": 0,
                                 "actual_small": 0, "actual_absent": 0})


def verify_ledger(source, data, combined):
    metadata = load_json(source / "run.json")
    if metadata["held_out_test_used"]:
        raise ValueError("Only inspect training patient selections from the development run")
    frozen = load_json(source / "frozen_acquisitions.json")
    train_ids = set(metadata["training"])
    if train_ids != set(frozen["warmup"]):
        raise ValueError("Frozen selection must match the development training roster")
    cases = {patient: _load_case(data / f"{patient}.npz") for patient in train_ids}
    metrics = {arm: {"fallback": 0, "strata": {}} for arm in ("targeted", "targeted_unbalanced")}
    for phase, paths in frozen["ledger"].items():
        if set(paths) != train_ids:
            raise ValueError("Expected one selection decision per training patient and phase")
        for patient, by_arm in paths.items():
            case = cases[patient]
            for arm, decision in by_arm.items():
                destination = metrics[arm]
                overall = combined.setdefault(arm, {"fallback": 0, "strata": {}})
                when = int(phase)
                label = decision["class"]
                index = decision["index"]
                if int(case.phase[index]) != when or index in frozen["warmup"][patient]:
                    raise ValueError("Invalid acquisition ledger")
                if label is None:
                    destination["fallback"] += 1
                    overall["fallback"] += 1
                    continue
                wanted = decision["predicted_state"]
                counts = np.count_nonzero(case.masks == label, axis=(1, 2))
                area = counts[index]
                peak = counts[case.phase == when].max()
                actual = "absent" if not area else ("small" if area * 4 <= peak else "typical_positive")
                for bucket in (destination, overall):
                    for key in (wanted, f"{phase}/{label}/{wanted}"):
                        row = row_for(bucket["strata"], key)
                        row["nominated"] += 1
                        row["hit"] += int(wanted == actual)
                        row["actual_small"] += int(actual == "small")
                        row["actual_absent"] += int(actual == "absent")
                        row["typical_positive"] += int(actual == "typical_positive")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    combined = {}
    folds = {str(fold): verify_ledger(args.runs / f"fold-{fold}", args.data, combined) for fold in (0, 1)}
    save_json(args.output, {"per_fold": folds, "combined": combined})
    for arm, outcome in combined.items():
        print(arm, outcome["fallback"], json.dumps({key: outcome["strata"][key] for key in ("small", "absent")}, indent=2))


if __name__ == "__main__":
    main()
