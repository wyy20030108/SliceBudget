"""Compare the same ACDC acquisitions before and after common budget expansion."""

import argparse
import json
from pathlib import Path

from summarize_annotation_evidence import read, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "original", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    methods = ("interval", "uncertain", "targeted")
    old = [args.original / f"image-{index}" for index in (0, 1)]
    extended = [args.experiment / f"budget-fold-{index}" for index in (0, 1)]
    calibration_costs = []
    for original, folder in zip(old, extended, strict=True):
        result = read(folder / "summary.json")
        added = read(folder / "audit_plan.json")
        previous = read(original / "frozen_acquisitions.json")["arms"]
        assert result["training_budget"] == 240
        for name in methods:
            patients = added["selections"][name]
            if sum(map(len, patients.values())) != 240:
                raise ValueError("Mismatched 240-mask allocation")
            for patient, positions in patients.items():
                if set(positions) != set(previous[name][patient]) | set(added["common_additions"][patient]):
                    raise ValueError("Common additional masks must be identical")
        calibration_costs.append(result["calibration_budget"])
    output = {"condition": "post-hoc, same official ACDC patients, common additional positions",
              "calibration_masks_per_fold": calibration_costs}
    for masks, folders in ((160, old), (240, extended)):
        results = {arm: [read(folder / f"{arm}.json") for folder in folders] for arm in methods}
        output[str(masks)] = summary(results, methods, ("interval", "uncertain"))
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
