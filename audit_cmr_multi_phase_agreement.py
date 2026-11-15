"""Replay the frozen cine analysis on cases with workbook-concordant phase proxies."""

import argparse
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from audit_external_cine import replay
from audit_external_transfer import digest
from experiment import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--files", required=True, nargs="+", type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    coverage = load_json(args.coverage)
    patient_order = sorted(coverage["cases"])
    indices = [i for i, patient in enumerate(patient_order)
               if abs(coverage["cases"][patient]["released_lvef"]
                      - coverage["cases"][patient]["reference_extrema_lvef"]) <= 0.000051]
    if len(indices) != coverage["released_lvef_check"]["within_four_decimals"]:
        raise ValueError("Selected subgroup disagrees with the pre-inference coverage audit")
    with TemporaryDirectory(prefix="cine-phase-replay-") as folder:
        staged_paths = []
        for original in args.files:
            with gzip.open(original, "rt", encoding="utf-8") as handle:
                fit = json.load(handle)
            if (fit["patients"] != len(patient_order) or fit["cohort_sha256"] != digest(args.coverage)):
                raise ValueError("Ledger is not aligned to the frozen full target cohort")
            if any(fit["missing_on_resized_grid"].values()):
                raise ValueError("A subset-specific grid-loss audit would be required")
            for arm in fit["arms"].values():
                for field in ("target", "dice"):
                    arm[field] = [arm[field][i] for i in indices]
            fit["patients"] = len(indices)
            newpath = Path(folder, original.name)
            with gzip.open(newpath, "wt", encoding="utf-8") as handle:
                json.dump(fit, handle, allow_nan=False)
            staged_paths.append(newpath)
        replay(SimpleNamespace(files=staged_paths, design=args.design, output=args.output, draws=4000))
    result = load_json(args.output)
    result["scope"] = ("Sensitivity on original registered inputs with absolute fractional LVEF "
                       "discrepancy at most 0.000051. Cases chosen from the pre-inference coverage audit.")
    save_json(args.output, result)


if __name__ == "__main__":
    main()
