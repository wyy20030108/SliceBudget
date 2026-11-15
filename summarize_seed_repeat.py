"""Compare original and repeated slice acquisition on the same ACDC patients."""

import argparse
import json
from pathlib import Path

from summarize_annotation_evidence import read, summary


METHODS = ("interval", "uncertain", "targeted_unbalanced", "targeted")
SPLIT_FIELDS = ("fold", "training", "calibration", "assessment", "held_out_test_used")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = [args.original / f"image-{fold}" for fold in (0, 1)]
    repeat = [args.repeat / f"fold-{fold}" for fold in (0, 1)]
    repeated_seed = None
    for left, right in zip(source, repeat, strict=True):
        baseline, replication = read(left / "run.json"), read(right / "run.json")
        if not baseline["held_out_test_used"] or any(baseline[key] != replication[key] for key in SPLIT_FIELDS):
            raise ValueError("Original and repeat do not use the same official test protocol")
        if baseline["seed"] == replication["seed"] or repeated_seed not in (None, replication["seed"]):
            raise ValueError("The repeated folds must share a new training and acquisition seed")
        repeated_seed = replication["seed"]
    arms = {method: [read(path / f"{method}.json") for path in repeat] for method in METHODS}
    output = summary(arms, METHODS, METHODS[:-1])
    output["repeat_seed"] = repeated_seed
    output["scope"] = "Post-hoc repeat reselecting masks; same official test patients, fold and within-fold data roles."
    output["by_fold"] = {str(fold): read(path / "summary.json")["methods"] for fold, path in enumerate(repeat)}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
