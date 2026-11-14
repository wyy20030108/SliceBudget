"""Pool the supplemental ACDC acquisition controls without exporting patient rows."""

import argparse
from pathlib import Path

from experiment import load_json, save_json
from summarize_annotation_evidence import summary


ARMS = ("uniform_random", "pooled_image_farthest", "targeted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--official-prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    folds = [args.runs / f"fold-{fold}" for fold in (0, 1)]
    plans = [load_json(path / "audit_plan.json") for path in folds]
    if [plan["fold"] for plan in plans] != [0, 1] or any(plan["masks_per_arm"] != 160 for plan in plans):
        raise ValueError("Expected the two 160-mask acquisition controls")
    arms = {arm: [load_json(path / f"{arm}.json") for path in folds] for arm in ARMS[:-1]}
    arms["targeted"] = [load_json(args.official_prior / f"image-{fold}" / "targeted.json") for fold in (0, 1)]
    for index, plan in enumerate(plans):
        if any(set(arms[arm][index]["assessment"]) != set(plan["official"]) for arm in ARMS):
            raise ValueError("The paired results must use the frozen ACDC test cohort")
    result = summary(arms, ARMS, ARMS[:-1])
    result["scope"] = "Post-hoc controls at the frozen 160-mask budget on the ACDC official test cohort"
    result["registration_sha256"] = plans[0]["registration_sha256"]
    result["random_selection_seed"] = plans[0]["random_seed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
