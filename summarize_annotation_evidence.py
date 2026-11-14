"""Summarize budgeted annotation studies without counting repeated patients twice."""

import argparse
import json
from pathlib import Path

import numpy as np


OUTCOMES = ("recall", "fpr", "near_fpr", "global_dice", "small_dice")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def patient_average(repeated):
    population = {}
    for run in repeated:
        for patient, values in run.items():
            cohort = population.setdefault(patient, {metric: [] for metric in OUTCOMES})
            for metric in OUTCOMES:
                if values[metric] is not None:
                    cohort[metric].append(values[metric])
    return {patient: {metric: float(np.mean(values)) if values else None for metric, values in outcomes.items()}
            for patient, outcomes in population.items()}


def summary(arms, methods, references):
    patients = {arm: patient_average([run.get("assessment", run.get("development")) for run in arms[arm]])
                for arm in methods}
    all_ids = set(patients[methods[0]])
    if any(set(records) != all_ids for records in patients.values()):
        raise ValueError("Paired comparison requires identical evaluation rosters")
    means = {}
    for arm, records in patients.items():
        means[arm] = {}
        for metric in OUTCOMES:
            values = [row[metric] for row in records.values() if row[metric] is not None]
            means[arm][metric] = {"mean": float(np.mean(values)) if values else None, "n": len(values)}
    contrasts = {}
    rng = np.random.default_rng(20261119)
    treatment = patients[methods[-1]]
    for baseline in references:
        contrasts[baseline] = {}
        for metric in OUTCOMES:
            differences = np.asarray([treatment[patient][metric] - patients[baseline][patient][metric]
                                      for patient in sorted(all_ids) if treatment[patient][metric] is not None
                                      and patients[baseline][patient][metric] is not None])
            if len(differences):
                simulated = differences[rng.integers(0, len(differences), size=(4000, len(differences)))].mean(axis=1)
                contrasts[baseline][metric] = {"difference": float(differences.mean()),
                                               "interval": np.quantile(simulated, [0.025, 0.975]).tolist()}
    pauc = {}
    for arm, runs in arms.items():
        details = [run["assessment_pauc"] for run in runs if "assessment_pauc" in run]
        if details:
            pauc[arm] = {subset: float(np.mean([run[subset] for run in details])) for subset in details[0]}
    return {"patients": len(all_ids), "means": means, "contrasts": contrasts, "mean_run_pauc": pauc}


def acquisition_audit(folders, filenames):
    output = {}
    for folder in folders:
        for arm, filename in filenames.items():
            data = read(folder / filename)["acquisition_audit"]
            total = {name: sum(row[name] for phase in data for row in phase) for name in ("small", "absent", "all")}
            counts = output.setdefault(arm, dict.fromkeys(total, 0))
            for name, value in total.items():
                counts[name] += value
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = {}
    for cohort in ("development", "official"):
        for study in ("truth", "image"):
            if cohort == "official":
                folders = [args.runs / "sparse-official-cloud" / f"{study}-{index}" for index in (0, 1)]
            else:
                name = "sparse-matched-cloud" if study == "truth" else "sparse-image-cloud"
                folders = [args.runs / name / f"fold-{index}" for index in (0, 1)]
            if not all((folder / "summary.json").is_file() for folder in folders):
                continue
            if any(read(folder / "run.json")["held_out_test_used"] != (cohort == "official") for folder in folders):
                raise ValueError("Invalid assessment and training split designation")
            if study == "truth":
                methods = ("interval", "random", "positive", "balanced")
                arms = {arm: [read(folder / f"{seed}-{arm}.json") for folder in folders for seed in (2025, 2026)]
                        for arm in methods}
                outcome = summary(arms, methods, ("interval", "positive"))
            else:
                methods = ("interval", "uncertain", "targeted_unbalanced", "targeted")
                arms = {arm: [read(folder / f"{arm}.json") for folder in folders] for arm in methods}
                outcome = summary(arms, methods, ("interval", "uncertain", "targeted_unbalanced"))
                outcome["acquisition_audit"] = acquisition_audit(folders, {arm: f"{arm}.json" for arm in methods})
                extra = [args.runs / "sparse-factorial-cloud" / f"fold-{index}" for index in (0, 1)]
                if all((folder / "summary.json").is_file() for folder in extra):
                    names = ("mixed_image", "truth_balanced")
                    by_name = {arm: f"{arm}-{cohort}.json" for arm in names}
                    for arm, filename in by_name.items():
                        arms[arm] = [read(folder / filename) for folder in extra]
                    first = summary(arms, methods + ("mixed_image",), methods)
                    second = summary(arms, methods + names, methods + ("mixed_image",))
                    outcome["same_budget_audit"] = {"additional_means": {arm: second["means"][arm] for arm in names},
                                                    "mixed_contrasts": first["contrasts"],
                                                    "truth_contrasts": second["contrasts"],
                                                    "additional_pauc": {arm: second["mean_run_pauc"][arm] for arm in names},
                                                    "acquisition_audit": acquisition_audit(extra, by_name)}
            output[f"{cohort}_{study}"] = outcome
    supervised = [args.runs / "sparse-supervised-cloud" / f"fold-{index}" for index in (0, 1)]
    if all((folder / "summary.json").is_file() for folder in supervised):
        methods = ("interval", "uncertain", "mixed_image", "targeted")
        arms = {arm: [read(folder / f"{arm}.json") for folder in supervised] for arm in methods}
        output["official_supervised_exploratory"] = summary(arms, methods, ("interval", "uncertain", "mixed_image"))
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
