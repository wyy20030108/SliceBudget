"""Frozen per-class decisions and six-patient source cutoff sensitivity."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_decision_curves import matched_recall, patient_histograms
from audit_external_transfer import digest
from experiment import load_json, log, save_json
from run_matched_slice_budget import calibrate, choose_threshold, collect
from segment import configure
from summarize_boundary_coverage import bootstrap_weights, describe


CLASSES = ("RV", "MYO", "LV")
GROUPS = ("small", "absent", "near_absent")
FPR = 0.20


def aligned_records(records, order, subsets):
    return [{CLASSES[label - 1]: {name: sorted(map(float, groups[name])) for name in subsets}
             for label, groups in records[patient][0].items()} for patient in order]


def score(args):
    run = load_json(args.fit / "run.json")
    design = load_json(args.design)
    seed, fold = run["seed"], run["fold"]
    donors = run["detection_calibration"]
    official = sorted(args.assessment.glob("patient*.npz"))
    target_ids = [p.stem for p in official]
    source_roles = run["training"] + run["reliability_fit"] + donors + run["development"]
    if (seed not in design["seeds"] or fold not in design["folds"] or len(donors) != 6
            or len(set(source_roles)) != 72 or len(official) != 50 or set(target_ids) & set(source_roles)):
        raise ValueError("Case roles do not match the locked source and official cohorts")
    source = {p: _load_case(args.source / f"{p}.npz") for p in donors}
    target = {p.stem: _load_case(p) for p in official}
    device = configure("cuda")
    ledger = {"seed": seed, "fold": fold, "design_sha256": digest(args.design), "arms": {}}
    for arm in design["arms"]:
        path = args.fit / f"{arm}.pt"
        previous = load_json(args.fit / f"{arm}.json")
        official_record = load_json(args.archived / f"{arm}.json")
        if previous["weight_sha256"] != digest(path):
            raise ValueError(f"Checkpoint mismatch for {arm}")
        model = SimpleNamespace(teacher=torch.load(path, map_location="cpu", weights_only=True))
        source_scores = collect(source, model, device)
        thresholds, support = calibrate(source_scores, FPR)
        if any(support[c] != previous["calibration_support"][str(c)]
               or abs(thresholds[c] - previous["cutoffs"][str(c)]) > 1e-6 for c in thresholds):
            raise ValueError(f"Archived source cutoff/support mismatch for {arm}")
        target_scores = collect(target, model, device)
        for pid, (groups, _, _) in target_scores.items():
            for name in GROUPS:
                margin = [float(x) - previous["cutoffs"][str(c)]
                          for c in thresholds for x in groups[c][name]]
                locked = official_record["scores"][pid][name]
                if len(margin) != len(locked) or not np.allclose(margin, locked, atol=1e-6, rtol=0):
                    raise ValueError(f"Archived target score/support mismatch for {arm}")
        ledger["arms"][arm] = {
            "thresholds": {CLASSES[c - 1]: previous["cutoffs"][str(c)] for c in thresholds},
            "support": {CLASSES[c - 1]: support[c] for c in thresholds},
            "source": aligned_records(source_scores, donors, ("absent",)),
            "target": aligned_records(target_scores, target_ids, GROUPS),
        }
        log("boundary_detector_replay_verified", fold=fold, seed=seed, arm=arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(ledger, handle, separators=(",", ":"), allow_nan=False)


def sample_cutoffs(arm, source_weights):
    values = {label: [row[label]["absent"] for row in arm["source"]] for label in CLASSES}
    pooled = [np.concatenate(rows) for rows in zip(*values.values(), strict=True)]
    results, report = {}, {}
    for label, by_patient in values.items():
        fallback = (source_weights @ np.asarray([len(row) > 0 for row in by_patient])) < 5
        if arm["support"][label] != sum(map(bool, by_patient)):
            raise ValueError("Source ledger support differs from recorded support")
        cutoffs = []
        for use_pool, weights in zip(fallback, source_weights, strict=True):
            rows = pooled if use_pool else by_patient
            chosen = [row for row, count in zip(rows, weights, strict=True) for _ in range(count) if len(row)]
            cutoffs.append(choose_threshold(chosen, FPR) if chosen else np.nan)
        results[label] = np.asarray(cutoffs)
        archived = arm["thresholds"][label]
        if abs(cutoffs[0] - archived) > 1e-6:
            raise ValueError("Replayed cutoff disagrees with the archive")
        results[label][0] = archived
        report[label] = {**describe(results[label]), "source_negative_patients": arm["support"][label],
                         "fallback_fraction": float(fallback[1:].mean()), "original_fallback": bool(fallback[0])}
    return results, report


def macro_decision(group_arrays, cutoffs, weights, labels):
    hits = np.zeros_like(weights, dtype=float)
    counts = np.zeros(weights.shape[1])
    for label in labels:
        for patient, sorted_scores in enumerate(group_arrays[label]):
            scores = np.asarray(sorted_scores)
            if len(scores):
                hits[:, patient] += len(scores) - np.searchsorted(scores, cutoffs[label], side="right")
                counts[patient] += len(scores)
    available = counts > 0
    numer = np.sum(weights[:, available] * (hits[:, available] / counts[available]), axis=1)
    denom = weights[:, available].sum(axis=1)
    rate = np.divide(numer, denom, out=np.full(len(denom), np.nan), where=denom > 0)
    for label in labels:
        rate[~np.isfinite(cutoffs[label])] = np.nan
    return rate


def outcomes(arm, weights, source_weights):
    cutoffs, source_report = sample_cutoffs(arm, source_weights)
    groups = {name: {label: [row[label][name] for row in arm["target"]] for label in CLASSES}
              for name in GROUPS}
    matched, frozen, joint = {}, {}, {}
    threshold_fixed = {label: np.full(len(weights), arm["thresholds"][label]) for label in CLASSES}
    for unit, labels in [*((c, (c,)) for c in CLASSES), ("pooled", CLASSES)]:
        margins = {}
        for group, rows in groups.items():
            margins[group] = [np.concatenate([np.asarray(arr) - threshold_fixed[label][0]
                                              for label, arr in zip(labels, person, strict=True)]).tolist()
                              for person in zip(*(rows[label] for label in labels), strict=True)]
        matched[unit] = {name: matched_recall(patient_histograms(margins["small"], margins[name]), weights, (FPR,))[:, 0]
                         for name in GROUPS[1:]}
        frozen[unit] = {name: macro_decision(rows, threshold_fixed, weights, labels) for name, rows in groups.items()}
        joint[unit] = {name: macro_decision(rows, cutoffs, weights, labels) for name, rows in groups.items()}
    for modes in (matched, frozen, joint):
        names = modes[CLASSES[0]]
        modes["equal_class"] = {metric: np.mean([modes[c][metric] for c in CLASSES], axis=0) for metric in names}
    support = {c: {name: {"patients": int(sum(map(bool, rows[c]))), "pairs": sum(map(len, rows[c]))}
                   for name, rows in groups.items()} for c in CLASSES}
    return {"matched": matched, "frozen_target_only": frozen, "frozen_source_and_target": joint}, source_report, support


def combine(rows):
    return {mode: {unit: {metric: np.mean([row[mode][unit][metric] for row in rows], axis=0) for metric in values}
                   for unit, values in units.items()} for mode, units in rows[0].items()}


def summary(rows, baseline=None):
    return {mode: {unit: {metric: describe(series if baseline is None else series - baseline[mode][unit][metric])
                          for metric, series in values.items()} for unit, values in units.items()}
            for mode, units in rows.items()}


def package_results(rows, arms):
    return {"methods": {name: summary(value) for name, value in rows.items()},
            "minus_image_difference": {name: summary(rows[name], rows[arms[0]]) for name in arms[1:]}}


def replay(args):
    design = load_json(args.design)
    fits = {}
    for path in args.files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            fit = json.load(handle)
        key = (fit["seed"], fit["fold"])
        if key in fits or digest(args.design) != fit["design_sha256"]:
            raise ValueError("Duplicate ledger or mismatched protocol")
        fits[key] = fit["arms"]
    seeds, folds, arms = design["seeds"], design["folds"], design["arms"]
    if set(fits) != {(s, f) for s in seeds for f in folds}:
        raise ValueError("Need every seed and fold")
    rng = np.random.default_rng(20261116)
    weights = bootstrap_weights(50, args.draws, rng)
    donors = {fold: bootstrap_weights(6, args.draws, rng) for fold in folds}
    calibration, seed_results, target_support = {}, {}, None
    for seed in seeds:
        by_fold = {}
        for fold in folds:
            grouped, reports = {}, {}
            for arm in arms:
                grouped[arm], reports[arm], support = outcomes(fits[seed, fold][arm], weights, donors[fold])
                if target_support is not None and support != target_support:
                    raise ValueError("Official reference support changed")
                target_support = support
            calibration[f"seed-{seed}-fold-{fold}"] = reports
            by_fold[fold] = grouped
        seed_results[seed] = {arm: combine([by_fold[fold][arm] for fold in folds]) for arm in arms}
    results = {arm: combine([seed_results[seed][arm] for seed in seeds]) for arm in arms}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"design_sha256": digest(args.design), "draws": args.draws,
                           "target_patients": 50, "target_support": target_support,
                           "source_cutoffs": calibration, "fixed_seed_joint": package_results(results, arms),
                           "by_seed": {str(seed): package_results(seed_results[seed], arms) for seed in seeds}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    gather = actions.add_parser("score")
    for field in ("source", "assessment", "fit", "archived", "design", "output"):
        gather.add_argument(f"--{field}", type=Path, required=True)
    aggregate = actions.add_parser("replay")
    aggregate.add_argument("--files", type=Path, nargs="+", required=True)
    aggregate.add_argument("--design", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    (score if args.action == "score" else replay)(args)


if __name__ == "__main__":
    main()
