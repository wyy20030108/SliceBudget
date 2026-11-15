"""Audit loss matching and replay class-wise ACDC scores for the swap control."""

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np

from acdc_data import _load_case
from audit_decision_curves import matched_recall, patient_histograms
from experiment import load_json, save_json
from summarize_boundary_coverage import (PRIMARY_FPR, SUBSETS, average_seeds, bootstrap_weights,
                                         contrast, read_cohort)


def swap_support(root, design):
    records, by_fit = [], {}
    for seed in design["seeds"]:
        for fold in design["folds"]:
            plan = load_json(root / "development" / f"seed-{seed}-fold-{fold}" / "frozen_acquisitions.json")
            baseline, bounded, control = (plan["arms"][arm] for arm in
                                          ("pooled_image_farthest", "bounded", "image_loss_matched"))
            swaps = plan["image_loss_matches"]
            if len(swaps) != design["requested_swaps"] or len({(r["patient"], r["phase"]) for r in swaps}) != len(swaps):
                raise ValueError("Expected eight unique patient-phase slots")
            replaced = dict((p, set(indices)) for p, indices in baseline.items())
            retargeted = dict((p, set(indices)) for p, indices in baseline.items())
            for row in swaps:
                p, old = row["patient"], row["old"]
                replaced[p].remove(old)
                retargeted[p].remove(old)
                replaced[p].add(row["control"])
                retargeted[p].add(row["bounded"])
            if any(replaced[p] != set(control[p]) or retargeted[p] != set(bounded[p]) for p in baseline):
                raise ValueError("Logged swaps do not exactly reconstruct both query rosters")
            by_fit[f"{seed}/fold{fold}"] = summarize_losses(swaps)
            records.extend(swaps)
    return {"all_fits": summarize_losses(records), "by_fit": by_fit}


def summarize_losses(rows):
    bounded = np.array([r["bounded_loss"] for r in rows])
    control = np.array([r["control_loss"] for r in rows])
    delta = control - bounded
    return {"matched_slots": len(rows), "distinct_replacements": sum(r["bounded"] != r["control"] for r in rows),
            "slots_by_phase": {phase: sum(r["phase"] == phase for r in rows) for phase in (0, 1)},
            "mean_bounded_distance_loss": float(bounded.mean()),
            "mean_control_distance_loss": float(control.mean()),
            "mean_absolute_loss_mismatch": float(abs(delta).mean()),
            "maximum_absolute_loss_mismatch": float(abs(delta).max())}


def class_counts(case):
    """Replicate the reference supports of run_matched_slice_budget.collect."""
    result = {}
    for label in (1, 2, 3):
        area = np.count_nonzero(case.masks == label, axis=(1, 2))
        small = np.zeros(len(area), dtype=bool)
        neighbor = np.zeros(len(area), dtype=bool)
        for phase in (0, 1):
            same = case.phase == phase
            small |= same & (area > 0) & (4 * area <= area[same].max())
            positions = case.slice_index[same].astype(int)
            marked = case.slice_index[same & small].astype(int)
            neighbor[same] = np.isin(positions - 1, marked) | np.isin(positions + 1, marked)
        result[label] = dict(zip(SUBSETS, map(np.count_nonzero, (small, area == 0, (area == 0) & neighbor)),
                                 strict=True))
    return result


def by_class(assessment, class_support, roster):
    result = {label: {s: [] for s in SUBSETS} for label in (1, 2, 3)}
    for patient in roster:
        for subset in SUBSETS:
            scores = assessment["scores"][patient][subset]
            counts = [class_support[patient][label][subset] for label in (1, 2, 3)]
            if sum(counts) != len(scores):
                raise ValueError("Mask-derived class supports disagree with the archived score lengths")
            for label, parts in enumerate(np.split(np.asarray(scores), np.cumsum(counts)[:-1]), 1):
                result[label][subset].append(parts.tolist())
    return result


def fit_metrics(assessment, supports, roster, weights):
    cells = by_class(assessment, supports, roster)
    outcomes = {}
    for label, name in enumerate(("RV", "MYO", "LV"), 1):
        for subset in SUBSETS[1:]:
            histogram = patient_histograms(cells[label]["small"], cells[label][subset])
            outcomes[f"{name}_{subset}"] = matched_recall(histogram, weights, (PRIMARY_FPR,))[:, 0]
    for subset in SUBSETS[1:]:
        outcomes[f"equal_class_{subset}"] = np.mean([outcomes[f"{name}_{subset}"] for name in ("RV", "MYO", "LV")], axis=0)
    return outcomes


def matched_roc(root, design, digest, assessment, draws):
    fits = read_cohort(root / "official", root / "development", design, digest, "official", False)
    roster = next(iter(fits.values()))[0]["roster"]
    supports = {p: class_counts(_load_case(assessment / f"{p}.npz")) for p in roster}
    weights = bootstrap_weights(len(roster), draws, np.random.default_rng(20261117))
    all_seed = {}
    for seed, rows in fits.items():
        all_seed[seed] = {}
        for arm in design["arms"]:
            fold_results = [fit_metrics(row["arms"][arm], supports, roster, weights) for row in rows]
            all_seed[seed][arm] = {k: (v + fold_results[1][k]) / 2 for k, v in fold_results[0].items()}
    return {"test_patients": len(roster),
            "reference_support": {
                name: {subset: {"patients": sum(item[label][subset] > 0 for item in supports.values()),
                                "pairs": int(sum(item[label][subset] for item in supports.values()))}
                       for subset in SUBSETS}
                for label, name in enumerate(("RV", "MYO", "LV"), 1)},
            "matched_fpr": PRIMARY_FPR,
            "fixed_seed_joint": contrast(average_seeds(all_seed), design["contender"]),
            "by_seed": {seed: contrast(row, design["contender"]) for seed, row in all_seed.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--assessment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    design = load_json(args.design)
    digest = sha256(args.design.read_bytes()).hexdigest()
    save_json(args.output, {"design_sha256": digest, "bootstrap_draws": args.draws,
                            "loss_matching": swap_support(args.root, design),
                            "official_classwise": matched_roc(args.root, design, digest, args.assessment, args.draws)})


if __name__ == "__main__":
    main()
