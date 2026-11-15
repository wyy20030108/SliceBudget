"""Evaluate frozen ACDC slice-acquisition models on independently labeled cine stacks."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_boundary_detector import CLASSES, FPR, aligned_records, combine, outcomes, package_results
from audit_external_transfer import digest
from experiment import load_json, log, save_json
from run_matched_slice_budget import calibrate, collect
from segment import configure, predict_many
from summarize_boundary_coverage import bootstrap_weights, describe


def reference(areas, phases, indices):
    if areas.shape != (len(phases), len(CLASSES)) or areas.dtype.kind not in "iu" or np.any(areas < 0):
        raise ValueError("Expected native nonnegative class areas, in model-channel order")
    result = {label: {} for label in CLASSES}
    has_annotation = np.any(areas > 0, axis=1)
    for c, label in enumerate(CLASSES):
        small = np.zeros(len(phases), dtype=bool)
        absent = small.copy()
        near = small.copy()
        for phase in (0, 1):
            in_phase = phases == phase
            native = indices[in_phase]
            area = areas[in_phase, c]
            if not len(area) or np.any(np.diff(native) != 1):
                raise ValueError("Incomplete native slice order at ED or ES")
            small_phase = (area > 0) & (4 * area <= area.max())
            absent_phase = (area == 0) & has_annotation[in_phase] & (area.max() > 0)
            near_phase = np.isin(native - 1, native[small_phase]) | np.isin(native + 1, native[small_phase])
            small[in_phase], absent[in_phase] = small_phase, absent_phase
            near[in_phase] = absent_phase & near_phase
        result[label] = {"small": small, "absent": absent, "near_absent": near}
    return result, has_annotation


def assess(cases, references, model, device):
    patient_scores, dice = [], []
    for case, (states, annotated), probability in zip(
            cases, references, predict_many(model, cases, device), strict=True):
        foreground = probability[:, 1:].reshape(len(probability), 3, -1)
        top = max(1, round(foreground.shape[-1] / 100))
        score = np.partition(foreground, -top, axis=-1)[..., -top:].mean(axis=-1)
        prediction = probability.argmax(axis=1)
        row, overlap = {}, {}
        global_dice, small_dice = [], []
        for c, label in enumerate(CLASSES):
            row[label] = {name: sorted(map(float, score[eligible, c])) for name, eligible in states[label].items()}
            estimated, truth = prediction == c + 1, case.masks == c + 1
            hits = 2 * (estimated & truth).sum(axis=(1, 2))
            count = estimated.sum(axis=(1, 2)) + truth.sum(axis=(1, 2))
            by_phase = []
            for phase in (0, 1):
                group = annotated & (case.phase == phase)
                denom = count[group].sum()
                if denom > 0:
                    by_phase.append(float(hits[group].sum() / denom))
            small = [float(hits[i] / count[i]) if count[i] else 0.0
                     for i in np.flatnonzero(states[label]["small"])]
            global_dice.extend(by_phase)
            small_dice.extend(small)
            overlap[label] = {"global": mean(by_phase), "small": mean(small)}
        overlap["pooled"] = {"global": mean(global_dice), "small": mean(small_dice)}
        patient_scores.append(row)
        dice.append(overlap)
    return patient_scores, dice


def mean(values):
    return float(np.mean(values)) if len(values) else None


def score(args):
    design = load_json(args.design)
    id_ = f"seed-{args.seed}-fold-{args.fold}"
    primary = args.bounded / id_
    swap = args.swap / id_
    base_run = load_json(primary / "run.json")
    swap_run = load_json(swap / "run.json")
    if (base_run["seed"], base_run["fold"]) != (args.seed, args.fold):
        raise ValueError("Wrong frozen fit")
    ids = base_run["detection_calibration"]
    if ids != swap_run["detection_calibration"] or len(ids) != 6:
        raise ValueError("Expected the same source cutoff panel in every arm")
    paths = sorted(args.target.glob("case-*.npz"))
    coverage_path = args.target / "coverage.json"
    coverage = load_json(coverage_path)
    if (not paths or set(coverage["cases"]) != {p.stem for p in paths}
            or coverage["patients"] != len(paths)):
        raise ValueError("All inputs must match the frozen target manifest")
    cases, references = [], []
    missing_at_resize = [0, 0, 0]
    for path in paths:
        if digest(path) != coverage["cases"][path.stem]["cache_sha256"]:
            raise ValueError("Target input changed after eligibility review")
        case = _load_case(path)
        with np.load(path, allow_pickle=False) as record:
            areas = record["native_areas"]
            eligible = record["verified_absent"]
        refs, annotated = reference(areas, case.phase, case.slice_index)
        if any(not np.array_equal(refs[label]["absent"], eligible[:, c])
               for c, label in enumerate(CLASSES)):
            raise ValueError("Eligibility must match the native reference audit")
        if not all(np.any(areas[:, c] > 0) for c in range(3)):
            raise ValueError("Case without released reference for every target class")
        for c in range(3):
            missing_at_resize[c] += int(((areas[:, c] > 0)
                                         & ~np.any(case.masks == c + 1, axis=(1, 2))).sum())
        cases.append(case)
        references.append((refs, annotated))
    target_sha = digest(coverage_path)
    sources = {p: _load_case(args.source / f"{p}.npz") for p in ids}
    device = configure("cuda")
    receipt = {"seed": args.seed, "fold": args.fold, "patients": len(paths),
               "design_sha256": digest(args.design), "cohort_sha256": target_sha,
               "missing_on_resized_grid": dict(zip(CLASSES, missing_at_resize, strict=True)), "arms": {}}
    for arm in design["arms"]:
        fit = swap if arm == "image_loss_matched" else primary
        meta, weights = load_json(fit / f"{arm}.json"), fit / f"{arm}.pt"
        if digest(weights) != meta["weight_sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch for {arm}")
        model = SimpleNamespace(teacher=torch.load(weights, map_location="cpu", weights_only=True))
        source_records = collect(sources, model, device)
        cutoffs, support = calibrate(source_records, FPR)
        for c in cutoffs:
            if abs(cutoffs[c] - meta["cutoffs"][str(c)]) > 1e-6 or support[c] != meta["calibration_support"][str(c)]:
                raise ValueError(f"Source-calibration replay drift for {arm}")
        target_records, overlap = assess(cases, references, model, device)
        receipt["arms"][arm] = {"thresholds": {CLASSES[c - 1]: meta["cutoffs"][str(c)] for c in cutoffs},
                                "support": {CLASSES[c - 1]: support[c] for c in cutoffs},
                                "weight_sha256": meta["weight_sha256"],
                                "source": aligned_records(source_records, ids, ("absent",)),
                                "target": target_records, "dice": overlap}
        log("external_cine_fit_scored", fold=args.fold, seed=args.seed, arm=arm, patients=len(paths))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(receipt, handle, separators=(",", ":"), allow_nan=False)


def overlap_bootstrap(rows, weights):
    result = {}
    for label in (*CLASSES, "pooled"):
        result[label] = {}
        for name in ("global", "small"):
            values = np.asarray([row[label][name] for row in rows], dtype=float)
            available = np.isfinite(values)
            numerator = weights[:, available] @ values[available]
            denominator = weights[:, available].sum(axis=1)
            result[label][name] = np.divide(numerator, denominator, out=np.full(len(denominator), np.nan),
                                            where=denominator > 0)
    return result


def replay(args):
    design = load_json(args.design)
    seeds, folds, arms = design["seeds"], design["folds"], design["arms"]
    fits, cohort = {}, None
    for path in args.files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            fit = json.load(handle)
        key = fit["seed"], fit["fold"]
        if key in fits or fit["design_sha256"] != digest(args.design) or set(fit["arms"]) != set(arms):
            raise ValueError("Duplicate or mismatching inference ledger")
        current = (fit["cohort_sha256"], fit["patients"], fit["missing_on_resized_grid"])
        if cohort is not None and current != cohort:
            raise ValueError("All fits must share the same frozen target reference")
        cohort, fits[key] = current, fit["arms"]
    if set(fits) != {(seed, fold) for seed in seeds for fold in folds}:
        raise ValueError("Require every seed and fold exactly once")
    rng = np.random.default_rng(20261118)
    weights = bootstrap_weights(cohort[1], args.draws, rng)
    donors = {fold: bootstrap_weights(6, args.draws, rng) for fold in folds}
    summaries, reported_support, cutoffs = {}, None, {}
    for seed in seeds:
        rows = {arm: [] for arm in arms}
        for fold in folds:
            cutoffs[f"seed-{seed}-fold-{fold}"] = {}
            for arm, item in fits[seed, fold].items():
                stats, source_report, support = outcomes(item, weights, donors[fold])
                if reported_support is not None and support != reported_support:
                    raise ValueError("Target support differs by fit")
                reported_support = support
                for mode in stats.values():
                    mode["equal_ventricular"] = {metric: (val + mode["LV"][metric]) / 2
                                                 for metric, val in mode["RV"].items()}
                stats["dice"] = overlap_bootstrap(item["dice"], weights)
                rows[arm].append(stats)
                cutoffs[f"seed-{seed}-fold-{fold}"][arm] = source_report
        summaries[seed] = {arm: combine(fit_stats) for arm, fit_stats in rows.items()}
    averages = {arm: combine([summaries[seed][arm] for seed in seeds]) for arm in arms}
    equivalents = {name: {metric: describe(averages["bounded"]["matched"][name][metric]
                                            - averages["image_loss_matched"]["matched"][name][metric])
                          for metric in ("absent", "near_absent")}
                   for name in ("pooled", "equal_class", "equal_ventricular", *CLASSES)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"design_sha256": digest(args.design), "cohort_sha256": cohort[0],
                           "patients": cohort[1], "native_positives_lost_on_192_grid": cohort[2],
                           "draws": args.draws, "support": reported_support, "source_cutoffs": cutoffs,
                           "fixed_seed_joint": package_results(averages, arms),
                           "bounded_minus_eight_swap_matched": equivalents,
                           "by_seed": {str(seed): package_results(summaries[seed], arms) for seed in seeds}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    gather = actions.add_parser("score")
    for field in ("source", "target", "bounded", "swap", "design", "output"):
        gather.add_argument(f"--{field}", type=Path, required=True)
    gather.add_argument("--fold", type=int, choices=(0, 1), required=True)
    gather.add_argument("--seed", type=int, choices=(2027, 2028, 2029), required=True)
    aggregate = actions.add_parser("replay")
    aggregate.add_argument("--files", type=Path, nargs="+", required=True)
    aggregate.add_argument("--design", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    (score if args.action == "score" else replay)(args)


if __name__ == "__main__":
    main()
