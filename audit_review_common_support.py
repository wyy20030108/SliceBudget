"""Check ACDC/HVSMR policy rankings after ventricle and area standardization."""

import argparse
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from acdc_data import _frame_pairs, _phase_frames
from audit_decision_curves import OPERATING_POINTS, interval, matched_recall, patient_histograms
from audit_external_transfer import digest
from audit_stratified_decisions import partial_auc
from experiment import load_json, save_json
from run_matched_slice_budget import choose_threshold


ARMS = ("targeted", "uncertain", "pooled_image_farthest")
LABELS = (1, 3)
ALL_LABELS = (1, 2, 3)
SUBSETS = ("small", "absent", "near_absent")
SCOPES = ("native_support", "opposite_ventricle")
AREA_STRATA = ("le10", "gt10_le25")
PRIMARY_FPR = 0.20


def cache_acdc(raw, cache, output):
    """Map native ACDC area to the exact phase and slice order used in inference."""
    patients = sorted(cache.glob("patient*.npz"))
    if len(patients) != 50:
        raise ValueError("Use the 50-patient official ACDC test cache")
    result = {}
    for patient in patients:
        paths = _frame_pairs(raw / patient.stem)
        frames = _phase_frames(raw / patient.stem, paths)
        with np.load(patient, allow_pickle=False) as saved:
            phase, positions = saved["phase"], saved["slice_index"]
        rows = np.zeros((len(phase), len(ALL_LABELS)), dtype=np.int64)
        for code, frame in enumerate(frames):
            mask = np.asarray(nib.load(paths[frame][1]).dataobj)
            if mask.ndim != 3 or not np.array_equal(positions[phase == code], np.arange(mask.shape[2])):
                raise ValueError("Native and cached ACDC z indices do not align")
            rows[phase == code] = np.stack([(mask == label).sum(axis=(0, 1)) for label in ALL_LABELS], axis=1)
        result[patient.stem] = {"area": rows.tolist(), "phase": phase.tolist(), "index": positions.tolist()}
    save_json(output, result)


def reference_scopes(area, phases, positions, published_absent):
    records = {scope: {} for scope in SCOPES}
    for label, opposite in zip(LABELS, LABELS[::-1], strict=True):
        target = area[:, label - 1]
        peak = np.zeros_like(target)
        common_absent = np.zeros(len(target), dtype=bool)
        for phase in np.unique(phases):
            relevant = phases == phase
            peak[relevant] = target[relevant].max()
            common_absent[relevant] = (target[relevant] == 0) & (peak[relevant] > 0) & (area[relevant, opposite - 1] > 0)
        small = (target > 0) & (4 * target <= peak)
        low = small & (10 * target <= peak)
        for scope, absent in zip(SCOPES, (published_absent[:, label - 1], common_absent), strict=True):
            near = np.zeros_like(small)
            for phase in np.unique(phases):
                relevant = phases == phase
                neighbors = positions[relevant & small]
                near[relevant] = absent[relevant] & (np.isin(positions[relevant] - 1, neighbors)
                                                     | np.isin(positions[relevant] + 1, neighbors))
            records[scope][label] = {"small": small, "absent": absent, "near_absent": near,
                                     AREA_STRATA[0]: low, AREA_STRATA[1]: small & ~low}
        if np.any(common_absent & ~published_absent[:, label - 1]):
            raise ValueError("Harmonized absences must be a subset of verified native absences")
    return records


def prepare_domains(acdc, external, native_path, external_cache, reference_groups):
    native = load_json(native_path)
    acdc_roster = sorted(native)
    hvsmr_roster = sorted(external[0]["cache_sha256"])
    groups = load_json(reference_groups)
    if set(groups) != set(hvsmr_roster) or sum(value == "manual_original" for value in groups.values()) != 20:
        raise ValueError("HVSMR reference groups must align and identify the 20 original cases")
    refs = {"acdc": {}, "hvsmr": {}}
    data = {"acdc": [{}, {}], "hvsmr": [{}, {}]}
    for patient in acdc_roster:
        row = native[patient]
        area, phases, indices = (np.asarray(row[key], dtype=np.int64) for key in ("area", "phase", "index"))
        refs["acdc"][patient] = reference_scopes(area, phases, indices, area == 0)
        for index, fold in enumerate(acdc):
            data["acdc"][index][patient] = {}
            for arm in ARMS:
                fit = fold["arms"][arm]
                rows = fit["cohorts"]["official"][patient]["rows"]
                cutoffs = fit["calibration"]["top1"]
                result = {}
                for label in LABELS:
                    class_rows = [row for row in rows if row["label"] == label]
                    if len(class_rows) != len(phases) or not np.array_equal([row["phase"] for row in class_rows], phases):
                        raise ValueError("ACDC archived score order differs from native phase reference")
                    result[label] = np.asarray([row["top1"] for row in class_rows]) - cutoffs[str(label)]["cutoff"]
                data["acdc"][index][patient][arm] = result

    for patient in hvsmr_roster:
        path = external_cache / f"{patient}.npz"
        cache_hash = external[0]["cache_sha256"][patient]
        if external[1]["cache_sha256"][patient] != cache_hash or digest(path) != cache_hash:
            raise ValueError("HVSMR annotation metadata does not match the caches originally scored")
        with np.load(path, allow_pickle=False) as saved:
            area, phase, positions, absent = (saved[key] for key in ("native_areas", "phase", "slice_index", "verified_absent"))
        if not np.all(phase == 0):
            raise ValueError("HVSMR audit uses the static axial cache")
        patient_refs = reference_scopes(area, phase, positions, absent)
        refs["hvsmr"][patient] = patient_refs
        for index, fold in enumerate(external):
            data["hvsmr"][index][patient] = {}
            for arm in ARMS:
                result = {}
                for label in LABELS:
                    cells = fold["methods"][arm]["patients"][patient]["0"][str(label)]
                    margins = np.full(len(phase), np.nan)
                    for subset, values in cells.items():
                        selected = patient_refs[SCOPES[0]][label][subset]
                        if len(values) != selected.sum():
                            raise ValueError("Stored HVSMR score count differs from exact native eligibility")
                        if subset == "near_absent" and not np.array_equal(margins[selected], values):
                            raise ValueError("Stored HVSMR near-absence order differs from all-absence scores")
                        margins[selected] = values
                    result[label] = margins
                data["hvsmr"][index][patient][arm] = result
    refs["hvsmr_manual"] = {p: refs["hvsmr"][p] for p in hvsmr_roster if groups[p] == "manual_original"}
    data["hvsmr_manual"] = data["hvsmr"]
    return refs, data


def bootstrap_weights(n, draws, rng):
    return np.vstack([np.ones(n, dtype=np.int64), rng.multinomial(n, np.full(n, 1 / n), size=draws)])


def calibration_inputs(paths, acdc, external):
    fits = sorted([load_json(path) for path in paths], key=lambda run: run["fold"])
    if [fit["fold"] for fit in fits] != [0, 1] or set(fits[0]["patients"]) & set(fits[1]["patients"]):
        raise ValueError("Expected two disjoint calibration rosters in original source folds")
    for i, fit in enumerate(fits):
        for arm, row in fit["arms"].items():
            archived = row["thresholds"]
            for c in ALL_LABELS:
                centers = (acdc[i]["arms"][arm]["calibration"]["top1"][str(c)]["cutoff"],
                           external[i]["methods"][arm]["source_thresholds"][str(c)])
                if row["max_abs_delta"] > 1e-6 or any(archived[str(c)] != center for center in centers):
                    raise ValueError("Bootstrap inputs do not match archived ACDC and external score centering")
    return fits


def source_cutoffs(fits, draws, rng):
    records = []
    reporting = []
    for fit in fits:
        patients = fit["patients"]
        weights = bootstrap_weights(len(patients), draws, rng)
        if len(patients) != 12 or fit["threshold_target"] != PRIMARY_FPR:
            raise ValueError("Use the original source clinical calibration sets")
        summary = {}
        cutoffs = {}
        for arm, row in fit["arms"].items():
            archived = row["thresholds"]
            values = {c: [np.asarray(row["by_patient"][p][str(c)]) for p in patients] for c in ALL_LABELS}
            pooled = [np.concatenate([values[c][j] for c in ALL_LABELS]) for j in range(len(patients))]
            cutoffs[arm] = {}
            summary[arm] = {}
            for c in LABELS:
                contributing = weights @ np.asarray([len(x) > 0 for x in values[c]])
                used_fallback = contributing < 5
                offsets = np.full(len(weights), np.nan)
                for draw, weight in enumerate(weights):
                    rows = pooled if used_fallback[draw] else values[c]
                    sampled = [row for row, count in zip(rows, weight, strict=True) for _ in range(count) if len(row)]
                    if sampled:
                        offsets[draw] = choose_threshold(sampled, PRIMARY_FPR) - archived[str(c)]
                if abs(offsets[0]) > 1e-6:
                    raise ValueError("Original cutoffs must reproduce at original calibration weights")
                offsets[0] = 0
                cutoffs[arm][c] = offsets
                summary[arm][str(c)] = {"orig": archived[str(c)], "bootstrap_cutoff_ci95": interval(offsets[1:] + archived[str(c)]),
                                         "fallback_fraction": float(used_fallback[1:].mean()),
                                         "n_invalid_draws": int(np.count_nonzero(~np.isfinite(offsets[1:])))}
        records.append(cutoffs)
        reporting.append(summary)
    return records, reporting


def make_ledger(refs, scores, fits, score_comparison):
    mapping = {}
    for cohort in ("acdc", "hvsmr"):
        patients = sorted(refs[cohort])
        mapping[cohort] = {p: f"e{j:03}" for j, p in enumerate(patients)}
    mapping["hvsmr_manual"] = mapping["hvsmr"]
    cohorts = {}
    for cohort, rows in refs.items():
        patients = sorted(rows, key=mapping[cohort].__getitem__)
        cohorts[cohort] = {mapping[cohort][patient]: [
            {scope: {arm: {str(label): {subset: sorted(run[patient][arm][label][kept].tolist())
                                        for subset, kept in refs[cohort][patient][scope][label].items()}
                            for label in LABELS} for arm in ARMS} for scope in SCOPES}
            for run in scores[cohort]] for patient in patients}
    calibration = []
    for fit in fits:
        patients = fit["patients"].copy()
        ids = {patient: f"c{fit['fold']}{j:02d}" for j, patient in enumerate(patients)}
        calibration.append({"fold": fit["fold"], "threshold_target": fit["threshold_target"],
                            "patients": list(ids.values()), "arms": {arm: {"thresholds": item["thresholds"],
                                "by_patient": {ids[p]: {label: sorted(values) for label, values in item["by_patient"][p].items()}
                                               for p in patients}} for arm, item in fit["arms"].items()}})
    return {"schema": 1, "cohorts": cohorts, "frozen_calibration": calibration,
            "legacy_pauc_raw_vs_centered_original_acdc": score_comparison}


def view(patients, index, arm, scope, labels, size):
    return {subset: [[value for label in labels for value in rows[index][scope][arm][str(label)][size if subset == "small" and size else subset]]
                     for rows in patients.values()] for subset in SUBSETS}


def describe(value):
    flat = value.reshape(len(value), -1)
    summary = [{"estimate": float(draws[0]) if np.isfinite(draws[0]) else None, "ci95": interval(draws[1:]),
                "valid_draws": int(np.isfinite(draws[1:]).sum())} for draws in flat.T]
    return summary[0] if value.ndim == 1 else summary


def decision(rows, weights, offsets):
    eligible = np.asarray([len(row) > 0 for row in rows])
    denom = weights @ eligible
    rates = np.stack([(len(row) - np.searchsorted(np.sort(row), offsets, side="right")) / len(row)
                      if len(row) else np.zeros(len(offsets)) for row in rows], axis=1)
    return np.divide((rates * weights).sum(axis=1), denom, out=np.full(len(weights), np.nan),
                     where=(denom > 0) & np.isfinite(offsets))


def evaluate(cohort, weights, scope, labels, size, offsets):
    points = (PRIMARY_FPR,) if size is not None else OPERATING_POINTS
    metrics = {}
    support = None
    for arm in ARMS:
        fits = []
        for index in (0, 1):
            rows = view(cohort, index, arm, scope, labels, size)
            observed = {subset: {"patients": sum(len(item) > 0 for item in items), "pairs": sum(map(len, items))}
                        for subset, items in rows.items()}
            if (support is not None and support != observed) or any(not np.isfinite(row).all() for items in rows.values() for row in items):
                raise ValueError("Invalid or unmatched assessment support across models and folds")
            support = observed
            estimates = {}
            for subset, values in rows.items():
                estimates[f"frozen_{subset}"] = decision(values, weights, np.zeros(len(weights)))
                if len(labels) == 1 and size is None:
                    estimates[f"recalibrated_{subset}"] = decision(values, weights, offsets[index][arm][labels[0]])
            for subset in SUBSETS[1:]:
                estimates[f"matched_{subset}"] = (matched_recall(patient_histograms(rows["small"], rows[subset]), weights, points)
                                                   if support["small"]["pairs"] and support[subset]["pairs"]
                                                   else np.full((len(weights), len(points)), np.nan))
                if size is None:
                    value = partial_auc(rows["small"], rows[subset])
                    estimates[f"centered_pauc_{subset}"] = np.array([np.nan if value is None else value])
            fits.append(estimates)
        metrics[arm] = {name: (value + fits[1][name]) / 2 for name, value in fits[0].items()}
    return metrics, support


def average_metrics(items):
    keys = set.intersection(*(set(item[ARMS[0]]) for item in items))
    return {arm: {key: np.stack([item[arm][key] for item in items]).mean(axis=0) for key in sorted(keys)} for arm in ARMS}


def summaries(metrics):
    return {"methods": {arm: {key: describe(value) for key, value in outcome.items()} for arm, outcome in metrics.items()},
            "targeted_minus": {arm: {key: describe(metrics["targeted"][key] - value)
                                     for key, value in metrics[arm].items()} for arm in ARMS[1:]}}


def pauc_ambiguity(acdc):
    result = {}
    for arm in ARMS:
        by_fold = []
        for fold in acdc:
            fit = fold["arms"][arm]
            patient_rows = list(fit["cohorts"]["official"].values())
            cutoffs = fit["calibration"]["top1"]
            by_score = {}
            for center in (False, True):
                rows = {subset: [[row["top1"] - (cutoffs[str(row["label"])]["cutoff"] if center else 0)
                                  for row in patient["rows"] if row[subset]] for patient in patient_rows] for subset in SUBSETS}
                by_score["class_centered" if center else "raw"] = {
                    subset: partial_auc(rows["small"], rows[subset]) for subset in SUBSETS[1:]}
            by_fold.append(by_score)
        result[arm] = {score: {subset: float(np.mean([fold[score][subset] for fold in by_fold]))
                               for subset in SUBSETS[1:]} for score in by_fold[0]}
    return result


def summarize(args):
    acdc, external = ([load_json(path) for path in files] for files in (args.acdc, args.external))
    if any([run["fold"] for run in folds] != [0, 1] for folds in (acdc, external)):
        raise ValueError("Provide assessment folds in original order 0, 1")
    refs, scores = prepare_domains(acdc, external, args.acdc_native, args.external_cache, args.reference_groups)
    fits = calibration_inputs(args.calibration, acdc, external)
    ledger = make_ledger(refs, scores, fits, pauc_ambiguity(acdc))
    if args.export_ledger:
        args.export_ledger.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.export_ledger, "wt", encoding="utf-8") as stream:
            json.dump(ledger, stream, separators=(",", ":"), allow_nan=False)
    calculate(ledger, args.output, args.draws)


def calculate(ledger, output, draws):
    cohorts = ledger["cohorts"]
    if ledger["schema"] != 1 or {name: len(rows) for name, rows in cohorts.items()} != {
        "acdc": 50, "hvsmr": 60, "hvsmr_manual": 20,
    } or not set(cohorts["hvsmr_manual"]) <= set(cohorts["hvsmr"]):
        raise ValueError("Expected the original ACDC and HVSMR evaluation supports")
    rng = np.random.default_rng(20261120)
    weights = {cohort: bootstrap_weights(len(rows), draws, rng) for cohort, rows in cohorts.items()}
    offsets, calibration = source_cutoffs(ledger["frozen_calibration"], draws, rng)
    report = {"bootstrap_draws": draws, "fpr_points": list(OPERATING_POINTS), "area_fpr_point": PRIMARY_FPR,
              "inference": "Frozen ACDC models. ACDC and HVSMR patients resampled independently; policies and folds share each evaluation sample. Source calibration draws are reused across policies and domains, and are independent across the two disjoint source calibration rosters. Intervals are pointwise and conditional on fixed models.",
              "area_standardization": "Average fixed bin-specific 20% FPR recalls within ventricle, then use equal weights across ventricles. Bin-specific ROC analyses retain the same negative scores for that ventricle.",
              "note_manual": "Original HVSMR manually derived annotation subset is nested within HVSMR; not a distinct cohort.",
              "calibration": calibration, "pauc_raw_vs_centered_original_acdc": ledger["legacy_pauc_raw_vs_centered_original_acdc"],
              "scopes": {}}
    for scope in SCOPES:
        metrics, support = {}, {}
        for cohort, rows in cohorts.items():
            metrics[cohort], support[cohort] = {}, {}
            for labels in ((1,), (3,), LABELS):
                name = {1: "RV", 3: "LV"}[labels[0]] if len(labels) == 1 else "pooled_LV_RV"
                key = (name, "full")
                metrics[cohort][key], support[cohort][key] = evaluate(rows, weights[cohort], scope, labels, None, offsets)
                if len(labels) == 1:
                    for size in AREA_STRATA:
                        key = (name, size)
                        metrics[cohort][key], support[cohort][key] = evaluate(rows, weights[cohort], scope, labels, size, offsets)
            metrics[cohort][("equal_LV_RV", "full")] = average_metrics([metrics[cohort][(name, "full")] for name in ("RV", "LV")])
        area_keys = [(label, size) for label in ("RV", "LV") for size in AREA_STRATA
                     if all(support[cohort][(label, size)]["small"]["patients"] for cohort in ("acdc", "hvsmr"))]
        if set(label for label, _ in area_keys) == {"RV", "LV"}:
            for cohort in cohorts:
                metrics[cohort][("equal_LV_RV", "equal_common_area_bins")] = average_metrics([
                    average_metrics([metrics[cohort][key] for key in area_keys if key[0] == label]) for label in ("RV", "LV")])
        interactions = {}
        for key in metrics["acdc"]:
            source, target = (metrics[cohort][key] for cohort in ("acdc", "hvsmr"))
            interactions["/".join(key)] = {arm: {name: describe((target["targeted"][name] - target[arm][name])
                                                               - (source["targeted"][name] - source[arm][name]))
                                                   for name in target[arm]} for arm in ARMS[1:]}
        report["scopes"][scope] = {
            "common_area_cells": [list(key) for key in area_keys], "domain_interactions": interactions,
            "domains": {cohort: {"patients": len(cohorts[cohort]),
                                 "cells": {"/".join(key): {"support": support[cohort].get(key), **summaries(value)}
                                           for key, value in estimates.items()}}
                        for cohort, estimates in metrics.items()},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="command", required=True)
    cache = tasks.add_parser("cache-acdc-native")
    for name in ("raw", "cache", "output"):
        cache.add_argument(f"--{name}", required=True, type=Path)
    compare = tasks.add_parser("compare")
    for name in ("acdc", "external", "calibration"):
        compare.add_argument(f"--{name}", required=True, type=Path, nargs=2)
    for name in ("acdc-native", "external-cache", "reference-groups", "output"):
        compare.add_argument(f"--{name}", required=True, type=Path)
    compare.add_argument("--export-ledger", type=Path, help="Save compressed numeric ledger with artificial case keys")
    compare.add_argument("--draws", type=int, default=4000)
    replay = tasks.add_parser("replay")
    replay.add_argument("--ledger", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    if args.command == "cache-acdc-native":
        cache_acdc(args.raw, args.cache, args.output)
    elif args.draws < 1:
        parser.error("Specify a positive number of bootstrap draws")
    elif args.command == "compare":
        summarize(args)
    else:
        with gzip.open(args.ledger, "rt", encoding="utf-8") as stream:
            calculate(json.load(stream), args.output, args.draws)


if __name__ == "__main__":
    main()
