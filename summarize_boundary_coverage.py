"""Summarize locked boundary acquisition fits without reopening source images or masks."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from audit_decision_curves import interval, matched_recall, patient_histograms
from audit_stratified_decisions import partial_auc
from experiment import load_json, save_json


SUBSETS = ("small", "absent", "near_absent")
OUTCOMES = ("small_dice", "global_dice", "recall", "fpr", "near_fpr")
COUNTS = (*SUBSETS, "adjacent_positive")
CLASS_NAMES = {"1": "RV", "2": "MYO", "3": "LV"}
NEGATIVE_GROUPS = SUBSETS[1:]
PRIMARY_FPR = 0.20


def bootstrap_weights(size, draws, rng):
    return np.vstack([np.ones(size, dtype=np.int64), rng.multinomial(size, np.full(size, 1 / size), size=draws)])


def describe(values):
    return {"estimate": float(values[0]) if np.isfinite(values[0]) else None,
            "ci95": interval(values[1:]), "valid_draws": int(np.isfinite(values[1:]).sum())}


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def contrast(methods, contender):
    return {"methods": {arm: {key: describe(values) for key, values in items.items()} for arm, items in methods.items()},
            f"{contender}_minus": {arm: {key: describe(methods[contender][key] - values) for key, values in items.items()}
                                   for arm, items in methods.items() if arm != contender} if contender in methods else {}}


def average_seeds(seed_metrics):
    first = next(iter(seed_metrics.values()))
    return {arm: {metric: np.mean([metrics[arm][metric] for metrics in seed_metrics.values()], axis=0)
                  for metric in row} for arm, row in first.items()}


def check_scores(assessment, expected_ids):
    if (set(assessment["outcomes"]) != set(expected_ids) or set(assessment["scores"]) != set(expected_ids)):
        raise ValueError("Prediction and outcomes must cover the expected paired patients")
    for patient, row in assessment["scores"].items():
        status = assessment["outcomes"][patient]
        for subset, measure in zip(SUBSETS, ("recall", "fpr", "near_fpr"), strict=True):
            scores = np.asarray(row[subset])
            if len(scores) != status[f"count_{subset}"] or not np.isfinite(scores).all():
                raise ValueError("Matched-score support differs from source operating-point output")
            outcome = status[measure]
            if (outcome is None) != (len(scores) == 0) or (len(scores) and np.mean(scores > 0) != outcome):
                raise ValueError("Archived source operating point did not replay at zero margin")


def read_cohort(root, development, design, digest, cohort, allow_incomplete):
    output = {}
    for seed in design["seeds"]:
        dirs = [root / f"seed-{seed}-fold-{fold}" for fold in design["folds"]]
        required = ["run.json", *(f"{arm}.json" for arm in design["arms"])]
        if cohort == "development":
            required += ["frozen_acquisitions.json", "nomination_audit.json"]
            if "max_replacements" in design:
                required.append("replacement_support.json")
        if any(not (path / file).is_file() for path in dirs for file in required):
            if allow_incomplete:
                continue
            raise ValueError(f"Missing at least one required output for {cohort} seed {seed}")
        fits = []
        for fold, path in zip(design["folds"], dirs, strict=True):
            run = load_json(path / "run.json")
            training, calibration, testing = (set(run[key]) for key in ("training", "calibration", "development"))
            if (run["fold"] != fold or run["seed"] != seed or run["design_sha256"] != digest
                    or tuple(map(len, (training, calibration, testing))) != (40, 12, 20)
                    or len(training | calibration | testing) != 72):
                raise ValueError("Patient roles or fit identity differ from the locked design")
            if "detection_calibration" in run:
                detector = set(run["detection_calibration"])
                reliability = set(run["reliability_fit"])
                if len(detector) != 6 or len(reliability) != 6 or detector | reliability != calibration:
                    raise ValueError("Expected six disjoint reliability-fit patients and six detector-calibration patients")
            arms = {arm: load_json(path / f"{arm}.json") for arm in design["arms"]}
            roster = sorted(arms[design["arms"][0]]["scores"])
            if cohort == "development":
                if set(roster) != testing:
                    raise ValueError("Development patients differ from nominated fold roles")
            elif len(roster) != 50 or set(roster) & (training | calibration | testing):
                raise ValueError("Expected the disjoint 50-patient official assessment")
            for arm in arms.values():
                check_scores(arm, roster)
            counts = {patient: tuple(arms[design["arms"][0]]["outcomes"][patient][f"count_{subset}"] for subset in SUBSETS)
                      for patient in roster}
            if any(any(counts[p] != tuple(rows["outcomes"][p][f"count_{subset}"] for subset in SUBSETS) for p in roster)
                   for rows in arms.values()):
                raise ValueError("Reference slice-class support must be matched across policies")
            if cohort == "official" and run != load_json(development / path.name / "run.json"):
                raise ValueError("Official evaluation does not identify the same frozen development fits")
            fits.append({"run": run, "roster": roster, "counts": counts, "arms": arms, "path": path})
        output[seed] = fits
    if output:
        first = next(iter(output.values()))
        for fits in output.values():
            for left, right in zip(first, fits, strict=True):
                if (left["roster"] != right["roster"] or left["counts"] != right["counts"]
                        or any(left["run"][k] != right["run"][k] for k in ("training", "calibration", "development",
                                                                        "registration_sha256", "code_sha256", "selection_sha256"))):
                    raise ValueError("Source code, all patient roles and evaluation support must match across seeds")
        overlap = set(first[0]["roster"]) & set(first[1]["roster"])
        if (cohort == "development" and overlap) or (cohort == "official" and len(overlap) != 50):
            raise ValueError("Development folds must be disjoint and official folds must share a roster")
        if cohort == "official" and first[0]["counts"] != first[1]["counts"]:
            raise ValueError("The two official evaluations must share reference support")
    return output


def assessment_metrics(fits, arms, weights):
    results = {}
    pauc = {}
    for arm in arms:
        metrics = {}
        by_fold = []
        for fit, counts in zip(fits, weights, strict=True):
            rows = fit["arms"][arm]["scores"]
            groups = {subset: [rows[patient][subset] for patient in fit["roster"]] for subset in SUBSETS}
            answers = fit["arms"][arm]["outcomes"]
            values = {metric: np.asarray([answers[p][metric] for p in fit["roster"]], dtype=float) for metric in OUTCOMES}
            sums = {metric: (counts @ np.nan_to_num(result), counts @ np.isfinite(result)) for metric, result in values.items()}
            matched = {subset: matched_recall(patient_histograms(groups["small"], groups[subset]), counts, (PRIMARY_FPR,))[:, 0]
                       if any(groups["small"]) and any(groups[subset]) else np.full(len(counts), np.nan)
                       for subset in NEGATIVE_GROUPS}
            by_fold.append((sums, matched, {subset: partial_auc(groups["small"], groups[subset]) for subset in NEGATIVE_GROUPS}))
        for metric in OUTCOMES:
            numerator = by_fold[0][0][metric][0] + by_fold[1][0][metric][0]
            denominator = by_fold[0][0][metric][1] + by_fold[1][0][metric][1]
            metrics[metric] = ratio(numerator, denominator)
        metrics.update({f"matched_{subset}": (by_fold[0][1][subset] + by_fold[1][1][subset]) / 2
                        for subset in NEGATIVE_GROUPS})
        results[arm] = metrics
        pauc[arm] = {subset: float(np.mean([item[2][subset] for item in by_fold]))
                     if all(item[2][subset] is not None for item in by_fold) else None for subset in NEGATIVE_GROUPS}
    return results, pauc


def assess(seeds, arms, contender, cohort, draws, rng):
    first = next(iter(seeds.values()))
    weights = [bootstrap_weights(len(first[0]["roster"]), draws, rng)]
    weights.append(weights[0] if cohort == "official" else bootstrap_weights(len(first[1]["roster"]), draws, rng))
    raw = {str(seed): assessment_metrics(fits, arms, weights) for seed, fits in seeds.items()}
    joint = average_seeds({seed: row[0] for seed, row in raw.items()})
    support = []
    for fit in first:
        support.append({subset: {"patients": sum(row[j] > 0 for row in fit["counts"].values()),
                                 "slice_class_pairs": sum(row[j] for row in fit["counts"].values())}
                        for j, subset in enumerate(SUBSETS)})
    pauc = {arm: {subset: float(np.mean([row[1][arm][subset] for row in raw.values()]))
                  if all(row[1][arm][subset] is not None for row in raw.values()) else None for subset in NEGATIVE_GROUPS}
            for arm in arms}
    return {"seeds": list(seeds), "patients": 50 if cohort == "official" else 40, "test_support_by_fold": support,
            "fixed_seed_joint": {**contrast(joint, contender), "mean_centered_standardized_partial_auc": pauc},
            "by_seed": {seed: {**contrast(row[0], contender), "mean_centered_standardized_partial_auc": row[1]}
                        for seed, row in raw.items()}}


def validate_archives(seeds, archive_root, cohort, design, contender):
    findings = {}
    for seed, fits in seeds.items():
        for fit in fits:
            fold, path = fit["run"]["fold"], fit["path"]
            comparisons = {}
            if "max_replacements" in design:
                parent = archive_root / "calibrated-boundary" / cohort / path.name
                for arm, result in fit["arms"].items():
                    if arm == contender:
                        continue
                    source = parent / f"{arm}.json"
                    if source.is_file():
                        if load_json(source) != result:
                            raise ValueError(f"A reused comparator differs from its calibrated parent run: {cohort}, {seed}, fold {fold}, {arm}")
                        findings[f"{seed}/fold{fold}/{arm}"] = {"exact_calibrated_parent_reuse": True}
            if seed == 2027 and cohort == "development":
                comparisons["targeted"] = archive_root / "sparse-image-cloud" / f"fold-{fold}" / "targeted.json"
            elif seed == 2027:
                comparisons["targeted"] = archive_root / "sparse-official-cloud" / f"image-{fold}" / "targeted.json"
                comparisons["pooled_image_farthest"] = archive_root / "acquisition-baselines" / f"fold-{fold}" / "pooled_image_farthest.json"
            elif cohort == "official":
                base = archive_root / "reviewer-seed-stability-thread-pinned" / path.name
                comparisons = {arm: base / f"{arm}.json" for arm in ("targeted", "pooled_image_farthest")}
            for arm, reference in comparisons.items():
                if arm not in fit["arms"] or not reference.is_file():
                    continue
                archived = load_json(reference)
                old = archived["assessment"] if "assessment" in archived else archived["development"]
                new = fit["arms"][arm]["outcomes"]
                if old.keys() != new.keys() or any((old[p][key] is None) != (new[p][key] is None) for p in old for key in old[p]):
                    raise ValueError("Archived model evaluation support does not match")
                new_cutoff_roster = "detection_calibration" in fit["run"]
                fields = [key for key in next(iter(old.values())) if not new_cutoff_roster
                          or key in ("small_dice", "global_dice") or key.startswith("count_")]
                delta = max(abs(old[p][key] - new[p][key]) for p in old for key in fields if old[p][key] is not None)
                result = {"max_absolute_outcome_difference": delta, "matches_within_1e-6": delta <= 1e-6,
                          "comparison_scope": "cutoff-independent scores and support" if new_cutoff_roster else "all archived outcomes"}
                if delta > 1e-6:
                    raise ValueError(f"An archived comparison does not replay: {cohort}, {seed}, fold {fold}, {arm}: {delta}")
                if not new_cutoff_roster and "cutoffs" in fit["arms"][arm] and "thresholds" in archived:
                    cutoff_delta = max(abs(v - archived["thresholds"][label]) for label, v in fit["arms"][arm]["cutoffs"].items())
                    if cutoff_delta > 1e-6:
                        raise ValueError(f"Archived source cutoffs do not match: {seed}, fold {fold}, {arm}")
                    result["max_absolute_cutoff_difference"] = cutoff_delta
                findings.setdefault(f"{seed}/fold{fold}/{arm}", {}).update(result)
            if cohort == "development":
                plan = load_json(path / "frozen_acquisitions.json")["arms"]
                baseline = archive_root / "acquisition-baselines" / f"fold-{fold}" / "audit_plan.json"
                legacy = (archive_root / "sparse-image-cloud" / f"fold-{fold}" if seed == 2027 else
                          archive_root / "acquisition-seed-2028" / f"fold-{fold}" if seed == 2028 else
                          archive_root / "reviewer-seed-stability-thread-pinned" / path.name) / "frozen_acquisitions.json"
                for arm, reference, key in (("pooled_image_farthest", baseline, "choices"), ("targeted", legacy, "arms")):
                    if arm in plan and reference.is_file():
                        if plan[arm] != load_json(reference)[key][arm]:
                            raise ValueError(f"The original acquisition list does not replay: {seed}, fold {fold}, {arm}")
                        findings.setdefault(f"{seed}/fold{fold}/{arm}", {})["exact_acquisition_roster_match"] = True
    return findings


def replacement_audit(fit, plan, contender, maximum):
    declared = load_json(fit["path"] / "replacement_support.json")
    replacements = plan["replacements"]
    if (declared != {"maximum": maximum, "replaced": len(replacements), "queries": 80}
            or len(replacements) > maximum):
        raise ValueError("Bounded acquisition differs from its declared replacement count")
    slots = {(item["patient"], item["phase"]) for item in replacements}
    if len(slots) != len(replacements) or any(phase not in (0, 1) for _, phase in slots):
        raise ValueError("A bounded fit cannot change one patient-phase slot more than once")
    baseline = plan["arms"]["pooled_image_farthest"]
    new = plan["arms"][contender]
    observed = ({(p, item) for p in baseline for item in set(baseline[p]) - set(new[p])},
                {(p, item) for p in new for item in set(new[p]) - set(baseline[p])})
    recorded = tuple({(item["patient"], item[key]) for item in replacements} for key in ("old", "new"))
    if recorded != observed or any(len(rows) != len(replacements) for rows in observed):
        raise ValueError("Replacement ledger must reconcile exactly with the old and new mask rosters")
    gains = [item["gain"] for item in replacements]
    if any(not np.isfinite(gain) or gain <= 0 for gain in gains):
        raise ValueError("Every logged replacement must increase the registered objective")
    return {**declared, "retained_image_difference_queries": 80 - len(replacements),
            "patients_changed": len({pair[0] for pair in slots}),
            "by_phase": {phase: sum(item["phase"] == code for item in replacements)
                         for code, phase in enumerate(("ED", "ES"))},
            "objective_gain_min": min(gains, default=None), "objective_gain_max": max(gains, default=None)}


def acquisition_seed(fits, design, contender, weights):
    totals = {}
    audit_rows = []
    overlap = {}
    replacements = []
    for fit in fits:
        training = sorted(fit["run"]["training"])
        plan = load_json(fit["path"] / "frozen_acquisitions.json")
        for arm, selections in plan["arms"].items():
            if (set(selections) != set(training) or set(plan["initial"]) != set(training)
                    or any(len(set(plan["initial"][p])) != 2 or len(set(selections[p])) != 4
                           or not set(plan["initial"][p]) <= set(selections[p]) for p in training)):
                raise ValueError("Expected 80 new masks and 80 initial masks for each fold arm")
            if arm == contender:
                continue
            contender_plan = plan["arms"][contender]
            sets = [(set(selections[p]) - set(plan["initial"][p]),
                     set(contender_plan[p]) - set(plan["initial"][p])) for p in training]
            overlap.setdefault(arm, []).append({"patients_with_same_two_new_masks": sum(left == right for left, right in sets),
                                               "shared_new_masks": sum(len(left & right) for left, right in sets)})
        audit_rows.append(load_json(fit["path"] / "nomination_audit.json"))
        if "max_replacements" in design:
            replacements.append(replacement_audit(fit, plan, contender, design["max_replacements"]))
    for arm in design["arms"]:
        by_fold = []
        for fit, counts in zip(fits, weights, strict=True):
            examples = fit["arms"][arm]["training_selection"]
            patients = sorted(fit["run"]["training"])
            if set(examples) != set(patients):
                raise ValueError("Selected sample reference audit must cover every training patient")
            values = np.asarray([[[examples[p][label][stat] for stat in COUNTS] for label in CLASS_NAMES]
                                 for p in patients], dtype=int)
            if (values < 0).any() or (values > 2).any() or (values[:, :, 0] + values[:, :, 1] > 2).any():
                raise ValueError("Patient-level acquired support is invalid")
            by_fold.append(np.einsum("dp,pck->dck", counts, values))
        totals[arm] = by_fold
    acquisition = {arm: {f"{scope}/{stat}": (fits[0][:, j, i] + fits[1][:, j, i]) / 2
                         for j, scope in enumerate(CLASS_NAMES.values())
                         for i, stat in enumerate(COUNTS)} for arm, fits in totals.items()}
    for arm, by_fold in totals.items():
        acquisition[arm].update({f"all_classes/{name}": (by_fold[0][:, :, k] + by_fold[1][:, :, k]).sum(axis=1) / 2
                                 for k, name in enumerate(COUNTS)})
    raw_counts = {arm: [{scope: {name: int(fold[0, j, k]) for k, name in enumerate(COUNTS)}
                        for j, scope in enumerate(CLASS_NAMES.values())} for fold in fits] for arm, fits in totals.items()}
    top_precision = {}
    top_raw = {}
    nomination_arms = tuple(arm for arm in design["arms"] if all(arm in record for record in audit_rows))
    if any(set(record) != set(nomination_arms) for record in audit_rows):
        raise ValueError("Both folds must audit the same registered nomination policies")
    for arm in nomination_arms:
        numerator = {stat: [] for stat in COUNTS}
        denominator = []
        top_raw[arm] = []
        for fit, rows, counts in zip(fits, audit_rows, weights, strict=True):
            patients = sorted(fit["run"]["training"])
            records = rows[arm]["nominations"]
            counts_by_patient = Counter(item["patient"] for item in records)
            if (len(records) != min(rows[arm]["available"], design["boundary_precision_k"])
                    or not set(counts_by_patient) <= set(patients)):
                raise ValueError("Top nomination list differs from its registered support cap")
            raw = {"available": rows[arm]["available"], "audited": len(records), "patients": len(counts_by_patient),
                   "unique_physical_slices": len({(r["patient"], r["index"]) for r in records}),
                   "true_counts": {stat: sum(r[stat] for r in records) for stat in COUNTS}}
            raw["by_class"] = {}
            for label, name in CLASS_NAMES.items():
                subset = [row for row in records if str(row["label"]) == label]
                raw["by_class"][name] = {"nomination_pairs": len(subset), "patients": len({r["patient"] for r in subset}),
                                         "true_counts": {stat: sum(r[stat] for r in subset) for stat in COUNTS}}
            top_raw[arm].append(raw)
            denominator.append(counts @ np.asarray([counts_by_patient[p] for p in patients]))
            for stat in COUNTS:
                tally = Counter(item["patient"] for item in records if item[stat])
                numerator[stat].append(counts @ np.asarray([tally[p] for p in patients]))
        top_precision[arm] = {stat: ratio(parts[0] + parts[1], denominator[0] + denominator[1])
                              for stat, parts in numerator.items()}
    return acquisition, top_precision, raw_counts, top_raw, overlap, replacements


def summarize_acquisition(seeds, design, contender, draws, rng):
    weights = [bootstrap_weights(40, draws, rng) for _ in design["folds"]]
    raw = {str(seed): acquisition_seed(fits, design, contender, weights) for seed, fits in seeds.items()}
    top = {seed: row[1] for seed, row in raw.items()}
    if any(tuple(item) != tuple(next(iter(top.values()))) for item in top.values()):
        raise ValueError("Expected the same nomination diagnostic across all registered seeds")
    joint_precision = contrast(average_seeds(top), contender) if next(iter(top.values())) else None
    result = {"new_masks_per_fit": 80, "training_masks_per_fit": 160,
            "units": "Reference slice-class states among 80 new masks per fit; a physical mask can contribute several class states. Equal weight per fold and seed.",
            "top_k_nomination_precision_definition": "Nomination-level fraction among fixed top K per fitted fold; patient-cluster bootstrap with common draws across arms and seeds.",
            "fixed_seed_joint": {"selection": contrast(average_seeds({seed: row[0] for seed, row in raw.items()}), contender),
                                "top_nomination_precision": joint_precision},
            "by_seed": {seed: {"selection": contrast(row[0], contender),
                                "top_nomination_precision": contrast(row[1], contender) if row[1] else None,
                                "raw_counts_by_fold": row[2], "top_nomination_support_by_fold": row[3] or None,
                                "acquisition_overlap_by_fold": row[4]}
                        for seed, row in raw.items()}}
    if "max_replacements" in design:
        result["replacement_support_by_seed_and_fold"] = {seed: row[5] for seed, row in raw.items()}
        result["replacement_support_total"] = {key: sum(fit[key] for row in raw.values() for fit in row[5])
                                               for key in ("queries", "replaced", "retained_image_difference_queries")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--cohorts", nargs="+", choices=("development", "official"), default=("development", "official"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("Use at least one bootstrap draw")
    design = load_json(args.design)
    contender = design.get("contender", design.get("primary_arm", design["arms"][0]))
    if contender not in design["arms"]:
        raise ValueError("Registered contender must be one of the study arms")
    digest = sha256(args.design.read_bytes()).hexdigest()
    rng = np.random.default_rng(20261117)
    report = {"design_sha256": digest, "bootstrap_draws": args.draws, "matched_fpr": PRIMARY_FPR,
              "estimand": "Fixed-fit patient bootstrap, shared over policies and seeds. The disjoint development rosters are resampled separately, and official patients share one resample across both fits. ROC thresholds are fitted retrospectively within each source fit and bootstrap draw and recall is averaged across its two folds. Standardized pAUC uses centered scores up to FPR 0.4, with equal patient weight within each outcome.",
              "primary": design["primary"], "contender": contender, "cohorts": {}}
    for cohort in args.cohorts:
        fits = read_cohort(args.root / cohort, args.root / "development", design, digest, cohort, args.allow_incomplete)
        if not fits:
            continue
        result = assess(fits, design["arms"], contender, cohort, args.draws, rng)
        if "reference_masks" in next(iter(fits.values()))[0]["run"]:
            result["declared_label_access_by_seed_and_fold"] = {str(seed): [fit["run"]["reference_masks"] for fit in rows]
                                                                 for seed, rows in fits.items()}
        if args.archive_root:
            result["archived_baseline_checks"] = validate_archives(fits, args.archive_root, cohort, design, contender)
        if cohort == "development":
            result["acquisition"] = summarize_acquisition(fits, design, contender, args.draws, rng)
            result["detector_calibration_by_seed_and_fold"] = {
                str(seed): [{arm: {"cutoffs": values["cutoffs"], "negative_patient_support": values["calibration_support"]}
                             for arm, values in fit["arms"].items()} for fit in rows] for seed, rows in fits.items()
            }
        report["cohorts"][cohort] = result
    if not report["cohorts"]:
        raise ValueError("No complete cohort/seed pairs were available")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)


if __name__ == "__main__":
    main()
