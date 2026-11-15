"""Replay paired input-geometry interactions under declared class pooling rules."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from audit_decision_curves import matched_recall, patient_histograms
from audit_external_transfer import digest
from experiment import load_json, save_json
from summarize_boundary_coverage import bootstrap_weights, describe


GROUPS = ("small", "absent", "near_absent")
CLASSES = ("RV", "MYO", "LV")
VENTRICLES = ("RV", "LV")
MODES = ("pooled_centered", "ventricular_pool_centered", "ventricular_pool_raw",
         "equal_ventricular", *VENTRICLES)


def read(directory, protocol):
    fits = {}
    for path in sorted(directory.glob("seed-*-fold-*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        key = (data["seed"], data["fold"])
        if key in fits or data["design_sha256"] != digest(protocol):
            raise ValueError("Unrecognized experimental input")
        fits[key] = data
    return fits


def score_groups(arm, labels, center):
    return {group: [[value - (arm["thresholds"][label] if center else 0.0)
                     for label in labels for value in patient[label][group]]
                    for patient in arm["target"]]
            for group in GROUPS}


def curves(arm, weights):
    modes = {
        "pooled_centered": (CLASSES, True),
        "ventricular_pool_centered": (VENTRICLES, True),
        "ventricular_pool_raw": (VENTRICLES, False),
        "RV": (("RV",), False),
        "LV": (("LV",), False),
    }
    result = {}
    for mode, (labels, center) in modes.items():
        groups = score_groups(arm, labels, center)
        result[mode] = {
            key: matched_recall(patient_histograms(groups["small"], groups[key]),
                                weights, (0.20,))[:, 0]
            for key in GROUPS[1:]
        }
    result["equal_ventricular"] = {
        key: (result["RV"][key] + result["LV"][key]) / 2 for key in GROUPS[1:]
    }
    return result


def align(a, b):
    if any(a[key] != b[key] for key in ("thresholds", "support", "source", "weight_sha256")):
        raise ValueError("Weight, cutoff or calibration source mismatch across geometries")
    a_counts = [[len(patient[label][key]) for label in CLASSES for key in GROUPS] for patient in a["target"]]
    b_counts = [[len(patient[label][key]) for label in CLASSES for key in GROUPS] for patient in b["target"]]
    if a_counts != b_counts:
        raise ValueError("Patient-aligned reference support changed across geometries")


def average(values):
    return {mode: {key: np.mean([item[mode][key] for item in values], axis=0) for key in GROUPS[1:]}
            for mode in MODES}


def summarize(native, physical):
    return {mode: {group: {
        "native": describe(native[mode][group]),
        "physical": describe(physical[mode][group]),
        "interaction": describe(physical[mode][group] - native[mode][group]),
    } for group in GROUPS[1:]} for mode in MODES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("native", "physical", "native-protocol", "physical-protocol", "design", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    plan = load_json(args.design)
    native = read(args.native, args.native_protocol)
    fov = read(args.physical, args.physical_protocol)
    keys = list(map(tuple, plan["seed_folds"]))
    if set(keys) != native.keys() or set(keys) != fov.keys():
        raise ValueError("Require all same three fixed seeds and both folds")
    baseline, method = plan["arms"]
    n = native[keys[0]]["patients"]
    weights = bootstrap_weights(n, 4000, np.random.default_rng(20261118))
    contrasts = {"native": {}, "physical": {}}
    for key in keys:
        original, physical = native[key], fov[key]
        if original["patients"] != n or physical["patients"] != n:
            raise ValueError("Target cohort size changed")
        for arm in (baseline, method):
            align(original["arms"][arm], physical["arms"][arm])
        for geometry, data in (("native", original), ("physical", physical)):
            first = curves(data["arms"][baseline], weights)
            second = curves(data["arms"][method], weights)
            contrasts[geometry][key] = {
                mode: {group: second[mode][group] - first[mode][group] for group in GROUPS[1:]}
                for mode in MODES
            }
    all_native = average(list(contrasts["native"].values()))
    all_fov = average(list(contrasts["physical"].values()))
    centered, raw = "ventricular_pool_centered", "ventricular_pool_raw"
    centering = {group: describe(
        (all_fov[centered][group] - all_native[centered][group])
        - (all_fov[raw][group] - all_native[raw][group])
    ) for group in GROUPS[1:]}
    by_seed = {str(seed): summarize(
        average([contrasts["native"][key] for key in keys if key[0] == seed]),
        average([contrasts["physical"][key] for key in keys if key[0] == seed]),
    ) for seed in sorted({k[0] for k in keys})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"definition": plan["interaction"], "patients": n,
                           "design_sha256": digest(args.design), "joint": summarize(all_native, all_fov),
                           "centering_interaction_difference": centering, "by_seed": by_seed})


if __name__ == "__main__":
    main()
