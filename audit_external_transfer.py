"""Replay frozen ACDC acquisitions and assess their transfer to an external cohort."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case, load_cases
from experiment import load_json, log, save_json, scores
from run_image_slice_budget import patient_roster
from run_matched_slice_budget import aggregate, calibrate, collect, partial_auc, rates
from run_sparse_annotation import paired_difference
from segment import configure, predict_many
from sparse_train import train_sparse
from summarize_annotation_evidence import OUTCOMES, patient_average


LABELS = (1, 2, 3)
PAUC_SUBSETS = {"all_absent": "absent", "near_absent": "near_absent"}


def digest(path):
    with path.open("rb") as handle:
        checksum = sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def frozen_protocol(args):
    settings = load_json(args.registration)
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    if len(cache) != 100:
        raise ValueError("Provide the 100-case official ACDC training cache")
    train_ids, cal_ids, _ = patient_roster(cache, settings, args.fold)
    plans = args.original / "frozen_acquisitions.json"
    official_plans = args.official_original / plans.name
    acquisition_digest = digest(plans)
    if acquisition_digest != digest(official_plans):
        raise ValueError("Development and official ACDC acquisition plans differ")
    plan = load_json(plans)
    registration_digest = digest(args.registration)
    for folder in (args.original, args.official_original):
        original = load_json(folder / "run.json")
        if (original["fold"] != args.fold or original["seed"] != settings["training_seed"]
                or original["registration_sha256"] != registration_digest
                or original["training"] != train_ids or original["calibration"] != cal_ids):
            raise ValueError(f"Original ACDC run and replay protocol differ: {folder}")
    if list(plan["warmup"]) != train_ids or set(plan["arms"]) != set(settings["methods"]):
        raise ValueError("Frozen warmup order or acquisition arms do not match registration")
    for arm, patients in plan["arms"].items():
        if set(patients) != set(train_ids):
            raise ValueError(f"Frozen acquisition roster does not match ACDC training: {arm}")
    thresholds = {arm: {int(label): value for label, value in load_json(
        args.official_original / f"{arm}.json")["thresholds"].items()} for arm in settings["methods"]}
    if any(set(values) != set(LABELS) for values in thresholds.values()):
        raise ValueError("The source thresholds must cover all ACDC labels")
    source = {"fold": args.fold, "seed": settings["training_seed"], "training": train_ids,
              "calibration": cal_ids, "registration_sha256": registration_digest,
              "frozen_acquisitions_sha256": acquisition_digest,
              "source_cache": {patient: digest(cache[patient]) for patient in train_ids + cal_ids},
              "source_thresholds": thresholds}
    return settings, cache, plan, source


def prepare_weights(args, settings, cache, plan, source, device):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "source_replay.json"
    if manifest_path.exists():
        manifest = replay_manifest(args.output, source)
    else:
        manifest = {**source, "pytorch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                    "threshold_replay": {}}
        save_json(manifest_path, manifest)
    pending = [arm for arm in settings["methods"] if not (args.output / f"{arm}.pt").is_file()]
    if not pending:
        return

    training = {patient: _load_case(cache[patient]) for patient in source["training"]}
    calibration = {patient: _load_case(cache[patient]) for patient in source["calibration"]}
    options = {"seed": source["seed"], "device": device, "batch_size": settings["batch_size"]}
    common = train_sparse(training, plan["warmup"], [], steps=settings["warmup_updates"], **options)
    log("acdc_replay_warmup", fold=args.fold)
    for arm in pending:
        branch = train_sparse(training, plan["arms"][arm], [], steps=settings["additional_updates"],
                              start=common, **options)
        replayed, support = calibrate(collect(calibration, branch, device), settings["calibration_false_positive_rate"])
        original = source["source_thresholds"][arm]
        delta = max(abs(replayed[label] - original[label]) for label in LABELS)
        manifest["threshold_replay"][arm] = {"thresholds": replayed, "support": support,
                                              "max_abs_delta_from_original": delta}
        save_json(manifest_path, manifest)
        if delta > 0.001:
            raise ValueError(f"Replayed ACDC calibration does not match the frozen source model: {arm}, delta={delta}")
        path = args.output / f"{arm}.pt"
        temporary = path.with_suffix(".pt.tmp")
        torch.save({name: tensor.cpu() for name, tensor in branch.teacher.items()}, temporary)
        temporary.replace(path)
        log("acdc_replay_arm", fold=args.fold, arm=arm, max_abs_threshold_delta=delta)


def replay_manifest(weights, source):
    manifest = load_json(weights / "source_replay.json")
    expected_thresholds = {arm: {str(label): value for label, value in cuts.items()}
                           for arm, cuts in source["source_thresholds"].items()}
    if any(manifest[key] != value for key, value in source.items() if key != "source_thresholds"):
        raise ValueError("Weights directory belongs to a different ACDC replay")
    if manifest["source_thresholds"] != expected_thresholds:
        raise ValueError("Weights directory uses different ACDC thresholds")
    return manifest


def external_records(cases, state, device, labels, positive_only, phases=(0, 1)):
    patients = {}
    for (patient, case), probabilities in zip(cases.items(), predict_many(state, cases.values(), device), strict=True):
        flat = probabilities[:, labels].reshape(len(probabilities), len(labels), -1)
        top = max(1, round(flat.shape[-1] / 100))
        evidence = np.partition(flat, -top, axis=-1)[..., -top:].mean(axis=-1)
        predicted = probabilities.argmax(axis=1)
        scored_truth = case.masks.copy()
        records = {}
        scored_phases = np.isin(case.phase, phases)
        for column, label in enumerate(labels):
            area = np.count_nonzero(case.masks == label, axis=(1, 2))
            small = np.zeros(len(area), dtype=bool)
            neighboring = np.zeros(len(area), dtype=bool)
            for phase in phases:
                in_phase = case.phase == phase
                small |= in_phase & (area > 0) & (4 * area <= area[in_phase].max())
                positions = case.slice_index[in_phase].astype(int)
                marked = case.slice_index[in_phase & small].astype(int)
                neighboring[in_phase] = np.isin(positions - 1, marked) | np.isin(positions + 1, marked)
            absent = (area == 0) & (not positive_only) & scored_phases
            values = evidence[:, column]
            records[label] = {"small": values[small], "absent": values[absent],
                              "near_absent": values[absent & neighboring]}
            if positive_only:
                # Without a contour, this class has an unknown rather than a negative slice label.
                predicted[(area == 0)[:, None, None] & (predicted == label)] = 0
        scored_truth[~np.isin(scored_truth, labels)] = 0
        predicted[~np.isin(predicted, labels)] = 0
        masks = scores(SimpleNamespace(masks=scored_truth[scored_phases], phase=case.phase[scored_phases]),
                       predicted[scored_phases])
        patients[patient] = (records, masks["global_dice"], masks["hard_dice"])
    return patients


def safe_pauc(records, absent):
    if all(any(len(group[subset]) for groups, _, _ in records.values() for group in groups.values())
           for subset in ("small", absent)):
        return partial_auc(records, absent)
    return None


def contrasts(results, methods, seed):
    draws = np.random.default_rng(seed).integers(0, len(results["targeted"]), size=(4000, len(results["targeted"])))
    return {arm: {metric: paired_difference(results["targeted"], results[arm], metric, draws) for metric in OUTCOMES}
            for arm in methods if arm != "targeted"}


def evaluate(args, settings, cache, source, device):
    reference = args.command == "reference"
    assessment = args.assessment if reference else args.external
    weights = args.weights or args.output
    cases = load_cases(assessment)
    official_ids = set(load_json(args.official_original / "run.json")["assessment"])
    if reference:
        if set(cases) != official_ids or official_ids.intersection(cache):
            raise ValueError("Reference mode requires precisely the original held-out ACDC test patients")
    elif (set(cache) | official_ids).intersection(cases):
        raise ValueError("Use an external cache with patient IDs distinct from ACDC")
    if args.weights is not None:
        if args.weights.resolve() == args.output.resolve():
            raise ValueError("Write assessment results to a separate directory from the supplied weights")
        manifest = replay_manifest(weights, source)
        if any(arm not in manifest["threshold_replay"] or not (weights / f"{arm}.pt").is_file()
               for arm in settings["methods"]):
            raise ValueError("Prepare and verify each ACDC weight before reference evaluation")
        args.output.mkdir(parents=True, exist_ok=True)
    source_shape = _load_case(cache[source["training"][0]]).images.shape[1:]
    if next(iter(cases.values())).images.shape[1:] != source_shape:
        raise ValueError(f"External cached images must use the ACDC spatial shape {source_shape}")
    cohort = {"name": args.cohort_name or assessment.name, "source_url": args.source_url,
              "assessment_kind": "ACDC held-out positive-contour reference" if reference else "external transfer",
              "classes": sorted(set(args.classes)), "positive_only": args.positive_only,
              "source_fit_only": True, "n_patients": len(cases),
              "external_cache": {patient: digest(assessment / f"{patient}.npz") for patient in cases}}
    phases = sorted(set(args.phases))
    if phases != [0, 1]:
        cohort["phases"] = phases
    if args.centers:
        centers = load_json(args.centers)
        if set(centers) != set(cases) or any(not isinstance(value, str) or not value for value in centers.values()):
            raise ValueError("Center manifest must map each external cached patient ID to a single known center")
        cohort["centers"] = centers
    manifest_path = args.output / "external_cohort.json"
    if manifest_path.exists() and load_json(manifest_path) != cohort:
        raise ValueError("Output directory already holds a different external cohort or assessment scope")
    save_json(manifest_path, cohort)

    methods = settings["methods"]
    results = {}
    for arm in methods:
        teacher = torch.load(weights / f"{arm}.pt", map_location="cpu", weights_only=True)
        model = SimpleNamespace(teacher=teacher)
        records = external_records(cases, model, device, cohort["classes"], args.positive_only, phases)
        thresholds = source["source_thresholds"][arm]
        assessment = rates(records, thresholds)
        results[arm] = assessment
        support = {subset: {"patient_slice_classes": sum(row[f"count_{subset}"] for row in assessment.values()),
                            "patients": sum(row[f"count_{subset}"] > 0 for row in assessment.values())}
                   for subset in ("small", "absent", "near_absent")}
        save_json(args.output / f"{arm}.json", {"assessment": assessment, "thresholds": thresholds,
                                               "assessment_pauc": {name: safe_pauc(records, subset)
                                                                   for name, subset in PAUC_SUBSETS.items()},
                                               "support": support})
        log("acdc_reference_arm" if reference else "external_transfer_arm",
            fold=args.fold, arm=arm, metrics=aggregate(assessment), support=support)
    save_json(args.output / "summary.json", {"fold": args.fold, "methods": {arm: aggregate(results[arm]) for arm in methods},
                                             "comparisons": contrasts(results, methods, source["seed"] + args.fold),
                                             "n_patients": len(cases), "training_and_calibration": "ACDC only"})


def summarize(folds, output, omit_case_identifiers=False):
    cohort = load_json(folds[0] / "external_cohort.json")
    summaries = [load_json(folder / "summary.json") for folder in folds]
    fold_ids = [item["fold"] for item in summaries]
    methods = list(summaries[0]["methods"])
    if len(set(fold_ids)) != len(fold_ids) or any(
            load_json(folder / "external_cohort.json") != cohort or list(summary["methods"]) != methods
            for folder, summary in zip(folds, summaries, strict=True)):
        raise ValueError("Supply distinct ACDC folds assessed on exactly the same cohort, scope, and arms")
    arm_results = {arm: [load_json(folder / f"{arm}.json") for folder in folds] for arm in methods}
    rosters = {tuple(result["assessment"]) for runs in arm_results.values() for result in runs}
    if len(rosters) != 1 or set(next(iter(rosters))) != set(cohort["external_cache"]):
        raise ValueError("Paired analysis needs identical, consistently ordered external rosters")
    patient_results = {arm: patient_average([item["assessment"] for item in runs])
                       for arm, runs in arm_results.items()}

    def describe(results):
        return {arm: {metric: {"mean": mean, "n": sum(row[metric] is not None for row in records.values())}
                      for metric, mean in aggregate(records).items() if metric in OUTCOMES}
                for arm, records in results.items()}

    per_center = {}
    if "centers" in cohort:
        for center in sorted(set(cohort["centers"].values())):
            ids = {patient for patient, site in cohort["centers"].items() if site == center}
            rows = {arm: {patient: row for patient, row in records.items() if patient in ids}
                    for arm, records in patient_results.items()}
            per_center[center] = {"n_patients": len(ids), "methods": describe(rows),
                                  "comparisons": contrasts(rows, methods, 20261119)}
    pauc_by_fold = {arm: {fold: result["assessment_pauc"] for fold, result in zip(fold_ids, runs, strict=True)}
                    for arm, runs in arm_results.items()}
    support_by_fold = {fold: {arm: arm_results[arm][index]["support"] for arm in methods}
                       for index, fold in enumerate(fold_ids)}
    if omit_case_identifiers:
        cohort = {key: value for key, value in cohort.items() if key not in {"external_cache", "centers"}}
        per_center = {center: {"n_patients": result["n_patients"]} for center, result in per_center.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, {"folds": fold_ids, "cohort": cohort, "methods": describe(patient_results),
                       "comparisons": contrasts(patient_results, methods, 20261119),
                       "assessment_pauc_by_fold": pauc_by_fold, "support_by_fold": support_by_fold,
                       "per_center": per_center,
                       "pooling": "Average fold outcomes per patient, then bootstrap paired patients"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "reference"):
        task = commands.add_parser(command)
        for path in ("data", "registration", "original", "official-original", "output"):
            task.add_argument(f"--{path}", type=Path, required=True)
        task.add_argument("--fold", type=int, choices=(0, 1), required=True)
        if command in ("run", "reference"):
            task.add_argument("--phases", type=int, choices=(0, 1), nargs="+", default=(0, 1))
            task.add_argument("--weights", type=Path, required=command == "reference")
        if command == "run":
            task.add_argument("--external", type=Path, required=True)
            task.add_argument("--classes", type=int, choices=LABELS, nargs="+", default=LABELS)
            task.add_argument("--positive-only", action="store_true")
            task.add_argument("--cohort-name")
            task.add_argument("--source-url")
            task.add_argument("--centers", type=Path)
        if command == "reference":
            task.add_argument("--assessment", type=Path, required=True)
            task.add_argument("--classes", type=int, choices=LABELS, nargs="+", default=(3,))
            task.set_defaults(positive_only=True, centers=None, cohort_name="ACDC official test",
                              source_url="https://www.creatis.insa-lyon.fr/Challenge/acdc/")
    task = commands.add_parser("aggregate")
    task.add_argument("--folds", type=Path, nargs="+", required=True)
    task.add_argument("--output", type=Path, required=True)
    task.add_argument("--omit-case-identifiers", action="store_true",
                      help="Publish an aggregate without cached patient identifiers or their site assignments")
    args = parser.parse_args()
    if args.command == "aggregate":
        summarize(args.folds, args.output, args.omit_case_identifiers)
        return
    settings, cache, plan, source = frozen_protocol(args)
    device = configure("cuda")
    if args.command == "prepare" or (args.command == "run" and args.weights is None):
        prepare_weights(args, settings, cache, plan, source, device)
    if args.command in ("run", "reference"):
        evaluate(args, settings, cache, source, device)


if __name__ == "__main__":
    main()
