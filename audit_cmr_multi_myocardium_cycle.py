"""Replay frozen myocardial detection on the exploratory cine-cycle-supported negatives."""

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from audit_boundary_detector import CLASSES, macro_decision, sample_cutoffs
from audit_external_transfer import digest
from experiment import load_json, log, save_json
from segment import configure, predict_many
from summarize_boundary_coverage import bootstrap_weights, describe


MYO = CLASSES.index("MYO")
GROUPS = ("all", "adjacent")
BATCH_SIZE = 16


def load_ledger(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def target_batches(directory):
    coverage_path = directory / "coverage.json"
    cycle_path = directory / "cycle-eligibility.json"
    coverage, cycle = load_json(coverage_path), load_json(cycle_path)
    case_hash = digest(coverage_path)
    paths = sorted(directory.glob("case-*.npz"))
    if (case_hash != cycle["coverage_sha256"] or len(paths) != coverage["patients"]
            or set(coverage["cases"]) != {p.stem for p in paths}
            or set(cycle["cycle_supported_blank"]) != set(coverage["cases"])):
        raise ValueError("Cycle and image manifests describe different cohorts")
    batches, mappings = [], []
    support = {group: np.zeros(len(paths), dtype=int) for group in GROUPS}
    for patient, path in enumerate(paths):
        flags = np.asarray(cycle["cycle_supported_blank"][path.stem], dtype=bool)[:, MYO]
        if not flags.any():
            continue
        if digest(path) != coverage["cases"][path.stem]["cache_sha256"]:
            raise ValueError(f"Changed target cache for {path.stem}")
        with np.load(path, allow_pickle=False) as record:
            images = record["images"]
            phase, native = record["phase"], record["slice_index"]
            areas, primary = record["native_areas"], record["verified_absent"]
        if len(flags) != len(images) or np.any(flags & ((areas > 0).any(axis=1) | primary[:, MYO])):
            raise ValueError("Cycle flags must add only currently all-blank native planes")
        close = np.zeros_like(flags)
        for k in (0, 1):
            group = phase == k
            area = areas[group, MYO]
            if np.any(flags[group]) and area.max() <= 0:
                raise ValueError("Target lacks positive reference in the selected phase")
            small = (area > 0) & (4 * area <= area.max())
            index = native[group]
            close[group] = np.isin(index - 1, index[small]) | np.isin(index + 1, index[small])
        for start in np.unique((np.flatnonzero(flags) // BATCH_SIZE) * BATCH_SIZE):
            slc = slice(start, start + BATCH_SIZE)
            batches.append(SimpleNamespace(images=images[slc]))
            mappings.append((patient, flags[slc], flags[slc] & close[slc]))
        support["all"][patient] = int(flags.sum())
        support["adjacent"][patient] = int((flags & close).sum())
    expected = {name: {"pairs": int(values.sum()), "patients": int((values > 0).sum())}
                for name, values in support.items()}
    return batches, mappings, support, expected, case_hash, digest(cycle_path)


def evaluate(batches, mapping, count, model, device):
    rows = [{name: [] for name in GROUPS} for _ in range(count)]
    for (patient, flags, near), prob in zip(
            mapping, predict_many(model, batches, device, batch_size=BATCH_SIZE), strict=True):
        if prob.shape[0] != len(flags):
            raise ValueError("Myocardial score batch support changed")
        pixels = prob[:, MYO + 1].reshape(len(prob), -1)
        top = max(1, round(pixels.shape[-1] / 100))
        scores = np.partition(pixels, -top, axis=-1)[..., -top:].mean(axis=-1)
        for name, eligible in zip(GROUPS, (flags, near), strict=True):
            rows[patient][name].extend(map(float, scores[eligible]))
    return [{name: sorted(values) for name, values in row.items()} for row in rows]


def freeze_arm(archive, fit, arm):
    data = archive["arms"][arm]
    json_path, weights = fit / f"{arm}.json", fit / f"{arm}.pt"
    meta = load_json(json_path)
    sha = digest(weights)
    threshold = {c: meta["cutoffs"][str(i)] for i, c in enumerate(CLASSES, 1)}
    support = {c: meta["calibration_support"][str(i)] for i, c in enumerate(CLASSES, 1)}
    expected = {"thresholds": threshold, "support": support, "weight_sha256": sha}
    if sha != meta["weight_sha256"] or any(data[k] != v for k, v in expected.items()):
        raise ValueError(f"Existing source calibration ledger differs from the archived fit for {arm}")
    if len(data["source"]) != 6:
        raise ValueError("Expected exactly six source calibration donors")
    return weights, {**expected, "source": data["source"]}


def score(args):
    design = load_json(args.design)
    fit_id = f"seed-{args.seed}-fold-{args.fold}"
    dirs = {"native": args.native, "fov": args.fov}
    archives = {"native": args.native_archive, "fov": args.fov_archive}
    model_folders = {"primary": args.bounded / fit_id, "swap": args.swap / fit_id}
    sources = {name: load_ledger(archives[name] / f"{fit_id}.json.gz") for name in design["conditions"]}
    panels = [load_json(folder / "run.json")["detection_calibration"] for folder in model_folders.values()]
    if panels[0] != panels[1] or len(panels[0]) != 6:
        raise ValueError("All models must have the same six-patient fold-specific detection panel")
    targets = {name: target_batches(dirs[name]) for name in design["conditions"]}
    for name, (_, _, _, support, checksum, _) in targets.items():
        archive = sources[name]
        if (archive["cohort_sha256"] != checksum or archive["fold"] != args.fold
                or archive["seed"] != args.seed or archive["patients"] != 113):
            raise ValueError("Mismatch between exploratory inputs and prior frozen transfer")
        if support != design["support_expected"]:
            raise ValueError("Unexpected exploratory target support")
    if any(not np.array_equal(targets["native"][2][group], targets["fov"][2][group]) for group in GROUPS):
        raise ValueError("Representations must have identical per-patient target support")
    receipt = {"seed": args.seed, "fold": args.fold, "design_sha256": digest(args.design),
               "cohorts": {k: {"coverage": item[4], "cycle": item[5]} for k, item in targets.items()},
               "patients": 113, "arms": {}}
    device = configure("cuda")
    for arm in design["arms"]:
        fit = model_folders["swap" if arm == "image_loss_matched" else "primary"]
        weights, frozen = freeze_arm(sources["native"], fit, arm)
        other = sources["fov"]["arms"][arm]
        if any(other[key] != value for key, value in frozen.items()):
            raise ValueError(f"Source evidence differs by input representation for {arm}")
        model = SimpleNamespace(teacher=torch.load(weights, map_location="cpu", weights_only=True))
        frozen["target"] = {k: evaluate(item[0], item[1], receipt["patients"], model, device)
                            for k, item in targets.items()}
        receipt["arms"][arm] = frozen
        log("myocardial_cycle_fit_scored", fold=args.fold, seed=args.seed, arm=arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    with gzip.open(temp, "wt", encoding="utf-8") as handle:
        json.dump(receipt, handle, separators=(",", ":"), allow_nan=False)
    temp.replace(args.output)


def paired_summary(metrics, arms):
    return {
        "methods": {arm: {name: describe(draws) for name, draws in metrics[arm].items()} for arm in arms},
        "bounded_minus": {arm: {name: describe(metrics["bounded"][name] - metrics[arm][name])
                                for name in metrics[arm]} for arm in arms if arm != "bounded"},
    }


def replay(args):
    design = load_json(args.design)
    fits, cohorts = {}, None
    for path in args.files:
        result = load_ledger(path)
        key = (result["seed"], result["fold"])
        if key in fits or result["design_sha256"] != digest(args.design):
            raise ValueError("Duplicate or unsupported inference ledger")
        if cohorts is None:
            cohorts = result["cohorts"]
        if result["cohorts"] != cohorts or result["patients"] != 113 or set(result["arms"]) != set(design["arms"]):
            raise ValueError("Cohort or study arms differ")
        fits[key] = result["arms"]
    expected = {(seed, fold) for seed in design["seeds"] for fold in design["folds"]}
    if set(fits) != expected:
        raise ValueError("Require every frozen seed and fold once")
    rng = np.random.default_rng(20261118)
    target_weights = bootstrap_weights(113, args.draws, rng)
    source_weights = {fold: bootstrap_weights(6, args.draws, rng) for fold in design["folds"]}
    rows, counts, cutoff_reports = {}, {}, {}
    baseline_support = None
    for (seed, fold), fit in sorted(fits.items()):
        run = f"seed-{seed}-fold-{fold}"
        rows[run], counts[run], cutoff_reports[run] = {}, {}, {}
        for arm, item in fit.items():
            sampled, info = sample_cutoffs(item, source_weights[fold])
            cutoff_reports[run][arm] = info["MYO"]
            threshold = item["thresholds"]["MYO"]
            fixed = {"MYO": np.full(len(target_weights), threshold)}
            for name in design["conditions"]:
                trial = item["target"][name]
                support = {group: [len(patient[group]) for patient in trial] for group in GROUPS}
                if baseline_support is None:
                    baseline_support = support
                if support != baseline_support:
                    raise ValueError("Comparison does not share the same patient-paired support")
                res = rows[run].setdefault(name, {}).setdefault(arm, {})
                errors = counts[run].setdefault(name, {}).setdefault(arm, {})
                for group in GROUPS:
                    arrays = {"MYO": [patient[group] for patient in trial]}
                    errors[group] = int(sum(value > threshold for a in arrays["MYO"] for value in a))
                    res[f"{group}_fpr"] = macro_decision(arrays, fixed, target_weights, ("MYO",))
                    res[f"{group}_fpr_source_and_target"] = macro_decision(
                        arrays, sampled, target_weights, ("MYO",))
    report = {"design_sha256": digest(args.design), "cohorts": cohorts, "draws": args.draws,
              "patients": 113, "support": design["support_expected"], "source_cutoffs": cutoff_reports,
              "raw_errors_by_fit": counts, "results": {}}
    for name in design["conditions"]:
        averages = {}
        for seed in (*design["seeds"], "all"):
            runs = [r for r in rows if seed == "all" or r.startswith(f"seed-{seed}-")]
            methods = {arm: {group: np.mean([rows[r][name][arm][group] for r in runs], axis=0)
                             for group in rows[runs[0]][name][arm]} for arm in design["arms"]}
            averages[str(seed)] = paired_summary(methods, design["arms"])
        report["results"][name] = averages
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    gather = actions.add_parser("score")
    for field in ("native", "fov", "bounded", "swap", "native-archive", "fov-archive", "design", "output"):
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
