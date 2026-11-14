"""Run the patient selection study on an explicit, disjoint ACDC protocol."""

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from acdc_data import load_cases, make_protocol
from segment import TrainState, configure, predict_many, train


def log(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dataset_digest(directory):
    manifest = hashlib.sha256()
    for file in sorted(directory.glob("patient*.npz")):
        manifest.update(file.name.encode("ascii"))
        manifest.update(hashlib.sha256(file.read_bytes()).digest())
    return manifest.hexdigest()


def scores(case, predicted, boundary=False):
    global_scores = []
    small = []
    distances = []
    missed = 0
    for label in (1, 2, 3):
        pred = predicted == label
        truth = case.masks == label
        overlap = (pred & truth).sum(axis=(1, 2))
        pred_count = pred.sum(axis=(1, 2))
        true_count = truth.sum(axis=(1, 2))
        for phase in np.unique(case.phase):
            phase_mask = case.phase == phase
            denominator = pred_count[phase_mask].sum() + true_count[phase_mask].sum()
            if denominator > 0:
                global_scores.append(2 * overlap[phase_mask].sum() / denominator)
            peak = true_count[phase_mask].max()
            region = phase_mask & (true_count > 0) & (true_count <= 0.25 * peak)
            for index in np.flatnonzero(region):
                small.append(2 * overlap[index] / (pred_count[index] + true_count[index]))
                missed += int(pred_count[index] == 0)
                if boundary:
                    spacing = getattr(case, "spacing_yx", None)
                    if spacing is None:
                        spacing_yx = (1.0, 1.0)
                    else:
                        spacing_yx = tuple(spacing[index])
                    distances.append(surface_distance(pred[index], truth[index], spacing_yx))
    return {
        "hard_dice": float(np.mean(small)) if small else None,
        "global_dice": float(np.mean(global_scores)) if global_scores else None,
        "hard_slice_classes": len(small),
        "hard_missed": missed,
        "hard_surface_distance": float(np.mean(distances)) if distances else None,
    }


def surface_distance(pred, truth, spacing):
    if not pred.any():
        return float(np.linalg.norm((np.asarray(pred.shape) - 1) * spacing))
    pred_edge = pred ^ ndimage.binary_erosion(pred)
    truth_edge = truth ^ ndimage.binary_erosion(truth)
    pred_to_truth = ndimage.distance_transform_edt(~truth_edge, sampling=spacing)[pred_edge]
    truth_to_pred = ndimage.distance_transform_edt(~pred_edge, sampling=spacing)[truth_edge]
    return float((pred_to_truth.mean() + truth_to_pred.mean()) / 2)


def evaluate(state, cases, patient_ids, device, boundary=False):
    if boundary and any(cases[patient].spacing_yx is None for patient in patient_ids):
        raise ValueError("ACDC boundary evaluation requires per-slice physical spacing")
    return {
        patient: scores(cases[patient], probabilities.argmax(axis=1), boundary)
        for patient, probabilities in zip(patient_ids, predict_many(state, (cases[item] for item in patient_ids), device))
    }


def vector(case, probabilities):
    # Bounded-size descriptors derived solely from information available before querying.
    images = case.images
    image_quantiles = np.quantile(images, [0.05, 0.25, 0.5, 0.75, 0.95])
    height_groups = np.array_split(images, 4, axis=1)
    appearance = [
        float(block.mean()) for row in height_groups for block in np.array_split(row, 4, axis=2)
    ]
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1))).sum(axis=1)
    foreground = 1 - probabilities[:, 0]
    output = [
        *image_quantiles, *appearance, *probabilities[:, 1:].mean(axis=(0, 2, 3)),
        float(entropy.mean()), float(entropy[foreground > 0.25].mean()) if (foreground > 0.25).any() else 0.0,
        float(np.quantile(entropy, 0.9)),
    ]
    return np.asarray(output, dtype=np.float64), output[-2]


def descriptors(state, cases, ids, initial, device):
    records = {
        patient: vector(cases[patient], probabilities)
        for patient, probabilities in zip(ids, predict_many(state, (cases[item] for item in ids), device))
    }
    scale = StandardScaler().fit(np.asarray([records[patient][0] for patient in initial]))
    initial_vectors = scale.transform([records[patient][0] for patient in initial])
    features = {}
    for patient, (values, _) in records.items():
        scaled = scale.transform([values])[0]
        distance = float(np.min(np.linalg.norm(initial_vectors - scaled, axis=1)))
        features[patient] = np.append(values, distance)
    return features, {patient: record[1] for patient, record in records.items()}


def representativeness(features, patients, initial):
    scaled = StandardScaler().fit_transform(np.asarray([features[item][:-1] for item in patients + initial]))
    candidates, starting = scaled[:len(patients)], scaled[len(patients):]
    distances = np.linalg.norm(candidates[:, None] - starting[None, :], axis=2).min(axis=1)
    ranking = []
    for _ in patients:
        index = int(distances.argmax())
        ranking.append(patients[index])
        distances = np.minimum(distances, np.linalg.norm(candidates - candidates[index], axis=1))
        distances[index] = -1
    return ranking


def mean_key(metrics, key):
    sample = [value[key] for value in metrics.values() if value[key] is not None]
    return float(np.mean(sample)) if sample else None


def paired_delta(after, before, key):
    sample = [
        after[patient][key] - before[patient][key]
        for patient in before
        if after[patient][key] is not None and before[patient][key] is not None
    ]
    if not sample:
        raise ValueError(f"Reference patients have no evaluable {key}; check anatomy coverage")
    return float(np.mean(sample))


def ranking_by_score(scores_by_id):
    return sorted(scores_by_id, key=lambda patient: (-scores_by_id[patient], patient))


def replay(args, cases, protocol, seed, common, candidates):
    initial = protocol["initial"]
    reference = protocol["reference"] + protocol["independent_reference"]
    unlabeled = protocol["replay"] + protocol["pool"]
    baseline = train(cases, initial, unlabeled, steps=args.steps, seed=seed, device=args.device,
                     start=common, batch_size=args.batch_size)
    before = evaluate(baseline, cases, reference, args.device)
    fit_ids = protocol["reference"]
    check_ids = protocol["independent_reference"]
    gains = {}
    for patient in candidates:
        started = time.monotonic()
        state = train(cases, initial + [patient], [item for item in unlabeled if item != patient],
                      steps=args.steps, seed=seed, device=args.device, start=common, batch_size=args.batch_size)
        after = evaluate(state, cases, reference, args.device)
        hard = paired_delta({item: after[item] for item in fit_ids}, {item: before[item] for item in fit_ids}, "hard_dice")
        global_gain = paired_delta({item: after[item] for item in fit_ids}, {item: before[item] for item in fit_ids}, "global_dice")
        independent = paired_delta({item: after[item] for item in check_ids},
                                   {item: before[item] for item in check_ids}, "hard_dice")
        gains[patient] = {"hard": hard, "global": global_gain, "independent_hard": independent,
                          "patient_metrics": after}
        log("replay_candidate", patient=patient, hard_gain=hard, global_gain=global_gain,
            independent_hard_gain=independent, seconds=round(time.monotonic() - started, 2))
    return {"baseline": before, "candidates": gains}


def reference_diagnostics(replay_results):
    candidates = list(replay_results["candidates"].values())
    primary = np.asarray([candidate["hard"] for candidate in candidates])
    independent = np.asarray([candidate["independent_hard"] for candidate in candidates])
    correlation = None
    if len(candidates) >= 3 and np.ptp(primary) > 0 and np.ptp(independent) > 0:
        correlation = float(spearmanr(primary, independent).statistic)
    return {"candidates": len(candidates), "primary_hard_gain_range": [float(primary.min()), float(primary.max())],
            "independent_hard_gain_range": [float(independent.min()), float(independent.max())],
            "identical_nonzero_sign": int((np.sign(primary) * np.sign(independent) > 0).sum()),
            "reference_spearman": correlation}


def fit_rankings(protocol, features, entropy, replay_results, seed, alpha):
    gain = replay_results["candidates"]
    candidates = sorted(gain)
    pool = protocol["pool"]
    x_fit = [features[patient] for patient in candidates]
    x_pool = [features[patient] for patient in pool]
    learned = {}
    for target in ("hard", "global"):
        predictor = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        predictor.fit(x_fit, [gain[patient][target] for patient in candidates])
        learned[target] = ranking_by_score(dict(zip(pool, map(float, predictor.predict(x_pool)))))
    return {
        **learned,
        "random": list(np.random.default_rng(seed + 1).permutation(pool)),
        "uncertainty": ranking_by_score({patient: entropy[patient] for patient in pool}),
        "representativeness": representativeness(features, pool, protocol["initial"]),
    }


def compare(args, cases, protocol, seed, common, rankings, eval_ids):
    initial = protocol["initial"]
    shared = set(protocol["validation"])
    used_by_ranker = shared | set(protocol["replay"]) | set(protocol["reference"]) | set(protocol["independent_reference"])
    ledgers = {}
    for method, order in rankings.items():
        development = used_by_ranker if method in ("hard", "global") else shared
        allowance = args.total_budget - len(development | set(initial))
        if not 1 <= allowance <= len(order):
            raise ValueError(f"Budget {args.total_budget} leaves {allowance} queries for {method} with pool {len(order)}")
        selected = order[:allowance]
        labels = set(initial + selected)
        unlabeled = [patient for patient in protocol["replay"] + protocol["pool"] if patient not in labels]
        started = time.monotonic()
        state = train(cases, initial + selected, unlabeled, steps=args.steps, seed=seed, device=args.device,
                      start=common, batch_size=args.batch_size)
        metrics = evaluate(state, cases, eval_ids, args.device, boundary=True)
        ledgers[method] = {
            "D": sorted(development), "S": initial, "Q": selected, "budget": len(development | labels),
            "patients": metrics, "hard_dice": mean_key(metrics, "hard_dice"),
            "global_dice": mean_key(metrics, "global_dice"),
            "hard_surface_distance": mean_key(metrics, "hard_surface_distance"),
        }
        log("method", method=method, seed=seed, queries=allowance, total=ledgers[method]["budget"],
            hard_dice=ledgers[method]["hard_dice"], global_dice=ledgers[method]["global_dice"],
            seconds=round(time.monotonic() - started, 2))
        save_state(args.output / f"{method}.pt", state)
    return ledgers


def bridge(args, cases, protocol, seed, common, rankings, count, baseline):
    results = {}
    initial = protocol["initial"]
    for method in (baseline, "hard"):
        selected = rankings[method][:count]
        labeled = initial + selected
        unlabeled = [item for item in protocol["replay"] + protocol["pool"] if item not in labeled]
        results[method] = {"selected": selected}
        for enabled in (False, True):
            state = train(cases, labeled, unlabeled, steps=args.steps, seed=seed, device=args.device,
                          start=common, batch_size=args.batch_size, uot=enabled)
            values = evaluate(state, cases, protocol["validation"], args.device, boundary=True)
            key = "uot_on" if enabled else "uot_off"
            results[method][key] = {"hard_dice": mean_key(values, "hard_dice"),
                                    "global_dice": mean_key(values, "global_dice"), "patients": values}
    return {"implementation": "DiffUnder UOT port in the common EMA trainer", "seed": seed,
            "query_budget": count, "budget_scope": "shared development annotations", "results": results}


def save_state(path, state):
    temporary = path.with_suffix(".part")
    torch.save(asdict(state), temporary)
    temporary.replace(path)


def load_state(path):
    return TrainState(**torch.load(path, map_location="cpu", weights_only=True))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_summary(held_out, method, key, seed=2026):
    pairs_by_seed = []
    for row in held_out.values():
        current, baseline = row["hard"]["patients"], row[method]["patients"]
        pair = {patient: current[patient][key] - baseline[patient][key]
                for patient in current if current[patient][key] is not None and baseline[patient][key] is not None}
        pairs_by_seed.append(pair)
    patients = sorted(set.intersection(*(set(pair) for pair in pairs_by_seed)))
    if not patients:
        return None
    difference = np.mean([[pair[patient] for patient in patients] for pair in pairs_by_seed], axis=0)
    rng = np.random.default_rng(seed)
    bootstrap = difference[rng.integers(0, len(difference), size=(4000, len(difference)))].mean(axis=1)
    return {"patient_count": len(patients), "mean": float(difference.mean()),
            "patient_bootstrap_ci95": list(map(float, np.quantile(bootstrap, [0.025, 0.975])))}


def verify_validation(runs, hard_margin, global_guard):
    """Return the baseline selected on validation and the predeclared paired checks."""
    reports = [load_json(run / "validation.json") for run in runs]
    baselines = ("random", "representativeness", "uncertainty")
    strongest = max(baselines, key=lambda method: np.mean([report[method]["hard_dice"] for report in reports]))
    checks = []
    for report in reports:
        hard, simple, global_reward = report["hard"], report[strongest], report["global"]
        hard_gain = hard["hard_dice"] - simple["hard_dice"]
        objective_gain = hard["hard_dice"] - global_reward["hard_dice"]
        whole_delta = hard["global_dice"] - simple["global_dice"]
        whole_objective_delta = hard["global_dice"] - global_reward["global_dice"]
        checks.append({
            "hard_vs_simple": hard_gain, "hard_vs_global_reward": objective_gain, "whole_vs_simple": whole_delta,
            "whole_vs_global_reward": whole_objective_delta,
            "passes": (hard_gain >= hard_margin and objective_gain >= hard_margin
                       and min(whole_delta, whole_objective_delta) >= -global_guard),
        })
    return strongest, checks


def confirm(args, cases, protocol, runs, registration_sha, data_sha):
    if len(runs) < 2 or args.hard_margin <= 0 or args.global_guard < 0:
        raise ValueError("Confirmation requires >=2 validation runs, a positive hard margin and nonnegative guard")
    configs = [load_json(run / "run.json") for run in runs]
    shared = ("protocol_sha256", "total_budget", "hard_definition", "steps", "warmup_steps", "ridge_alpha", "batch_size",
              "decision_registration_sha256", "dataset_sha256", "code_sha256")
    if any(config["mode"] != "validation" or any(config[key] != configs[0][key] for key in shared) for config in configs):
        raise ValueError("Confirmation requires matched full validation runs")
    if len({config["seed"] for config in configs}) != len(configs):
        raise ValueError("Validation runs must have separate paired seeds")
    if configs[0]["decision_registration_sha256"] != registration_sha:
        raise ValueError("Use the original validation registration to open the holdout")
    if configs[0]["dataset_sha256"] != data_sha or configs[0]["code_sha256"] != code_digest():
        raise ValueError("Keep the frozen data cache and experiment implementation for holdout evaluation")
    if any(load_json(run / "protocol.json") != protocol for run in runs):
        raise ValueError("Validation runs do not match this patient protocol")
    strongest, checks = verify_validation(runs, args.hard_margin, args.global_guard)
    bridge_exists = any((run / "bridge.json").is_file() for run in runs)
    result = {"strongest_simple": strongest, "hard_margin": args.hard_margin, "global_guard": args.global_guard,
              "checks": checks, "bridge_completed": bridge_exists,
              "runs": [str(run) for run in runs], "held_out_test_used": False}
    save_json(args.output / "validation_checks.json", result)
    if not bridge_exists or not all(check["passes"] for check in checks):
        log("holdout_sealed", strongest_simple=strongest, checks=checks, bridge_completed=bridge_exists)
        return
    marker = args.data / ".case_selection_holdout_started.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)
    save_json(args.output / "holdout_started.json", result)
    held_out = {}
    for run, config in zip(runs, configs):
        reports = {}
        for method, ledger in load_json(run / "validation.json").items():
            state = load_state(run / f"{method}.pt")
            values = evaluate(state, cases, protocol["test"], args.device, boundary=True)
            reports[method] = {"D": ledger["D"], "S": ledger["S"], "Q": ledger["Q"], "patients": values,
                               "hard_dice": mean_key(values, "hard_dice"), "global_dice": mean_key(values, "global_dice"),
                               "hard_surface_distance": mean_key(values, "hard_surface_distance")}
        held_out[str(config["seed"])] = reports
    comparisons = {
        method: {metric: paired_summary(held_out, method, metric)
                 for metric in ("hard_dice", "global_dice", "hard_surface_distance")}
        for method in (strongest, "global")
    }
    save_json(args.output / "holdout.json", {**result, "held_out_test_used": True, "results": held_out,
                                            "paired_comparisons": comparisons})
    log("holdout_complete", strongest_simple=strongest)


def code_digest():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("acdc_data.py", "segment.py", "experiment.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-budget", type=int)
    parser.add_argument("--bridge-budget", type=int)
    parser.add_argument("--bridge-baseline", default="random", choices=("random", "uncertainty", "representativeness"))
    parser.add_argument("--pilot-candidates", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--confirm-runs", type=Path, nargs="+")
    parser.add_argument("--hard-margin", type=float)
    parser.add_argument("--global-guard", type=float)
    parser.add_argument("--decision-registration", type=Path)
    args = parser.parse_args()
    if args.confirm_runs and (args.hard_margin is None or args.global_guard is None or args.decision_registration is None):
        parser.error("Confirmation requires the registration file, hard margin and overall guard")
    if not args.confirm_runs and (args.seed is None or args.warmup_steps is None or args.steps is None):
        parser.error("Pilot and validation require seed, warm-up steps and comparison steps")
    if not args.confirm_runs and (args.warmup_steps < 1 or args.steps < 1 or args.ridge_alpha <= 0 or args.pilot_candidates < 0):
        parser.error("Training steps and ridge alpha must be positive and pilot count nonnegative")
    if not args.confirm_runs and not args.pilot_candidates and (args.total_budget is None or args.decision_registration is None):
        parser.error("Provide a development-frozen budget and registration for full validation")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("Pick a fresh run directory to preserve existing results")
    configure(args.device)
    cases = load_cases(args.data)
    data_sha = dataset_digest(args.data)
    protocol = make_protocol(cases, args.protocol_seed)
    used = list(itertools.chain.from_iterable(protocol.values()))
    if len(used) != len(set(used)) or set(used) != set(cases):
        raise ValueError("Patient roles must partition the declared cohort without overlap")
    registration_sha = None
    if args.decision_registration is not None:
        reg_bytes = args.decision_registration.read_bytes()
        registration_sha = hashlib.sha256(reg_bytes).hexdigest()
        registration = json.loads(reg_bytes)
        declared = ("total_budget", "warmup_steps", "steps", "batch_size", "ridge_alpha", "hard_margin", "global_guard")
        match = ("hard_margin", "global_guard") if args.confirm_runs else ("total_budget", "warmup_steps", "steps", "batch_size", "ridge_alpha")
        if any(key not in registration for key in declared) or any(registration[key] != getattr(args, key) for key in match):
            parser.error("Arguments must match all declared settings in the registration file")
        if registration["hard_margin"] <= 0 or registration["global_guard"] < 0:
            parser.error("Registration must contain a positive hard margin and nonnegative global guard")
        if registration.get("protocol_seed", args.protocol_seed) != args.protocol_seed:
            parser.error("Patient split seed differs from registration")
        if not args.confirm_runs and "seeds" in registration and args.seed not in registration["seeds"]:
            parser.error("Training seed differs from registration")
        if not args.confirm_runs and "bridge" in registration:
            bridge_plan = registration["bridge"]
            expected = bridge_plan["queries"] if args.seed == bridge_plan["seed"] else None
            if args.bridge_budget != expected or args.bridge_baseline != bridge_plan["baseline"]:
                parser.error("Bridge configuration differs from registration")
    if args.confirm_runs:
        confirm(args, cases, protocol, args.confirm_runs, registration_sha, data_sha)
        return
    if args.pilot_candidates > len(protocol["replay"]):
        parser.error("Pilot candidate count exceeds replay pool")
    if args.bridge_budget is not None and not 1 <= args.bridge_budget <= len(protocol["pool"]):
        parser.error("Bridge query budget exceeds available selection patients")
    required_evaluation = protocol["reference"] + protocol["independent_reference"] + protocol["validation"]
    for patient in required_evaluation:
        metrics = scores(cases[patient], cases[patient].masks)
        log("evaluation_coverage", patient=patient, hard_slice_classes=metrics["hard_slice_classes"])
    config = {**vars(args), "data": str(args.data), "output": str(args.output),
              "decision_registration": str(args.decision_registration) if args.decision_registration else None,
              "decision_registration_sha256": registration_sha,
              "mode": "pilot" if args.pilot_candidates else "validation",
              "dataset_sha256": data_sha, "code_sha256": code_digest(),
              "pytorch_version": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(torch.device(args.device)) if torch.device(args.device).type == "cuda" else None,
              "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
              "hard_definition": "Positive slice/class area <= 25% of that patient's class peak in the same phase"}
    save_json(args.output / "run.json", config)
    save_json(args.output / "protocol.json", protocol)
    start = time.monotonic()
    common = train(cases, protocol["initial"], protocol["replay"] + protocol["pool"],
                   steps=args.warmup_steps, seed=args.seed, device=args.device, batch_size=args.batch_size)
    log("warmup", seconds=round(time.monotonic() - start, 2), train_seconds=round(common.seconds, 2),
        supervised_loss=common.supervised_loss, pseudo_fraction=common.pseudo_fraction)
    candidates = protocol["replay"][:args.pilot_candidates] if args.pilot_candidates else protocol["replay"]
    results = replay(args, cases, protocol, args.seed, common, candidates)
    save_json(args.output / "replay.json", results)
    diagnostics = reference_diagnostics(results)
    save_json(args.output / "reference_diagnostics.json", diagnostics)
    log("reference_diagnostics", **diagnostics)
    if args.pilot_candidates:
        log("pilot_complete", held_out_test_used=False)
        return
    ids = protocol["initial"] + protocol["replay"] + protocol["pool"]
    features, entropy = descriptors(common, cases, ids, protocol["initial"], args.device)
    rankings = fit_rankings(protocol, features, entropy, results, args.seed, args.ridge_alpha)
    save_json(args.output / "rankings.json", rankings)
    ledgers = compare(args, cases, protocol, args.seed, common, rankings, protocol["validation"])
    save_json(args.output / "validation.json", ledgers)
    if args.bridge_budget is not None:
        save_json(args.output / "bridge.json", bridge(args, cases, protocol, args.seed, common, rankings,
                                                       args.bridge_budget, args.bridge_baseline))
    log("validation_complete", held_out_test_used=False)


if __name__ == "__main__":
    main()
