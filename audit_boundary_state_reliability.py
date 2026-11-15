"""Assess archived boundary-state score rules on complete development candidate sets."""

import argparse
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from calibrated_boundary_coverage import build_groups, fit, reference_states, select
from experiment import load_json, log, save_json
from run_boundary_coverage import view_summary
from run_calibrated_boundary_coverage import anchors_for
from run_image_slice_budget import centers, image_view, patient_roster
from segment import configure
from sparse_train import train_sparse


CLASS_NAMES = ("RV", "MYO", "LV")
STATE_NAMES = ("small", "adjacent_absence", "other")
FEATURES = (("calibrated", True), ("calibrated_local", False))
ARMS = ("class_frequency", "ordered", "local")
BINS = np.asarray([0.0, 0.10, 0.25, 0.50, 0.75, 1.00])
DRAWS = 4000


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def truth_for(case, group, phase):
    indices = np.flatnonzero(case.phase == phase)
    indices = indices[np.argsort(case.slice_index[indices])]
    return reference_states(case.masks[indices])[np.isin(indices, group.indices)]


def probability_audit(probability, target):
    if (target.shape != probability.shape or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.allclose(probability.sum(axis=-1), 1.0)):
        raise ValueError("State forecasts must be finite simplex vectors over all candidates")
    assigned = np.clip(np.sum(probability * target, axis=-1), 1e-15, 1.0)
    brier = np.square(probability - target).mean(axis=0)
    log_loss = -np.log(assigned).mean(axis=0)
    bucket = np.searchsorted(BINS[1:-1], probability, side="right")
    bins = []
    for index in range(len(BINS) - 1):
        selected = bucket == index
        bins.append(np.stack([selected.sum(axis=0), (selected * probability).sum(axis=0),
                              (selected * target).sum(axis=0)]))
    return {"brier": brier.tolist(), "log_loss": log_loss.tolist(),
            "forecast_count": probability.sum(axis=0).tolist(), "bins": np.asarray(bins).tolist()}


def replay_fold(args):
    design, settings = load_json(args.design), load_json(args.registration)
    paths = {p.stem: p for p in args.data.glob("patient*.npz")}
    if len(paths) != 100:
        raise ValueError("Expected only the 100-case ACDC training release")
    training, calibration, development = patient_roster(paths, settings, args.fold)
    if tuple(map(len, (training, calibration, development))) != (40, 12, 20) or len(
            set(training + calibration + development)) != 72:
        raise ValueError("Expected disjoint training, calibration and development roles")
    reliability = calibration[::2]
    development = sorted(development)
    images = {p: image_view(paths[p]) for p in training + reliability + development}
    references = {p: _load_case(paths[p]) for p in images}
    train_images = {p: images[p] for p in training}
    train_cases = {p: references[p] for p in training}
    panel_images = {p: images[p] for p in reliability}
    panel_cases = {p: references[p] for p in reliability}
    dev_images = {p: images[p] for p in development}
    dev_cases = {p: references[p] for p in development}
    panel_initial = {p: centers(case) for p, case in panel_images.items()}
    dev_initial = {p: centers(case) for p, case in dev_images.items()}
    panel_anchors = anchors_for(panel_images, panel_cases, panel_initial)
    dev_anchors = anchors_for(dev_images, dev_cases, dev_initial)
    device = configure("cuda")
    results = {}
    for seed in design["seeds"]:
        archive = args.v2 / f"seed-{seed}-fold-{args.fold}"
        metadata = load_json(archive / "run.json")
        if (metadata["seed"] != seed or metadata["fold"] != args.fold
                or metadata["design_sha256"] != digest(args.design)
                or metadata["registration_sha256"] != digest(args.registration)
                or metadata["training"] != training or metadata["calibration"] != calibration
                or metadata["reliability_fit"] != reliability
                or metadata["detection_calibration"] != calibration[1::2]
                or sorted(metadata["development"]) != development):
            raise ValueError("Frozen fit metadata do not match the reassessment")
        roster_path = archive / "frozen_acquisitions.json"
        roster = load_json(roster_path)
        initial = roster["initial"]
        train_anchors = anchors_for(train_images, train_cases, initial)

        def recover(warmup):
            raw, views = view_summary({**train_images, **panel_images}, warmup, device, design)
            del raw
            sample = build_groups(panel_images, panel_initial, panel_anchors,
                                  {p: views[p] for p in reliability}, truth=panel_cases)
            candidates = build_groups(train_images, initial, train_anchors, {p: views[p] for p in training})
            models = []
            for arm, contiguous in FEATURES:
                model = fit(sample, contiguous=contiguous)
                selected, _, _ = select(candidates, initial, model, contiguous=contiguous, strength=design["strength"])
                if selected != roster["arms"][arm]:
                    return None
                models.append(model)
            return sample, models

        checkpoint = args.warmups / f"fold-{args.fold}" / "warmup.pt"
        recovered = None
        origin = "deterministic warm-up replay"
        if seed == 2027 and checkpoint.is_file():
            state = SimpleNamespace(teacher=torch.load(checkpoint, map_location="cpu", weights_only=True))
            recovered = recover(state)
            if recovered is not None:
                origin = "retained warm-up checkpoint"
        if recovered is None:
            state = train_sparse(train_cases, initial, [], steps=settings["warmup_updates"], seed=seed,
                                 device=device, batch_size=settings["batch_size"])
            recovered = recover(state)
        if recovered is None:
            raise ValueError("Frozen ordered/local acquisition replay failed")
        sample, models = recovered
        class_frequency = np.mean([np.eye(3)[np.concatenate([sample[phase][p].states for phase in sorted(sample)])]
                                   .mean(axis=0) for p in reliability], axis=0)
        raw, views = view_summary(dev_images, state, device, design)
        del raw
        candidates = build_groups(dev_images, dev_initial, dev_anchors, views)
        rows = []
        for p in development:
            groups = [candidates[phase][p] for phase in sorted(candidates)]
            target = np.eye(3)[np.concatenate([truth_for(dev_cases[p], candidates[phase][p], phase)
                                               for phase in sorted(candidates)])]
            forecasts = [np.broadcast_to(class_frequency, target.shape)]
            for attribute, model in zip(("anchored", "local"), models, strict=True):
                rows_by_slice = np.concatenate([getattr(group, attribute) for group in groups])
                forecasts.append(model.predict(rows_by_slice.reshape(-1, rows_by_slice.shape[-1])).reshape(target.shape))
            rows.append({"candidates": len(target), "reference_count": target.sum(axis=0).tolist(),
                         "arms": {arm: probability_audit(probability, target)
                                  for arm, probability in zip(ARMS, forecasts, strict=True)}})
        results[seed] = {"rows": rows, "training_panel_frequency": class_frequency.tolist(), "warm_up": origin,
                         "frozen_acquisitions_sha256": digest(roster_path), "exact_replay": list(dict(FEATURES))}
        log("boundary_state_reliability_complete", fold=args.fold, seed=seed, patients=len(development),
            nonanchor_slices=sum(row["candidates"] for row in rows))
    save_json(args.output, {"fold": args.fold, "results": results, "roster": development,
                           "design_sha256": digest(args.design), "code_sha256": digest(Path(__file__))})


def estimates(values, weights):
    patient_values = values.mean(axis=1)
    observed = patient_values.mean(axis=(0, 1))
    sampled = np.mean([draw @ rows / len(rows)
                       for draw, rows in zip(weights, patient_values, strict=True)], axis=0)
    return {"estimate": float(observed), "ci95": np.quantile(sampled, [0.025, 0.975]).tolist()}


def summarize_bins(array):
    output = {}
    for c, class_name in enumerate(CLASS_NAMES):
        output[class_name] = {}
        for k, state in enumerate(STATE_NAMES):
            rows = []
            for b, (left, right) in enumerate(zip(BINS[:-1], BINS[1:], strict=True)):
                count, prediction, observed = np.moveaxis(array[..., b, :, c, k], -1, 0)
                have = count > 0
                rows.append({"left": float(left), "right": float(right),
                             "patients": int(np.count_nonzero(have.any(axis=1))),
                             "candidate_assessments": int(count.sum()),
                             "mean_score": float(np.mean(prediction[have] / count[have])) if have.any() else None,
                             "reference_rate": float(np.mean(observed[have] / count[have])) if have.any() else None})
            output[class_name][state] = rows
    return output


def aggregate(args):
    files = sorted([load_json(path) for path in args.inputs], key=lambda value: value["fold"])
    if ([row["fold"] for row in files] != [0, 1] or len({row["design_sha256"] for row in files}) != 1
            or len({row["code_sha256"] for row in files}) != 1
            or len(set(files[0]["roster"] + files[1]["roster"])) != 40):
        raise ValueError("Development folds, provenance, and roster size must match")
    seeds = ("2027", "2028", "2029")
    fits = [[file["results"][seed] for seed in seeds] for file in files]
    reads = [[fit["rows"] for fit in resolved] for resolved in fits]
    if any(len(rows) != 20 for fold in reads for rows in fold):
        raise ValueError("Expected 20 candidates-bearing patients per development fold")

    def extract(getter):
        return np.asarray([[[getter(row) for row in rows] for rows in fold] for fold in reads])

    observed = extract(lambda row: row["reference_count"])
    counts = extract(lambda row: row["candidates"])
    if (not np.all(observed == observed[:, :1]) or not np.all(counts == counts[:, :1])
            or not np.all(observed.sum(axis=-1) == counts[..., None]) or np.any(counts < 1)):
        raise ValueError("Each repeated run must use the same full candidate census")
    rng = np.random.default_rng(20261117)
    weights = [rng.multinomial(20, np.full(20, 1 / 20), DRAWS) for _ in files]
    methods = {}
    metrics = {}
    for arm in ARMS:
        loss = extract(lambda row: row["arms"][arm]["log_loss"])
        brier = extract(lambda row: row["arms"][arm]["brier"])
        metrics[arm] = {"log_loss": loss.mean(axis=-1), "brier": brier.sum(axis=-1).mean(axis=-1)}
        forecasts = extract(lambda row: row["arms"][arm]["forecast_count"])
        bins = extract(lambda row: row["arms"][arm]["bins"])
        methods[arm] = {
            **{metric: estimates(values, weights) for metric, values in metrics[arm].items()},
            "by_class": {name: {
                "log_loss": estimates(loss[..., c], weights),
                "brier": estimates(brier[..., c, :].sum(axis=-1), weights),
                "states": {state: {
                    "brier": estimates(brier[..., c, k], weights),
                    "expected_count_across_40_patients": float(forecasts[..., c, k].mean(axis=1).sum()),
                    "reference_count_across_40_patients": int(observed[:, 0, :, c, k].sum()),
                } for k, state in enumerate(STATE_NAMES)},
            } for c, name in enumerate(CLASS_NAMES)},
            "reliability_bins": summarize_bins(bins),
        }
    comparisons = {}
    for arm, other in (("ordered", "class_frequency"), ("local", "class_frequency"), ("ordered", "local")):
        comparisons[f"{arm}_minus_{other}"] = {}
        for metric in ("brier", "log_loss"):
            values = metrics[arm][metric] - metrics[other][metric]
            comparisons[f"{arm}_minus_{other}"][metric] = {
                **estimates(values, weights),
                "by_seed": dict(zip(seeds, values.mean(axis=(0, 2)).tolist(), strict=True)),
            }
    report = {"scope": "Exploratory audit on all non-anchor development candidates; 20 held-out patients per fit.",
              "unit": "Patient-macro with equal class weights within patient; seeds are fixed and repeated.",
              "brier_definition": "Sum of the three one-vs-rest squared errors; lower is better.",
              "log_loss_definition": "Negative natural logarithm assigned to the observed state with 1e-15 floor.",
              "frequency_baseline": "Patient-equal class-wise frequency on the same six reliability-fitting patients.",
              "bin_definition": "Left inclusive, right exclusive except for final bin. Support counts unique patients; "
                                "candidate assessments count repeated seeds. Curves weight each nonempty patient-seed cell equally.",
              "patients": 40, "nonanchor_slices_by_fold": counts[:, 0].sum(axis=1).tolist(),
              "bootstrap_draws": DRAWS,
              "reference_support": {name: {state: int(np.count_nonzero(observed[:, 0, :, c, k]))
                                           for k, state in enumerate(STATE_NAMES)} for c, name in enumerate(CLASS_NAMES)},
              "methods": methods, "comparisons": comparisons,
              "provenance": {"design_sha256": files[0]["design_sha256"], "code_sha256": files[0]["code_sha256"],
                             "fits": [{seed: {k: v for k, v in fit.items() if k != "rows"}
                                       for seed, fit in zip(seeds, fold, strict=True)} for fold in fits]}}
    save_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    for field in ("registration", "design", "data", "v2", "warmups", "output"):
        score.add_argument(f"--{field}", required=True, type=Path)
    score.add_argument("--fold", choices=(0, 1), required=True, type=int)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--inputs", nargs=2, required=True, type=Path)
    combine.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (replay_fold if args.command == "score" else aggregate)(args)


if __name__ == "__main__":
    main()
