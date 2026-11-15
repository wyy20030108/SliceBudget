"""Compare bounded coverage to state-free, image-loss matched slice replacements."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np

from acdc_data import _load_case
from audit_acquisition_baselines import block_features
from audit_external_transfer import digest
from experiment import load_json, log, save_json
from replay_audit_baselines import save_teacher
from run_boundary_coverage import acquisition_counts, score_cases
from run_image_slice_budget import image_view
from run_matched_slice_budget import aggregate, calibrate, collect
from segment import configure
from sparse_train import train_sparse


REFERENCE = "pooled_image_farthest"
CONTROL = "image_loss_matched"


def substitutions(images, initial, baseline, bounded, seed):
    """Choose alternate image-distance neighbors without access to state evidence."""
    rng = np.random.default_rng(seed)
    result = {patient: list(slices) for patient, slices in baseline.items()}
    swaps = []
    for patient, case in sorted(images.items()):
        features = block_features(case.images)
        for phase in (0, 1):
            anchor, = (i for i in initial[patient] if case.phase[i] == phase)
            eligible = np.flatnonzero((case.phase == phase) & (np.arange(len(case.images)) != anchor))
            old, = (i for i in baseline[patient] if i in eligible)
            state, = (i for i in bounded[patient] if i in eligible)
            if old == state:
                continue
            distances = np.square(features[eligible] - features[anchor]).sum(axis=1)
            distances /= max(float(distances.max()), 1e-12)
            novelty = dict(zip(eligible.tolist(), distances.tolist(), strict=True))
            losses = {i: novelty[old] - value for i, value in novelty.items() if i != old}
            alternates = [i for i in losses if i != state]
            if alternates:
                gap = min(abs(losses[i] - losses[state]) for i in alternates)
                tied = [i for i in alternates if np.isclose(abs(losses[i] - losses[state]), gap, rtol=0, atol=1e-12)]
                chosen = int(rng.choice(tied))
            else:
                chosen = state
            result[patient].remove(old)
            result[patient].append(chosen)
            swaps.append({"patient": patient, "phase": phase, "old": old, "bounded": state, "control": chosen,
                          "bounded_loss": losses[state], "control_loss": losses[chosen],
                          "distinct_alternatives": len(alternates)})
    return {p: sorted(slices) for p, slices in result.items()}, swaps


def fit(args):
    design = load_json(args.design)
    settings = load_json(args.registration)
    prior = load_json(args.prior / "run.json")
    if (prior["registration_sha256"] != digest(args.registration)
            or prior["design_sha256"] != digest(args.design.with_name(design["parent_design"]))
            or prior["seed"] not in design["seeds"] or prior["fold"] not in design["folds"]):
        raise ValueError("The source and bounded designs must match the archived experiment")
    training, reliability, calibration, development = (prior[key] for key in
        ("training", "reliability_fit", "detection_calibration", "development"))
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    if (len(cache) != 100 or tuple(map(len, (training, reliability, calibration, development))) != (40, 6, 6, 20)
            or len(set(training + reliability + calibration + development)) != 72):
        raise ValueError("Unexpected roster sizes or non-disjoint roles")
    frozen = load_json(args.prior / "frozen_acquisitions.json")
    images = {p: image_view(cache[p]) for p in training}
    initial = frozen["initial"]
    baseline, bounded = (frozen["arms"][arm] for arm in (REFERENCE, "bounded"))
    selected, swaps = substitutions(images, initial, baseline, bounded, prior["seed"])
    if len(swaps) != design["requested_swaps"] or any(len(set(s)) != 4 for s in selected.values()):
        raise ValueError("Expected eight slot-matched replacements and four masks per patient")
    args.output.mkdir(parents=True)
    metadata = {**prior, "parent_design_sha256": prior["design_sha256"], "design_sha256": digest(args.design),
                "code_sha256": digest(Path(__file__))}
    save_json(args.output / "run.json", metadata)
    save_json(args.output / "frozen_acquisitions.json",
              {"initial": initial, "arms": {REFERENCE: baseline, "bounded": bounded, CONTROL: selected},
               "image_loss_matches": swaps})
    save_json(args.output / "nomination_audit.json", {})
    for arm in (REFERENCE, "bounded"):
        weights, results = args.prior / f"{arm}.pt", args.prior / f"{arm}.json"
        if digest(weights) != load_json(results)["weight_sha256"]:
            raise ValueError(f"Invalid archived checkpoint for {arm}")
        path = args.output / weights.name
        try:
            path.hardlink_to(weights.resolve())
        except OSError:
            copyfile(weights, path)
        copyfile(results, args.output / results.name)
    truth = {p: _load_case(cache[p]) for p in training}
    cutoff_cases = {p: _load_case(cache[p]) for p in calibration}
    evaluation = {p: _load_case(cache[p]) for p in development}
    device = configure("cuda")
    options = {"seed": prior["seed"], "batch_size": settings["batch_size"], "device": device}
    warmup = train_sparse(truth, initial, [], steps=settings["warmup_updates"], **options)
    model = train_sparse(truth, selected, [], steps=settings["additional_updates"], start=warmup, **options)
    cutoffs, support = calibrate(collect(cutoff_cases, model, device), settings["calibration_false_positive_rate"])
    result = score_cases(evaluation, model, device, cutoffs)
    save_json(args.output / f"{CONTROL}.json", {**result, "cutoffs": cutoffs, "calibration_support": support,
              "weight_sha256": save_teacher(args.output / f"{CONTROL}.pt", model),
              "training_selection": acquisition_counts(truth, initial, selected)})
    log("image_loss_matched_complete", fold=prior["fold"], seed=prior["seed"],
        distinct=sum(row["bounded"] != row["control"] for row in swaps), metrics=aggregate(result["outcomes"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "data", "registration", "design", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    fit(parser.parse_args())
