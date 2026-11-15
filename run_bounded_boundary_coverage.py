"""Replace a bounded number of image-difference queries using calibrated state gains."""

import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np

from acdc_data import _load_case
from audit_external_transfer import digest
from calibrated_boundary_coverage import ADJACENT_ABSENCE, SMALL, build_groups, fit as fit_calibration
from experiment import load_json, log, save_json
from replay_audit_baselines import save_teacher
from run_boundary_coverage import acquisition_counts, score_cases, view_summary
from run_calibrated_boundary_coverage import anchors_for
from run_image_slice_budget import centers, image_view
from run_matched_slice_budget import aggregate, calibrate, collect
from segment import configure
from sparse_train import train_sparse


def bounded_selection(groups, initial, baseline, model, maximum, strength):
    coverage = {}
    choices = {}
    remaining = {}
    for phase, members in sorted(groups.items()):
        coverage[phase] = np.zeros(6)
        for patient, options in sorted(members.items()):
            matches = np.flatnonzero(np.isin(options.indices, baseline[patient]))
            if len(matches) != 1:
                raise ValueError("Expected one reference acquisition per patient and phase")
            current = int(matches[0])
            n_slices, n_labels, n_features = options.anchored.shape
            probabilities = model.predict(options.anchored.reshape(-1, n_features)).reshape(n_slices, n_labels, 3)
            credits = probabilities[:, :, [SMALL, ADJACENT_ABSENCE]].reshape(n_slices, 6)
            coverage[phase] += credits[current]
            slot = (phase, patient)
            choices[slot] = int(options.indices[current])
            remaining[slot] = (options, current, credits)
    replacements = []
    for _ in range(maximum):
        best_gain = 0.0
        for slot, (options, current, credits) in remaining.items():
            phase, _ = slot
            saturation = len(groups[phase]) / len(coverage[phase])
            delta = credits - credits[current]
            gain = options.novelty - options.novelty[current] + strength * saturation * np.log1p(
                delta / (saturation + coverage[phase])).sum(axis=1)
            alternate = int(gain.argmax())
            if gain[alternate] > best_gain:
                winner, choice, change = slot, alternate, delta[alternate]
                best_gain = float(gain[alternate])
        if best_gain <= 0:
            break
        options, _, _ = remaining.pop(winner)
        prior = choices[winner]
        choices[winner] = int(options.indices[choice])
        coverage[winner[0]] += change
        replacements.append({"patient": winner[1], "phase": winner[0], "old": prior,
                             "new": choices[winner], "gain": best_gain})
    selected = {p: sorted(initial[p] + [choices[phase, p] for phase in (0, 1)]) for p in initial}
    return selected, replacements


def fit(args):
    design = load_json(args.design)
    prior = load_json(args.prior / "run.json")
    original = load_json(args.prior / "frozen_acquisitions.json")
    settings = load_json(args.registration)
    if prior["registration_sha256"] != digest(args.registration):
        raise ValueError("Source protocol mismatch")
    initial = original["initial"]
    cache = {path.stem: path for path in args.data.glob("patient*.npz")}
    training, reliability, detection, development = (prior[key] for key in
        ("training", "reliability_fit", "detection_calibration", "development"))
    if (len(cache) != 100 or (len(training), len(reliability), len(detection), len(development)) != (40, 6, 6, 20)
            or len(set(training + reliability + detection + development)) != 72):
        raise ValueError("Expected the archived disjoint acquisition, calibration and evaluation roles")
    images = {patient: image_view(cache[patient]) for patient in training + reliability}
    training_cases = {p: _load_case(cache[p]) for p in training}
    reliability_cases = {p: _load_case(cache[p]) for p in reliability}
    reliability_images = {p: images[p] for p in reliability}
    reliability_initial = {p: centers(images[p]) for p in reliability}
    training_images = {p: images[p] for p in training}
    device = configure("cuda")
    options = {"seed": prior["seed"], "device": device, "batch_size": settings["batch_size"]}
    args.output.mkdir(parents=True)
    metadata = {**prior, "design_sha256": digest(args.design), "code_sha256": digest(Path(__file__))}
    save_json(args.output / "run.json", metadata)
    warmup = train_sparse(training_cases, initial, [], steps=settings["warmup_updates"], **options)
    raw, views = view_summary(images, warmup, device, design)
    del raw
    reference = build_groups(reliability_images, reliability_initial,
                             anchors_for(reliability_images, reliability_cases, reliability_initial),
                             {p: views[p] for p in reliability}, truth=reliability_cases)
    selection = build_groups(training_images, initial, anchors_for(training_images, training_cases, initial),
                             {p: views[p] for p in training})
    model = fit_calibration(reference, contiguous=True)
    chosen, replacements = bounded_selection(selection, initial, original["arms"]["pooled_image_farthest"],
                                            model, design["max_replacements"], design["strength"])
    if any(len(set(r)) != 4 for r in chosen.values()) or len(replacements) > design["max_replacements"]:
        raise ValueError("Acquisition violates the fixed mask budget")
    save_json(args.output / "frozen_acquisitions.json", {
        "initial": initial, "arms": {**{a: original["arms"][a] for a in design["arms"] if a != "bounded"},
                                     "bounded": chosen}, "replacements": replacements})
    save_json(args.output / "nomination_audit.json", {})
    save_json(args.output / "replacement_support.json", {"maximum": design["max_replacements"],
                                                         "replaced": len(replacements), "queries": 80})
    for arm in design["arms"]:
        if arm == "bounded":
            continue
        weight, result = args.prior / f"{arm}.pt", args.prior / f"{arm}.json"
        if digest(weight) != load_json(result)["weight_sha256"]:
            raise ValueError("Comparator checkpoint digest changed")
        path = args.output / weight.name
        try:
            path.hardlink_to(weight.resolve())
        except OSError:
            copyfile(weight, path)
        copyfile(result, args.output / result.name)
    model = train_sparse(training_cases, chosen, [], steps=settings["additional_updates"], start=warmup, **options)
    cutoff_cases = {p: _load_case(cache[p]) for p in detection}
    assessment = {p: _load_case(cache[p]) for p in development}
    cutoffs, support = calibrate(collect(cutoff_cases, model, device), settings["calibration_false_positive_rate"])
    result = score_cases(assessment, model, device, cutoffs)
    save_json(args.output / "bounded.json", {
        **result, "cutoffs": cutoffs, "calibration_support": support,
        "weight_sha256": save_teacher(args.output / "bounded.pt", model),
        "training_selection": acquisition_counts(training_cases, initial, chosen)})
    log("bounded_boundary_complete", fold=prior["fold"], seed=prior["seed"], replacements=len(replacements),
        metrics=aggregate(result["outcomes"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "data", "registration", "design", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    fit(parser.parse_args())
