"""Compare warm-up teacher proposals before and after balanced assignment."""

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from acdc_data import _load_case
from audit_external_transfer import digest
from experiment import load_json, save_json
from image_slice_budget import CLASSES, allocate_image, balanced_assign, teacher_options
from segment import configure, predict_many
from sparse_train import train_sparse


GROUPS = ("offered", "selected", "passed_over")
STATES = ("small", "absent", "typical_positive")
CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def audit_fold(data, original, weights, fold, device):
    development = load_json(original / "run.json")
    acquisition_path = original / "frozen_acquisitions.json"
    acquisition = load_json(acquisition_path)
    replay = load_json(weights / "source_replay.json")
    checkpoint = weights / "warmup.pt"
    if (development["fold"] != fold or development["held_out_test_used"] or len(development["training"]) != 40
            or replay["fold"] != fold or replay["training"] != development["training"]
            or replay["registration_sha256"] != development["registration_sha256"]
            or replay["frozen_acquisitions_sha256"] != digest(acquisition_path)
            or replay["warmup_checkpoint_sha256"] != digest(checkpoint)):
        raise ValueError("The warm-up checkpoint and archived development plan must match")
    ids = sorted(development["training"])
    if set(ids) != set(acquisition["warmup"]):
        raise ValueError("The development roster and frozen warm-up mask roster differ")
    cases = {key: _load_case(data / f"{key}.npz") for key in ids}
    teacher = SimpleNamespace(teacher=torch.load(checkpoint, map_location="cpu", weights_only=True))
    return audit_proposals(cases, acquisition["warmup"], teacher, device,
                           dict(acquisition, seed=development["seed"]))


def audit_proposals(cases, initial, teacher, device, acquisition):
    probabilities = dict(zip(cases, predict_many(teacher, cases.values(), device), strict=True))
    replayed, ledger = allocate_image(cases, initial, probabilities, acquisition["seed"])
    for arm in ("targeted", "uncertain"):
        if replayed[arm] != acquisition["arms"][arm]:
            raise ValueError(f"Replayed {arm} acquisition differs from the archived roster")
    candidates = {0: {}, 1: {}}
    for patient, case in cases.items():
        for phase in candidates:
            indices = np.flatnonzero(case.phase == phase)
            indices = indices[np.argsort(case.slice_index[indices])]
            first = next(index for index in initial[patient] if case.phase[index] == phase)
            _, proposals = teacher_options(probabilities[patient][indices], indices, case.slice_index[indices], first)
            candidates[phase][patient] = proposals

    rows = []
    fallback = 0
    for phase, options in candidates.items():
        balanced = balanced_assign(list(options.values()))
        for (patient, offered), winner in zip(options.items(), balanced, strict=True):
            archived = (acquisition.get("ledger") or ledger)[str(phase)][patient]["targeted"]
            archived_label = archived["class"]
            archived_intent = None if archived_label is None else (archived_label, archived["predicted_state"])
            if winner != archived_intent:
                raise ValueError("Replayed assignment differs from the archived class/state")
            if winner is None:
                fallback += 1
            else:
                index, quality = offered[winner]
                if index != archived["index"] or not np.isclose(quality, archived["score"], atol=1e-7, rtol=0):
                    raise ValueError("Replayed proposal differs from the archived chosen slice or score")
            case = cases[patient]
            class_areas = {label: np.count_nonzero(case.masks == label, axis=(1, 2)) for label in CLASSES}
            for (label, intent), (index, _) in offered.items():
                areas = class_areas[label]
                area = areas[index]
                peak = areas[case.phase == phase].max()
                truth = "absent" if not area else "small" if 4 * area <= peak else "typical_positive"
                rows.append((phase, label, intent, truth, (label, intent) == winner,
                             index == archived["index"], patient, index))
    return rows, fallback


def summarize(proposals):
    counts = defaultdict(lambda: {
        group: {"count": 0, "true_state": dict.fromkeys(STATES, 0), "slice_acquired_under_another_intent": 0}
        for group in GROUPS
    })
    for phase, label, requested, true, selected, same_slice, _, _ in proposals:
        for key in (requested, f"{CLASS_NAMES[label]}/{requested}",
                    f"{CLASS_NAMES[label]}/{('ED', 'ES')[phase]}/{requested}"):
            for group in ("offered", "selected" if selected else "passed_over"):
                tally = counts[key][group]
                tally["count"] += 1
                tally["true_state"][true] += 1
                tally["slice_acquired_under_another_intent"] += int(not selected and same_slice)
    for key, by_group in counts.items():
        state = key.rsplit("/", 1)[-1]
        for tally in by_group.values():
            hits = tally["true_state"][state]
            tally["hits"] = hits
            tally["precision"] = hits / tally["count"] if tally["count"] else None
    return dict(sorted(counts.items()))


def repeat(args):
    settings = load_json(args.registration)
    folder = Path(f"fold-{args.fold}")
    source = load_json(args.runs / folder / "run.json")
    device = configure("cuda")
    if args.seed == settings["training_seed"]:
        rows, fallback = audit_fold(args.data, args.runs / folder, args.weights / folder, args.fold, device)
        training = source["training"]
    else:
        run_dir = args.seed_runs / f"seed-{args.seed}-fold-{args.fold}"
        run = load_json(run_dir / "run.json")
        acquisition = load_json(run_dir / "frozen_acquisitions.json")
        training = run["training"]
        if (run["fold"] != args.fold or run["seed"] != args.seed or source["training"] != training
                or run["registration_sha256"] != digest(args.registration)):
            raise ValueError("Repeated candidate audit requires the archived source patient roles and schedule")
        initial = acquisition["initial"]
        cases = {patient: _load_case(args.data / f"{patient}.npz") for patient in sorted(training)}
        if set(initial) != set(cases) or len(cases) != 40:
            raise ValueError("Warm-up masks must identify exactly the 40 training cases")
        training_cases = {patient: cases[patient] for patient in training}
        teacher = train_sparse(training_cases, initial, [], seed=args.seed, device=device,
                               batch_size=settings["batch_size"], steps=settings["warmup_updates"])
        rows, fallback = audit_proposals(cases, initial, teacher, device, dict(acquisition, seed=args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"fold": args.fold, "seed": args.seed, "training": training,
                            "fallback": fallback, "rows": rows})


def cluster_summary(rows, training, draws):
    report = summarize(rows)
    index = {key: i for i, key in enumerate(training)}
    counts = defaultdict(lambda: np.zeros((2, len(training)), dtype=np.int32))
    physical = defaultdict(set)
    for phase, label, intent, truth, selected, _, patient, position in rows:
        for key in (intent, f"{CLASS_NAMES[label]}/{intent}", f"{CLASS_NAMES[label]}/{('ED', 'ES')[phase]}/{intent}"):
            for group in ("offered", "selected" if selected else "passed_over"):
                counts[key, group][0, index[patient]] += int(intent == truth)
                counts[key, group][1, index[patient]] += 1
                physical[key, group].add((patient, position))
    for (key, group), (hits, n) in counts.items():
        denominators = draws @ n
        values = (draws @ hits)[denominators > 0] / denominators[denominators > 0]
        report[key][group].update({
            "patients": int((n > 0).sum()), "physical_slices": len(physical[key, group]),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None,
        })
    return report


def combine(args):
    by_seed = defaultdict(dict)
    for path in args.repeat_files:
        result = load_json(path)
        seed, fold = result["seed"], result["fold"]
        if fold in by_seed[seed]:
            raise ValueError("A seed and fold occurred twice")
        by_seed[seed][fold] = result
    if set(by_seed) != {2027, 2028, 2029}:
        raise ValueError("Expected the three previously assessed seeds")
    output = {}
    for seed, folds in sorted(by_seed.items()):
        if set(folds) != {0, 1}:
            raise ValueError("Two independent training rosters are needed per seed")
        training = [sorted(folds[fold]["training"]) for fold in (0, 1)]
        if any(len(roster) != len(set(roster)) or len(roster) != 40 for roster in training) or set(training[0]) & set(training[1]):
            raise ValueError("Expected two disjoint 40-case source training blocks")
        rng = np.random.default_rng(20261129)
        draws = np.hstack([rng.multinomial(40, np.full(40, 1 / 40), size=4000) for _ in folds])
        rows = [row for fold in folds.values() for row in fold["rows"]]
        output[seed] = {"groups": cluster_summary(rows, training[0] + training[1], draws),
                        "fallback": sum(item["fallback"] for item in folds.values()), "assignments": 160}
    for state, hits, count in (("absent", 18, 76), ("small", 168, 317)):
        original = output[2027]["groups"][state]["offered"]
        if (original["hits"], original["count"]) != (hits, count):
            raise ValueError(f"Original {state} audit does not reproduce")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {"seeds": output, "bootstrap_draws": 4000,
                           "precision_interval": "Patient-clustered stratified bootstrap within each fitted source block; assignments held fixed",
                           "scored_after_assignment": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "weights", "output", "registration", "seed-runs"):
        parser.add_argument(f"--{name}", required=name == "output", type=Path)
    parser.add_argument("--repeat-seed", dest="seed", type=int, choices=(2027, 2028, 2029))
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--repeat-files", type=Path, nargs="+")
    args = parser.parse_args()
    if args.repeat_files:
        combine(args)
        return
    if not all((args.data, args.runs, args.weights)):
        parser.error("--data, --runs and --weights are required to replay candidates")
    if args.seed:
        if args.fold is None or not all((args.registration, args.seed_runs)):
            parser.error("seed replay requires --fold, --registration, and --seed-runs")
        repeat(args)
        return
    device = configure("cuda")
    all_proposals = []
    fallback = {}
    by_fold = {}
    for fold in (0, 1):
        folder = Path(f"fold-{fold}")
        proposals, fallback[fold] = audit_fold(args.data, args.runs / folder, args.weights / folder, fold, device)
        by_fold[fold] = summarize(proposals)
        all_proposals.extend(proposals)
    total = summarize(all_proposals)
    for intent, hits, count in (("absent", 14, 54), ("small", 53, 98)):
        if (total[intent]["selected"]["hits"], total[intent]["selected"]["count"]) != (hits, count):
            raise ValueError(f"Selected {intent} count does not reproduce the archived audit")
    if sum(fallback.values()) != 8:
        raise ValueError("Interval fallback does not match the archived audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, {
        "scope": "Frozen source warm-up proposals in two 40-patient ACDC development training folds",
        "proposal_unit": "One best offered slice per patient, phase, class and requested state. "
                         "A slice may be proposed for multiple class/state pairs.",
        "selected_definition": "Same class/state and slice chosen by the balanced assignment. "
                               "Interval fallback has no selected class/state.",
        "small_definition": "Positive reference area at most one quarter of that patient's class peak in its phase.",
        "reference_use": "Reference masks are used only after reproducing the candidate set and assignment.",
        "fallback_by_fold": fallback,
        "groups": total,
        "by_fold": by_fold,
    })
    for intent in ("absent", "small"):
        print(intent, total[intent], flush=True)


if __name__ == "__main__":
    main()
