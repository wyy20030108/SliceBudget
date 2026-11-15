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
from image_slice_budget import CLASSES, balanced_assign, teacher_options
from segment import configure, predict_many


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
    candidates = {0: {}, 1: {}}
    for (patient, case), probabilities in zip(cases.items(), predict_many(teacher, cases.values(), device), strict=True):
        for phase in candidates:
            indices = np.flatnonzero(case.phase == phase)
            indices = indices[np.argsort(case.slice_index[indices])]
            first = next(index for index in acquisition["warmup"][patient] if case.phase[index] == phase)
            _, proposals = teacher_options(probabilities[indices], indices, case.slice_index[indices], first)
            candidates[phase][patient] = proposals

    rows = []
    fallback = 0
    for phase, options in candidates.items():
        balanced = balanced_assign(list(options.values()))
        for (patient, offered), winner in zip(options.items(), balanced, strict=True):
            archived = acquisition["ledger"][str(phase)][patient]["targeted"]
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
                rows.append((phase, label, intent, truth, (label, intent) == winner, index == archived["index"]))
    return rows, fallback


def summarize(proposals):
    counts = defaultdict(lambda: {
        group: {"count": 0, "true_state": dict.fromkeys(STATES, 0), "slice_acquired_under_another_intent": 0}
        for group in GROUPS
    })
    for phase, label, requested, true, selected, same_slice in proposals:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "runs", "weights", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
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
