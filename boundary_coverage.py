"""Allocate slice masks using uncertain boundary evidence from measured anchors."""

import numpy as np
from scipy.special import softmax

from audit_acquisition_baselines import block_features
from image_slice_budget import CLASSES


def endpoint_credit(scores, center, contiguous):
    """Return inside and first-outside weights along both sides of the anchor."""
    inside = np.ones(len(scores))
    boundary = np.zeros(len(scores))
    for side in (np.arange(center - 1, -1, -1), np.arange(center + 1, len(scores))):
        evidence = scores[side]
        if contiguous:
            log_odds = np.log(evidence) - np.log1p(-evidence)
            endpoint = softmax(np.r_[0., np.cumsum(log_odds)])
            inside[side] = np.cumsum(endpoint[:0:-1])[::-1]
            boundary[side] = endpoint[:-1]
        else:
            inside[side] = evidence
            boundary[side] = (1 - evidence) * np.r_[1., evidence][:-1] if len(side) else []
    return inside, boundary


def scores_from_views(probabilities, tail_fraction, area_cutoff):
    foreground = probabilities[:, 1:].reshape(len(probabilities), len(CLASSES), -1)
    count = max(1, round(foreground.shape[-1] * tail_fraction))
    tail = np.partition(foreground, -count, axis=-1)[..., -count:].mean(axis=-1)
    area = (foreground >= area_cutoff).sum(axis=-1)
    return tail, area


def state_credits(evidence, areas, center, anchor, contiguous):
    credits = np.zeros((len(evidence), 2 * len(CLASSES)))
    for column, label in enumerate(CLASSES):
        if not (anchor == label).any():
            continue
        size = areas[:, column]
        small = (size > 0) & (4 * size <= size.max())
        inside, boundary = endpoint_credit(evidence[:, column], center, contiguous)
        credits[:, 2 * column] = inside * small
        credits[:, 2 * column + 1] = boundary
    return credits


def greedy_queries(groups, strength):
    """One slice per group, maximizing the marginal concave state coverage."""
    pending = dict(groups)
    count = next(iter(groups.values()))[1].shape[-1]
    coverage = np.zeros(count)
    saturation = len(groups) / count
    selection = {}
    while pending:
        best_value = -np.inf
        for patient, (indices, credits, distance) in pending.items():
            bonus = saturation * np.log1p(credits / (saturation + coverage))
            value = distance + strength * bonus.sum(axis=1)
            position = int(value.argmax())
            if value[position] > best_value:
                best_value = value[position]
                winner = patient
                entry = indices[position], credits[position]
        index, credit = entry
        selection[winner] = (int(index), credit.tolist())
        coverage += credit
        del pending[winner]
    return selection


def select_boundaries(cases, initial, anchors, view_summaries, strength, contiguous):
    """Use images, frozen-teacher summaries, and the two existing masks only."""
    if set(cases) != set(initial) or set(cases) != set(anchors) or set(cases) != set(view_summaries):
        raise ValueError("Patient sets for acquisition differ")
    features = {patient: block_features(case.images) for patient, case in cases.items()}
    selected, ledger = {patient: list(initial[patient]) for patient in sorted(cases)}, {}
    nominations = []
    for phase in (0, 1):
        groups = {}
        for patient in sorted(cases):
            case = cases[patient]
            indices = np.flatnonzero(case.phase == phase)
            indices = indices[np.argsort(case.slice_index[indices])]
            if not np.all(np.diff(case.slice_index[indices]) == 1):
                raise ValueError("Expected contiguous native phase positions")
            anchor_index = next(index for index in initial[patient] if case.phase[index] == phase)
            center = int(np.flatnonzero(indices == anchor_index)[0])
            evidence, area = view_summaries[patient]
            credits = state_credits(evidence[indices], area[indices], center, anchors[patient][anchor_index], contiguous)
            keep = indices != anchor_index
            candidates = indices[keep]
            distance = np.square(features[patient][candidates] - features[patient][anchor_index]).sum(axis=1)
            novelty = distance / max(distance.max(), 1e-12)
            groups[patient] = (candidates, credits[keep], novelty)
            for column, label in enumerate(CLASSES):
                absence = credits[keep, 2 * column + 1]
                best = int(absence.argmax())
                if absence[best] > 0:
                    nominations.append((patient, phase, label, int(candidates[best]), float(absence[best])))
        ledger[phase] = greedy_queries(groups, strength)
        for patient, (index, _) in ledger[phase].items():
            selected[patient].append(index)
    return {patient: sorted(indices) for patient, indices in selected.items()}, ledger, nominations
