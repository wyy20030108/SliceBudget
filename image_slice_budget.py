"""Allocate a second round of dense slice labels using images and teacher predictions."""

from hashlib import blake2s

import numpy as np
from scipy.optimize import linear_sum_assignment


CLASSES = (1, 2, 3)
STRATA = tuple((label, state) for label in CLASSES for state in ("small", "absent"))


def upper_pixel_average(probabilities):
    flat = probabilities.reshape(*probabilities.shape[:-2], -1)
    count = max(1, flat.shape[-1] // 100)
    tail = np.partition(flat, flat.shape[-1] - count, axis=-1)[..., -count:]
    return tail.mean(axis=-1)


def regular_slice(candidates, positions, first, rng):
    before = candidates[positions[candidates] < positions[first]]
    after = candidates[positions[candidates] > positions[first]]
    groups = [group for group in (before, after) if len(group)]
    longest = max(map(len, groups))
    largest = [group for group in groups if len(group) == longest]
    group = largest[int(rng.integers(len(largest)))]
    return int(group[len(group) // 2])


def teacher_options(probability, indices, native_indices, first):
    prediction = probability.argmax(axis=1)
    clipping = np.maximum(probability, 1e-8)
    entropy = -np.sum(probability * np.log(clipping), axis=1)
    candidates = np.flatnonzero(indices != first)
    uncertainty = int(candidates[np.argmax(upper_pixel_average(entropy)[candidates])])

    mass = upper_pixel_average(probability[:, CLASSES])
    options = {}
    for column, label in enumerate(CLASSES):
        pixels = prediction == label
        area = pixels.sum(axis=(1, 2))
        apparent = np.flatnonzero(area > 0)
        if len(apparent) == 0:
            continue
        relative_area = area / area.max()
        local = mass[:, column]
        predicted_small = candidates[(area[candidates] > 0) & (relative_area[candidates] <= 0.25)]
        if len(predicted_small):
            confidence = (probability[:, label] * pixels).sum(axis=(1, 2)) / np.maximum(area, 1)
            best = int(predicted_small[np.argmax(confidence[predicted_small])])
            options[(label, "small")] = (int(indices[best]), float(confidence[best]))

        absent = candidates[(area[candidates] == 0) & (local[candidates] <= 0.35)]
        if len(absent):
            neighbor_distance = np.abs(native_indices[absent, None] - native_indices[None, apparent]).min(axis=1)
            # Prefer credible absence near the predicted footprint over easy, remote background.
            quality = (1 - local[absent] / 0.35) / np.sqrt(neighbor_distance)
            best = int(np.argmax(quality))
            options[(label, "absent")] = (int(indices[absent[best]]), float(quality[best]))
    return int(indices[uncertainty]), options


def balanced_assign(options):
    count = len(options)
    discount = np.log1p(np.arange(1, count + 1)) - np.log1p(np.arange(count))
    utility = np.full((count, (len(STRATA) + 1) * count), -1.0)
    utility[:, len(STRATA) * count:] = 0
    for row, feasible in enumerate(options):
        for stratum, (_, quality) in feasible.items():
            start = STRATA.index(stratum) * count
            utility[row, start:start + count] = discount + (1e-4 / count) * quality
    rows, columns = linear_sum_assignment(utility, maximize=True)
    assert list(rows) == list(range(count))
    return [STRATA[column // count] if column < len(STRATA) * count else None for column in columns]


def allocate_image(cases, warmup_selection, probabilities_by_patient, seed):
    """Return complete four-slice lists and an image-prediction-only allocation ledger."""
    patients = sorted(cases)
    if not patients or set(patients) != set(warmup_selection) or set(patients) != set(probabilities_by_patient):
        raise ValueError("Supply identical nonempty patient sets for cases, warmup and predictions")
    arms = {arm: {patient: list(map(int, warmup_selection[patient])) for patient in patients}
            for arm in ("interval", "uncertain", "targeted", "targeted_unbalanced")}
    ledger = {}
    for phase in (0, 1):
        available, phase_selections = [], []
        for patient in patients:
            case = cases[patient]
            probability = np.asarray(probabilities_by_patient[patient])
            selected = arms["interval"][patient][:2]
            if (probability.ndim != 4 or probability.shape[:2] != (len(case.phase), 4)
                or len(selected) != 2 or min(selected) < 0 or max(selected) >= len(case.phase)
                or sorted(np.asarray(case.phase)[selected].tolist()) != [0, 1]):
                raise ValueError(f"Invalid predictions or warmup slices for {patient}")
            indices = np.flatnonzero(case.phase == phase)
            native = np.asarray(case.slice_index)[indices].astype(int)
            order = np.argsort(native)
            indices, native = indices[order], native[order]
            if len(indices) < 2 or len(np.unique(native)) != len(native):
                raise ValueError(f"Expected at least two distinct slices per phase for {patient}")
            first = next(slice_id for slice_id in selected if case.phase[slice_id] == phase)
            patient_key = int.from_bytes(blake2s(patient.encode(), digest_size=8).digest(), "little")
            rng = np.random.default_rng(np.random.SeedSequence([seed, phase, patient_key]))
            center = int(np.flatnonzero(indices == first)[0])
            regular = regular_slice(np.flatnonzero(indices != first), native, center, rng)
            uncertain, options = teacher_options(probability[indices], indices, native, first)
            available.append(options)
            phase_selections.append((int(indices[regular]), uncertain))

        chosen = balanced_assign(available)
        ledger[str(phase)] = {}
        for patient, assignment, feasible, (interval, uncertain) in zip(
                patients, chosen, available, phase_selections, strict=True):
            unbalanced = max(feasible, key=lambda key: feasible[key][1], default=None)
            arms["interval"][patient].append(interval)
            arms["uncertain"][patient].append(uncertain)
            entries = {}
            for arm, stratum in (("targeted", assignment), ("targeted_unbalanced", unbalanced)):
                slice_id, confidence = (interval, None) if stratum is None else feasible[stratum]
                arms[arm][patient].append(slice_id)
                entries[arm] = {"index": slice_id, "class": stratum[0] if stratum else None,
                                "predicted_state": stratum[1] if stratum else None, "score": confidence}
            ledger[str(phase)][patient] = entries

    return {arm: {patient: sorted(indices) for patient, indices in roster.items()}
            for arm, roster in arms.items()}, ledger
