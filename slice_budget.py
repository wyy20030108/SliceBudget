"""Retrospectively assign the same number of complete slice masks per arm."""

from hashlib import blake2s

import numpy as np


def phase_pairs(case, positions):
    indexes = case.slice_index[positions]
    labels = (1, 2, 3)
    areas = np.stack([np.count_nonzero(case.masks[positions] == label, axis=(1, 2)) for label in labels], axis=1)
    present = areas > 0
    candidates = {}
    for axis, label in enumerate(labels):
        small = np.flatnonzero(present[:, axis] & (areas[:, axis] * 4 <= areas[:, axis].max()))
        absent = np.flatnonzero(~present[:, axis])
        pairs = []
        for positive in small:
            distances = np.abs(indexes[absent].astype(int) - int(indexes[positive]))
            adjacent = absent[distances == 1]
            for near in adjacent:
                farther = absent[(distances >= 2) & np.all(present[absent] == present[near], axis=1)]
                if len(farther):
                    far = min(farther, key=lambda row: (abs(int(indexes[row]) - int(indexes[positive])), int(row)))
                    pairs.append((int(positions[positive]), int(positions[near]), int(positions[far])))
        if pairs:
            candidates[label] = pairs
    return candidates


def allocate(case, seed, budget=4):
    if budget != 4:
        raise ValueError("This comparison assigns two masks per phase")
    patient_hash = int.from_bytes(blake2s(case.patient_id.encode(), digest_size=8).digest(), "little")
    streams = np.random.SeedSequence([seed, patient_hash]).spawn(4)
    interval, random, adjacent, nonadjacent, records = [], [], [], [], []
    for phase in (0, 1):
        random_rng = np.random.default_rng(streams[2 * phase])
        pairing_rng = np.random.default_rng(streams[2 * phase + 1])
        positions = np.flatnonzero(case.phase == phase)
        positions = positions[np.argsort(case.slice_index[positions])]
        if len(positions) < 2:
            raise ValueError("Each phase needs at least two slices")
        regular = positions[np.floor(np.array([0.25, 0.75]) * len(positions)).astype(int)].tolist()
        interval.extend(regular)
        random.extend(random_rng.choice(positions, 2, replace=False).tolist())
        eligible = phase_pairs(case, positions)
        if eligible:
            target = int(pairing_rng.choice(list(eligible)))
            positive, near, far = eligible[target][pairing_rng.integers(len(eligible[target]))]
            adjacent.extend((positive, near))
            nonadjacent.extend((positive, far))
            records.append({"phase": phase, "target_class": target, "positive": positive,
                            "adjacent": near, "nonadjacent": far})
        else:
            adjacent.extend(regular)
            nonadjacent.extend(regular)
            records.append({"phase": phase, "target_class": None})
    return {"interval": sorted(interval), "random": sorted(random), "paired_adjacent": sorted(adjacent),
            "matched_nonadjacent": sorted(nonadjacent)}, {"phase_pairs": records}
