"""Cohort-level allocation of two slice annotations per patient and phase."""

import numpy as np


def phase_options(case, phase):
    positions = np.flatnonzero(case.phase == phase)
    positions = positions[np.argsort(case.slice_index[positions])]
    if len(positions) < 2:
        raise ValueError("Each cardiac phase needs at least two slices")
    areas = np.stack([(case.masks[positions] == label).sum(axis=(1, 2)) for label in (1, 2, 3)], axis=1)
    options = {}
    for col in range(3):
        area = areas[:, col]
        small = positions[(area > 0) & (4 * area <= area.max())]
        absent = positions[area == 0]
        regular = positions[(area > 0) & (4 * area > area.max())]
        if len(small) and len(absent) and len(regular):
            options[col + 1] = (small, absent, regular)
    return positions, options


def allocate_cohort(cases, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(cases))
    assignments = {arm: {patient: [] for patient in sorted(cases)}
                   for arm in ("interval", "random", "positive", "balanced")}
    labels = {patient: {} for patient in sorted(cases)}
    for phase in (0, 1):
        tasks = {str(patient): phase_options(cases[patient], phase) for patient in order}
        support = np.array([sum(label in options for _, options in tasks.values()) for label in (1, 2, 3)])
        used = np.zeros(3, dtype=int)
        for patient in sorted(tasks, key=lambda p: len(tasks[p][1])):
            positions, options = tasks[patient]
            interval = positions[(np.array([0.25, 0.75]) * len(positions)).astype(int)].tolist()
            assignments["interval"][patient].extend(interval)
            assignments["random"][patient].extend(rng.choice(positions, 2, replace=False).tolist())
            if not options:
                assignments["positive"][patient].extend(interval)
                assignments["balanced"][patient].extend(interval)
                labels[patient][str(phase)] = {"class": None}
                continue
            label = min(options, key=lambda label: (used[label - 1], support[label - 1], label))
            small, absent, regular = options[label]
            positive = int(rng.choice(small))
            alternative = int(min(regular, key=lambda row: (abs(int(row) - positive), int(row))))
            negative = int(min(absent, key=lambda row: (abs(int(row) - positive), int(row))))
            used[label - 1] += 1
            assignments["positive"][patient].extend((positive, alternative))
            assignments["balanced"][patient].extend((positive, negative))
            labels[patient][str(phase)] = {"class": label, "small": positive, "absent": negative,
                                           "typical_positive": alternative}
    return {arm: {patient: sorted(indices) for patient, indices in records.items()}
            for arm, records in assignments.items()}, labels
