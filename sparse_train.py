"""Expose only selected slice masks to the existing segmentation trainer."""

from types import SimpleNamespace

import numpy as np

from segment import train


def sparse_views(cases, labeled_slices, unlabeled_ids, consistency):
    views = {}
    labeled = []
    unlabeled = []
    if len(unlabeled_ids) != len(set(unlabeled_ids)) or set(labeled_slices).intersection(unlabeled_ids):
        raise ValueError("Use distinct partially annotated and completely unannotated patients")
    for patient, selection in labeled_slices.items():
        case = cases[patient]
        selected = np.asarray(selection)
        if (selected.ndim != 1 or not len(selected) or selected.dtype.kind not in "iu"
                or selected.min() < 0 or selected.max() >= len(case.images)
                or len(np.unique(selected)) != len(selected)):
            raise ValueError(f"Invalid annotated indices for {patient}")
        labeled_key = (patient, "annotated")
        views[labeled_key] = SimpleNamespace(images=case.images[selected], masks=case.masks[selected])
        labeled.append(labeled_key)
        if consistency and len(selected) < len(case.images):
            keep = np.ones(len(case.images), dtype=bool)
            keep[selected] = False
            other_key = (patient, "unannotated")
            views[other_key] = SimpleNamespace(images=case.images[keep])
            unlabeled.append(other_key)
    if consistency:
        for patient in unlabeled_ids:
            key = (patient, "unannotated")
            views[key] = SimpleNamespace(images=cases[patient].images)
            unlabeled.append(key)
    return views, labeled, unlabeled


def train_sparse(cases, labeled_slices, unlabeled_ids, *, steps, seed, device, batch_size=4,
                 start=None, uot=False, consistency=True):
    views, labeled, unlabeled = sparse_views(cases, labeled_slices, unlabeled_ids, consistency)
    return train(views, labeled, unlabeled, steps=steps, seed=seed, device=device,
                 batch_size=batch_size, start=start, uot=uot)
