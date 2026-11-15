"""Calibrate local and anchor-conditioned boundary features on selection labels."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from audit_acquisition_baselines import block_features
from boundary_coverage import endpoint_credit, greedy_queries
from image_slice_budget import CLASSES


SMALL, ADJACENT_ABSENCE, OTHER = range(3)


@dataclass(frozen=True)
class Group:
    indices: np.ndarray
    novelty: np.ndarray
    local: np.ndarray
    anchored: np.ndarray
    states: np.ndarray | None


@dataclass(frozen=True)
class Calibration:
    contiguous: bool
    observed_states: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def matrix(self, rows):
        standardized = (rows - self.center) / self.scale
        return np.column_stack((np.ones(len(rows)), standardized))

    def predict(self, rows):
        probabilities = np.zeros((len(rows), 3), dtype=np.float64)
        probabilities[:, self.observed_states] = softmax(self.matrix(rows) @ self.coefficients, axis=1)
        return probabilities


def reference_states(masks):
    areas = np.stack([(masks == label).sum(axis=(1, 2)) for label in CLASSES], axis=1)
    small = (areas > 0) & (4 * areas <= areas.max(axis=0))
    previous = np.pad(small[:-1], ((1, 0), (0, 0)))
    subsequent = np.pad(small[1:], ((0, 1), (0, 0)))
    adjacent_absence = (areas == 0) & (previous | subsequent)
    return np.where(small, SMALL, np.where(adjacent_absence, ADJACENT_ABSENCE, OTHER))


def build_groups(cases, initial, anchors, view_summaries, truth=None):
    """Prepare identical candidate rosters for both feature sets in each phase."""
    roster = set(cases)
    if any(set(values) != roster for values in (initial, anchors, view_summaries)):
        raise ValueError("Images, initial indices, anchors, and evidence must align")
    if truth is not None and (set(truth) != roster or len(truth) != 6):
        raise ValueError("Selection-calibration truth must cover exactly six patients")

    groups = {phase: {} for phase in (0, 1)}
    class_count = len(CLASSES)
    for patient, case in sorted(cases.items()):
        image_features = block_features(case.images)
        evidence, areas = view_summaries[patient]
        if evidence.shape != (len(case.images), class_count) or areas.shape != evidence.shape:
            raise ValueError("Evidence and soft areas must align with slices and classes")
        if (not np.isfinite(evidence).all() or not np.isfinite(areas).all()
                or (evidence < 0).any() or (evidence > 1).any() or (areas < 0).any()):
            raise ValueError("Invalid teacher evidence or soft areas")
        for phase, by_patient in groups.items():
            indices = np.flatnonzero(case.phase == phase)
            indices = indices[np.argsort(case.slice_index[indices])]
            anchor_indices = np.flatnonzero(np.isin(indices, initial[patient]))
            if len(indices) < 2 or len(anchor_indices) != 1 or not np.all(np.diff(case.slice_index[indices]) == 1):
                raise ValueError("Expected a single anchor and consecutive native slices in each phase")
            center = int(anchor_indices[0])
            anchor_index = int(indices[center])
            anchor = anchors[patient][anchor_index]
            keep = np.arange(len(indices)) != center
            candidates = indices[keep]
            distance = np.square(image_features[candidates] - image_features[anchor_index]).sum(axis=1)
            novelty = distance / max(float(distance.max()), 1e-12)
            phase_evidence = np.clip(evidence[indices], 0.01, 0.99)
            phase_areas = areas[indices]
            relative_areas = phase_areas / np.maximum(phase_areas.max(axis=0), 1)
            axial = np.arange(len(indices)) - center
            axial = axial / max(center, len(indices) - center - 1)
            phase_one_hot = np.tile(np.eye(2)[phase], (len(indices), 1))
            base_columns, extra_columns = [], []
            for column, label in enumerate(CLASSES):
                observed_presence = float((anchor == label).any())
                scores = phase_evidence[:, column].copy()
                scores[center] = observed_presence
                predecessor = scores[np.arange(len(scores)) - np.sign(axial).astype(int)]
                local_boundary = (1.0 - scores) * predecessor
                class_one_hot = np.tile(np.eye(class_count)[column], (len(indices), 1))
                base_columns.append(np.column_stack((
                    phase_one_hot, class_one_hot, scores, relative_areas[:, column], axial,
                    local_boundary, np.full(len(indices), observed_presence),
                )))
                if observed_presence:
                    inside, boundary = endpoint_credit(scores, center, contiguous=True)
                else:
                    # Without a positive anchor, use the same local evidence in both models.
                    inside, boundary = scores, local_boundary
                extra_columns.append(np.column_stack((inside, boundary)))
            local = np.stack(base_columns, axis=1)[keep]
            anchored = np.concatenate((local, np.stack(extra_columns, axis=1)[keep]), axis=2)
            states = None if truth is None else reference_states(truth[patient].masks[indices])[keep]
            by_patient[patient] = Group(candidates, novelty, local, anchored, states)
    return groups


def fit(groups, *, contiguous):
    """Fit a ridge multinomial model with equal total weight per patient."""
    entries = [(patient, group) for by_patient in groups.values() for patient, group in by_patient.items()]
    patients = {patient for patient, _ in entries}
    if len(patients) != 6 or any(group.states is None for _, group in entries):
        raise ValueError("Fit requires the same six labeled selection-calibration patients")
    available_rows = Counter()
    for patient, group in entries:
        available_rows[patient] += group.states.size
    parts, labels, weights = [], [], []
    for patient, group in entries:
        features = group.anchored if contiguous else group.local
        parts.append(features.reshape(-1, features.shape[-1]))
        labels.append(group.states.ravel())
        weights.append(np.full(group.states.size, 1 / (len(patients) * available_rows[patient])))
    rows, states, weights = map(np.concatenate, (parts, labels, weights))
    observed = np.unique(states)
    targets = (states[:, None] == observed[None, :]).astype(np.float64)
    center = weights @ rows
    scale = np.sqrt(weights @ np.square(rows - center))
    scale[scale < 1e-6] = 1.0
    shape = (rows.shape[1] + 1, len(observed))
    coefficients = np.zeros(shape)
    calibration = Calibration(contiguous, observed, center, scale, coefficients)
    if len(observed) > 1:
        matrix = calibration.matrix(rows)
        ridge = 0.01

        def objective(flat):
            coefficients = flat.reshape(shape)
            logits = matrix @ coefficients
            logp = logits - logsumexp(logits, axis=1, keepdims=True)
            loss = -(weights @ (logp * targets).sum(axis=1)) + 0.5 * ridge * np.square(coefficients[1:]).sum()
            gradient = matrix.T @ ((np.exp(logp) - targets) * weights[:, None])
            gradient[1:] += ridge * coefficients[1:]
            return loss, gradient.ravel()

        result = minimize(objective, coefficients.ravel(), method="L-BFGS-B", jac=True)
        if not result.success:
            raise RuntimeError(f"Selection-calibration fit failed: {result.message}")
        coefficients = result.x.reshape(shape)
    return Calibration(contiguous, observed, center, scale, coefficients)


def select(groups, initial, model, *, contiguous, strength=1.0):
    """Allocate calibrated class-state credits with the shared greedy rule."""
    if model.contiguous != contiguous:
        raise ValueError("Calibrator and feature set differ")
    if set(groups) != {0, 1} or any(set(group) != set(initial) for group in groups.values()):
        raise ValueError("Candidate phases and initial roster differ")
    selected = {patient: list(indices) for patient, indices in sorted(initial.items())}
    ledger, nominations = {}, []
    for phase, by_patient in sorted(groups.items()):
        candidates = {}
        for patient, group in sorted(by_patient.items()):
            features = group.anchored if contiguous else group.local
            n_slices, n_classes, feature_count = features.shape
            probabilities = model.predict(features.reshape(-1, feature_count)).reshape(n_slices, n_classes, 3)
            states = probabilities[:, :, [SMALL, ADJACENT_ABSENCE]]
            credits = states.reshape(n_slices, 2 * n_classes)
            candidates[patient] = (group.indices, credits, group.novelty)
            for column, label in enumerate(CLASSES):
                absence = states[:, column, 1]
                best = int(absence.argmax())
                if absence[best] > 0:
                    nominations.append((patient, phase, label, int(group.indices[best]), float(absence[best])))
        ledger[phase] = greedy_queries(candidates, strength)
        for patient, (index, _) in ledger[phase].items():
            selected[patient].append(index)
    return {patient: sorted(indices) for patient, indices in selected.items()}, ledger, nominations
