"""Patient-level outcomes for small foreground and adjacent true absence."""

import numpy as np

from experiment import scores


def rate(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def patient_metrics(case, predicted):
    overall = scores(case, predicted)
    false_positive = borderline_count = absent_count = absent_fp = 0
    for phase in (0, 1):
        in_phase = case.phase == phase
        order = case.slice_index[in_phase].astype(int)
        reference = case.masks[in_phase]
        prediction = predicted[in_phase]
        for label in (1, 2, 3):
            areas = np.count_nonzero(reference == label, axis=(1, 2))
            small = (areas > 0) & (areas * 4 <= areas.max())
            absent = areas == 0
            positive_prediction = np.any(prediction == label, axis=(1, 2))
            nearby = np.isin(order - 1, order[small]) | np.isin(order + 1, order[small])
            border = absent & nearby
            false_positive += int((border & positive_prediction).sum())
            borderline_count += int(border.sum())
            absent_count += int(absent.sum())
            absent_fp += int((absent & positive_prediction).sum())
    return {"global_dice": overall["global_dice"], "small_dice": overall["hard_dice"],
            "small_miss_rate": rate(overall["hard_missed"], overall["hard_slice_classes"]),
            "bordering_absent_false_positive_rate": rate(false_positive, borderline_count),
            "all_absent_false_positive_rate": rate(absent_fp, absent_count),
            "small_positive_count": overall["hard_slice_classes"], "bordering_absent_count": borderline_count,
            "all_absent_count": absent_count}
