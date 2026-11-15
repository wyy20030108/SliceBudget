"""Count optional blank-plane negatives with independently labeled cine-cycle support."""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from prepare_cmr_multi import CLASSES, LUT, file_sha256


def cycle_support(raw, cache, output):
    root = raw / "CINE_MULTI" if (raw / "CINE_MULTI").is_dir() else raw
    receipt_path = cache / "coverage.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rows, extra_all, extra_near = {}, [], []
    for case, info in receipt["cases"].items():
        mask_path = root / f"SAX_{info['release_split']}" / "anno" / f"{case.removeprefix('case-')}.nii.gz"
        if file_sha256(mask_path) != info["source_mask_sha256"]:
            raise ValueError(f"Source mask differs from preparation receipt: {case}")
        mask = np.asanyarray(nib.load(str(mask_path)).dataobj)
        frames = info["frames_in_cycle"]
        n_slices = info["slices_per_phase"]
        mask = LUT[mask.astype(np.uint8)].reshape(*mask.shape[:2], n_slices, frames)
        support_z_phase = (mask > 0).any(axis=(0, 1))
        with np.load(cache / f"{case}.npz", allow_pickle=False) as cached:
            areas = cached["native_areas"].reshape(2, n_slices, 3)
        additions, adjacents = [], []
        for per_phase, phase in zip(areas, info["ED_ES_frames"], strict=True):
            if not np.array_equal(per_phase.sum(axis=1) > 0, support_z_phase[:, phase]):
                raise ValueError(f"Native reference and cache coverage disagree: {case}")
            blank = ~support_z_phase[:, phase] & support_z_phase.any(axis=1)
            valid = blank[:, None] & (per_phase > 0).any(axis=0, keepdims=True)
            small = (per_phase > 0) & (4 * per_phase <= per_phase.max(axis=0))
            neighbor = np.zeros_like(small)
            neighbor[1:] |= small[:-1]
            neighbor[:-1] |= small[1:]
            additions.append(valid)
            adjacents.append(valid & neighbor)
        added = np.concatenate(additions)
        adjacent = np.concatenate(adjacents)
        rows[case] = added.tolist()
        extra_all.append(added.sum(axis=0))
        extra_near.append(adjacent.sum(axis=0))
    report = {"classes": CLASSES, "coverage_sha256": file_sha256(receipt_path),
              "rule": "Target present elsewhere in the same selected phase. All three structures absent on this "
                      "plane at this phase, and at least one independently labeled structure present on the "
                      "same native slice at another phase. Counts form a separate sensitivity analysis.",
              "case_order": "Per-case rows are ED native slices followed by ES native slices.",
              "additional_support": {}, "cycle_supported_blank": rows}
    for name, matrix in [("all", extra_all), ("adjacent", extra_near)]:
        values = np.asarray(matrix)
        report["additional_support"][name] = {"pairs": values.sum(axis=0).tolist(),
                                               "patients": (values > 0).sum(axis=0).tolist()}
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report["additional_support"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(cycle_support(args.raw, args.cache, args.output)))


if __name__ == "__main__":
    main()
