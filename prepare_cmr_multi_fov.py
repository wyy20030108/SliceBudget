"""Reproduce a physical-spacing sensitivity cache without changing reference eligibility."""

import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from prepare_cmr_multi import LUT, file_sha256, write_npz


def source_spacing(source):
    files = sorted(source.glob("patient*.npz"))
    if len(files) != 100:
        raise ValueError("Use the 100-case ACDC source training-release cache")
    values = {}
    for file in files:
        with np.load(file, allow_pickle=False) as case:
            spacings = case["spacing_yx"][case["phase"] == 0]
            if not len(spacings) or not np.allclose(spacings[0], spacings):
                raise ValueError(f"Missing or inconsistent source ED spacing: {file.name}")
            values[file.stem] = spacings[0].tolist()
    vectors = np.asarray(list(values.values()))
    if not np.isfinite(vectors).all() or (vectors <= 0).any():
        raise ValueError("Invalid source pixel spacing")
    signature = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return np.median(vectors, axis=0), signature


def physical_canvas(image, mask, native_spacing, target_spacing, size):
    image = ((image - image.mean(dtype=np.float64)) /
             max(image.std(dtype=np.float64), 1e-6)).astype(np.float32)
    shape = np.asarray(image.shape[1:])
    dimensions = np.maximum(1, np.rint(shape * native_spacing / target_spacing).astype(int))
    if (dimensions > size).any():
        raise ValueError("Canvas cannot fit the native field of view without cropping")
    factors = (1., *(dimensions / shape))
    if not np.array_equal(dimensions, shape):
        image = zoom(image, factors, order=1, mode="nearest", grid_mode=True)
        mask = zoom(mask, factors, order=0, mode="nearest", grid_mode=True)
    padding = ((0, 0), *((int(pad // 2), int((pad + 1) // 2)) for pad in size - dimensions))
    return np.pad(image, padding), np.pad(mask, padding), (native_spacing / factors[1:]).astype(np.float32)


def case_arrays(root, primary, patient, case, target_spacing, size):
    filename = f"{patient.removeprefix('case-')}.nii.gz"
    folder = root / f"SAX_{case['release_split']}"
    paths = (folder / "image" / filename, folder / "anno" / filename)
    for path, key in zip(paths, ("image_sha256", "source_mask_sha256"), strict=True):
        if file_sha256(path) != case[key]:
            raise ValueError(f"Source file differs from the primary receipt: {filename}")
    input_cache = primary / f"{patient}.npz"
    if file_sha256(input_cache) != case["cache_sha256"]:
        raise ValueError(f"Primary cache differs from the primary receipt: {patient}")
    with np.load(input_cache, allow_pickle=False) as original:
        arrays = {key: original[key] for key in ("native_areas", "verified_absent", "phase", "slice_index")}
    scan, reference = (nib.load(path) for path in paths)
    pixels = scan.get_fdata(dtype=np.float32)
    labels = np.asanyarray(reference.dataobj).astype(np.uint8)
    frames = case["frames_in_cycle"]
    native_spacing = np.asarray(scan.header.get_zooms()[:2][::-1], dtype=np.float64)
    images, masks, spacings = [], [], []
    for phase, frame in enumerate(case["ED_ES_frames"]):
        image = pixels[:, :, frame::frames].transpose(2, 1, 0)
        mask = LUT[labels[:, :, frame::frames]].transpose(2, 1, 0)
        raw_areas = np.stack([(mask == index).sum(axis=(1, 2)) for index in (1, 2, 3)], axis=1)
        if not np.array_equal(raw_areas, arrays["native_areas"][arrays["phase"] == phase]):
            raise ValueError(f"Native reference support changed under sensitivity preparation: {patient}")
        image, mask, spacing = physical_canvas(image, mask, native_spacing, target_spacing, size)
        images.append(image)
        masks.append(mask)
        spacings.append(np.tile(spacing, (len(image), 1)))
    arrays.update(images=np.concatenate(images), masks=np.concatenate(masks), spacing_yx=np.concatenate(spacings))
    transformed = np.stack([(arrays["masks"] == index).any(axis=(1, 2)) for index in (1, 2, 3)], axis=1)
    lost = (arrays["native_areas"] > 0) & ~transformed
    return arrays, lost.sum(axis=0).tolist()


def prepare(raw, source, primary, output):
    root = raw / "CINE_MULTI" if (raw / "CINE_MULTI").is_dir() else raw
    receipt = primary / "coverage.json"
    report = json.loads(receipt.read_text(encoding="utf-8"))
    size = report["plane_shape"][0]
    if report["plane_shape"] != [192, 192] or report["patients"] != 113:
        raise ValueError("Use the locked 113-patient primary cine cache")
    target_spacing, signature = source_spacing(source)
    output.mkdir(parents=True, exist_ok=True)
    if {path.stem for path in output.glob("*.npz")} - set(report["cases"]):
        raise ValueError("Output directory contains unrelated cached patients")
    losses = []
    for patient, case in report["cases"].items():
        arrays, lost = case_arrays(root, primary, patient, case, target_spacing, size)
        write_npz(output, patient, arrays)
        case["primary_cache_sha256"] = case["cache_sha256"]
        case["cache_sha256"] = file_sha256(output / f"{patient}.npz")
        case["native_positive_lost_at_resize"] = lost
        losses.append(lost)
        print(f"Prepared {len(losses)}/{report['patients']} source-scale cases", flush=True)
    losses = np.asarray(losses)
    report["support"]["native_positive_lost_at_resize"] = {
        "slices": losses.sum(axis=0).tolist(), "patients": (losses > 0).sum(axis=0).tolist()}
    report["geometry_sensitivity"] = {
        "scope": "Exploratory physical-spacing sensitivity following the native-grid transfer result.",
        "input_rule": "Native frame-wise z-score, physical resampling, and zero center-padding to 192 square.",
        "orientation": "Preserve native in-plane order and the frozen selection of frames and planes.",
        "spacing_yx_mm": target_spacing.tolist(),
        "source_spacing_rule": "Median of one ED model-input spacing vector per patient in the 100-case ACDC source cache.",
        "source_spacing_vectors_sha256": signature,
        "primary_coverage_sha256": file_sha256(receipt),
    }
    (output / "coverage.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "source", "primary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if len({getattr(args, name).resolve() for name in ("raw", "source", "primary", "output")}) != 4:
        parser.error("Choose four distinct data directories")
    prepare(args.raw, args.source, args.primary, args.output)


if __name__ == "__main__":
    main()
