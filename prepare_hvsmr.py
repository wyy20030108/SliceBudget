"""Prepare the cropped HVSMR-2.0 release for a static axial MRI stress test.

The dataset has a dense interpretation of the eight released heart structures.
ACDC labels 1 and 3 correspond to the released RV and LV cavities, respectively.
ACDC label 2 is not annotated and must not be scored. Phase 0 is a storage code
for the sole static acquisition, with no inferred cardiac phase.
"""

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


PATTERN = re.compile(r"pat(\d+)_cropped\.nii\.gz")
SOURCE_DOI = "10.6084/m9.figshare.25226366.v2"
DATA_PAPER = "https://doi.org/10.1038/s41597-024-03469-9"
NATIVE_LABELS = (2, 1)
ACDC_COLUMNS = (0, 2)
TECHNICAL_CSV_GIT_BLOB = "caef45fbd73430cd3302acd0d004ddbcdc242d3f"


def file_sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def axial_inputs(image_path, mask_path, endpoints_path):
    source = nib.load(str(image_path))
    if len(source.shape) != 3 or min(source.shape) == 0:
        raise ValueError(f"Expected a nonempty 3D CMR volume: {image_path.name}")
    labels = []
    for path in (mask_path, endpoints_path):
        annotation = nib.load(str(path))
        if annotation.shape != source.shape or not np.allclose(annotation.affine, source.affine,
                                                               rtol=1e-5, atol=1e-3):
            raise ValueError(f"Image and annotation must share the voxel grid: {path.name}")
        labels.append(nib.as_closest_canonical(annotation))
    canonical = nib.as_closest_canonical(source)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise ValueError(f"Cannot identify an axial slice axis: {image_path.name}")
    # Use nearest axial voxel planes. Reorientation only permutes and flips axes.
    # No interpolation changes native masks before the presence ledger is made.
    obliquity = float(np.rad2deg(nib.affines.obliquity(canonical.affine)[2]))
    if not np.isfinite(obliquity) or obliquity > 15:
        raise ValueError(f"Slice normal is not close to the axial axis: {image_path.name}, {obliquity:.2f} degrees")
    image = canonical.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    mask, endpoints = (np.asarray(item.dataobj).transpose(2, 1, 0) for item in labels)
    if (not np.isfinite(image).all() or not np.isin(mask, np.arange(9)).all()
            or not np.isin(endpoints, (0, 3, 5, 6, 7, 8)).all()):
        raise ValueError(f"Unexpected intensities or annotation vocabulary: {image_path.name}")
    spacing = np.asarray(canonical.header.get_zooms()[:2][::-1], dtype=np.float32)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError(f"Nonpositive axial spacing: {image_path.name}")
    return image, mask, endpoints, spacing, obliquity


def resize_axial(images, masks, spacing, size):
    std = images.std(dtype=np.float64)
    images = ((images - images.mean(dtype=np.float64)) / max(std, 1e-6)).astype(np.float32)
    _, height, width = images.shape
    scale = size / max(height, width)
    new_height, new_width = (max(1, round(length * scale)) for length in (height, width))
    factors = (1., new_height / height, new_width / width)
    if (new_height, new_width) != (height, width):
        images = zoom(images, factors, order=1, mode="nearest", grid_mode=True)
        masks = zoom(masks, factors, order=0, mode="nearest", grid_mode=True)
    pad_y, pad_x = size - new_height, size - new_width
    padding = ((0, 0), (pad_y // 2, (pad_y + 1) // 2), (pad_x // 2, (pad_x + 1) // 2))
    return np.pad(images, padding), np.pad(masks, padding), (spacing / factors[1:]).astype(np.float32)


def convert(image_path, mask_path, endpoints_path, size):
    images, source_mask, endpoints, spacing, obliquity = axial_inputs(image_path, mask_path, endpoints_path)
    masks = np.zeros(source_mask.shape, dtype=np.uint8)
    native_areas = np.zeros((len(images), 3), dtype=np.int32)
    verified_absent = np.zeros_like(native_areas, dtype=bool)
    for native_label, column in zip(NATIVE_LABELS, ACDC_COLUMNS, strict=True):
        structure = source_mask == native_label
        areas = structure.sum(axis=(1, 2))
        masks[structure] = column + 1
        native_areas[:, column] = areas
        # An annotated structure on the same plane anchors the comparison in a
        # required heart region. Optional vessel tails alone cannot anchor it.
        other_required = ((source_mask > 0) & ~structure & (endpoints == 0)).any(axis=(1, 2))
        verified_absent[:, column] = (areas == 0) & other_required & (areas > 0).any()

    images, masks, spacing = resize_axial(images, masks, spacing, size)
    resized_presence = np.stack([np.any(masks == index, axis=(1, 2)) for index in (1, 2, 3)], axis=1)
    dropped = (native_areas > 0) & ~resized_presence
    result = {"images": images, "masks": masks, "phase": np.zeros(len(images), dtype=np.int8),
              "slice_index": np.arange(len(images), dtype=np.int32),
              "spacing_yx": np.tile(spacing, (len(images), 1)),
              "native_areas": native_areas, "verified_absent": verified_absent}
    counts = {"axial_slices": len(images), "obliquity_degrees": obliquity,
              "positive": (native_areas > 0).sum(axis=0).tolist(),
              "verified_absent": verified_absent.sum(axis=0).tolist(),
              "native_positive_lost_at_resize": dropped.sum(axis=0).tolist()}
    return result, counts


def coverage_description(cohort, provenance, size):
    counts = list(cohort.values())
    support = {}
    for metric in ("positive", "verified_absent", "native_positive_lost_at_resize"):
        values = np.asarray([case[metric] for case in counts])
        support[metric] = {"slices": values.sum(axis=0).tolist(),
                           "patients": (values > 0).sum(axis=0).tolist()}
    return {
        "title": "HVSMR-2.0 (cropped)", "source": f"https://doi.org/{SOURCE_DOI}",
        "provider_doi": SOURCE_DOI, "paper": DATA_PAPER, "site": "Boston Children's Hospital",
        "license": "CC BY 4.0", "variant": "cropped around the heart, provider intensities unnormalized",
        "test_scope": "static axial MRI with congenital heart disease; out-of-protocol stress test",
        "plane": "nearest axial voxel planes after canonical RAS reorientation, not cardiac short axis",
        "phase_scope": "static", "phase_zero": "static acquisition, no cine phase designation",
        "in_plane_size": size, "labels": [1, 3], "patients": len(counts),
        "axial_slices": sum(case["axial_slices"] for case in counts),
        "max_obliquity_degrees": max(case["obliquity_degrees"] for case in counts),
        "columns": ["RV", "MYO (unknown)", "LV"], "area_units": "native in-plane pixels",
        "negative_definition": "Target absent in native annotation, present elsewhere in this volume, and another"
                               " required annotated heart structure intersects this axial plane.",
        "reference_annotation": "All 60 volumes were expert-reviewed. The original 20 HVSMR cases used manual"
                                " whole-heart labeling; the subsequent 40 used semi-automatic labeling with review.",
        "input_provenance": {key: value for key, value in provenance.items() if key != "files"},
        "reference_support": support,
    }


def reference_groups(technical_csv):
    contents = technical_csv.read_bytes()
    prefix = f"blob {len(contents)}\0".encode("ascii")
    if hashlib.sha1(prefix + contents).hexdigest() != TECHNICAL_CSV_GIT_BLOB:
        raise ValueError("Technical CSV differs from the pinned mirror")
    rows = [row for row in csv.DictReader(io.StringIO(contents.decode("utf-8-sig"))) if any(row.values())]
    original = {int(row["Pat"]) for row in rows if row["HVSMR2016"] in {"train", "test"}}
    if (len(rows) != 60 or {int(row["Pat"]) for row in rows} != set(range(60)) or len(original) != 20
            or any(row["HVSMR2016"] not in {"train", "test", "no"} for row in rows)):
        raise ValueError("Expected 20 original HVSMR cases and 40 additional cases")
    return {f"hvsmr{number:02}": "manual_original" if number in original else "reviewed_semiautomatic"
            for number in range(60)}


def prepare(raw, output, size, technical_csv=None):
    groups = reference_groups(technical_csv) if technical_csv is not None else None
    candidates = {int(match[1]): path for path in raw.glob("*.nii.gz") if (match := PATTERN.fullmatch(path.name))}
    if set(candidates) != set(range(60)):
        raise ValueError("Expected exactly the 60 patients of the cropped HVSMR-2.0 release")
    paths = {number: (image, raw / f"pat{number}_cropped_seg.nii.gz",
                      raw / f"pat{number}_cropped_seg_endpoints.nii.gz") for number, image in candidates.items()}
    inputs = {path.name: path for triple in paths.values() for path in triple}
    manifest_path = raw / "provenance.json"
    provenance = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if provenance and provenance["provider_doi"] != SOURCE_DOI:
        raise ValueError("Source provenance must identify the cropped HVSMR-2.0 release")
    files = {name: {"size": path.stat().st_size, "sha256": file_sha256(path)} for name, path in sorted(inputs.items())}
    if provenance:
        expected = {row["name"]: {"size": row["size"], "sha256": row["sha256"]} for row in provenance["files"]}
        if files != expected:
            raise ValueError("Cropped NIfTI files differ from the verified origin inventory")

    output.mkdir(parents=True, exist_ok=True)
    intended = {f"hvsmr{number:02}" for number in candidates}
    if extras := {path.stem for path in output.glob("*.npz")} - intended:
        raise ValueError(f"Remove unrelated case files from the output directory: {sorted(extras)}")
    cohort = {}
    for number, (image, mask, endpoints) in sorted(paths.items()):
        patient = f"hvsmr{number:02}"
        arrays, counts = convert(image, mask, endpoints, size)
        with NamedTemporaryFile(dir=output, suffix=".npz.tmp", delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            with staged.open("wb") as stream:
                np.savez_compressed(stream, patient_id=patient, **arrays)
            staged.replace(output / f"{patient}.npz")
        finally:
            staged.unlink(missing_ok=True)
        cohort[patient] = counts
        print(f"Prepared {len(cohort)}/{len(candidates)} static axial volumes", flush=True)

    summary = coverage_description(cohort, provenance, size)
    if groups is not None:
        summary["input_provenance"]["group_table_sha256"] = file_sha256(technical_csv)
        (output / "reference_groups.json").write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")
    (output / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--technical-csv", type=Path, help="Write private case-to-annotation-group mapping")
    args = parser.parse_args()
    if not 16 <= args.size <= 512 or args.raw.resolve() == args.output.resolve():
        parser.error("Use distinct raw and output directories and a size from 16 to 512 pixels")
    prepare(args.raw, args.output, args.size, args.technical_csv)


if __name__ == "__main__":
    main()
