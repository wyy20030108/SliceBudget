"""Convert the official CC0 Sunnybrook release to a positive-only cache.

Each input ZIP pair is the DICOM and contour release for one SourceForge part.
The source supplies LV endocardial polygons for selected ED/ES images. Slices
without a polygon are used for phase normalization but are never evaluation
examples. An optional secondary cache also includes ED LV myocardium derived
from matched epicardial and endocardial polygons.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen
from zipfile import ZipFile, is_zipfile

import numpy as np
from pydicom import dcmread
from pydicom.pixels import apply_modality_lut
from scipy.ndimage import zoom

from acdc_data import Case, _save_case, load_cases


SOURCE = "https://sourceforge.net/projects/cardiac-mr/files/"
MIRROR = "https://master.dl.sourceforge.net/project/cardiac-mr/"
PARTS = (1, 2, 3)
ARCHIVE_TYPES = ("DICOM", "Contours")
PATIENT = re.compile(r"SC-(?:HF-I|HF-NI|HYP|N)-\d+", re.IGNORECASE)
IMAGE = re.compile(r"(IM-\d+)-(\d+)\.dcm", re.IGNORECASE)
CONTOUR = re.compile(r"(IM-\d+)-(\d+)-icontour-manual\.txt", re.IGNORECASE)
EPICARDIUM = re.compile(r"(IM-\d+)-(\d+)-ocontour-manual\.txt", re.IGNORECASE)
EXPECTED_PATIENTS_PER_PART = 15
MYO_LABEL = 2
LV_LABEL = 3


def fetch(root: Path, kind: str, part: int) -> None:
    destination = root / f"{kind}Part{part}.zip"
    if destination.exists() and is_zipfile(destination):
        return
    filename = quote(f"Sunnybrook Cardiac MR Database {kind}Part{part}.zip")
    url = f"{MIRROR}{filename}"
    with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as output:
        temporary = Path(output.name)
        try:
            with urlopen(url, timeout=90) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.close()
            if not is_zipfile(temporary):
                raise ValueError(f"Sunnybrook archive download is not a ZIP: {url}")
            temporary.replace(destination)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def inventory(archive: ZipFile, pattern: re.Pattern) -> dict[str, dict[tuple[str, int], str]]:
    patients = defaultdict(dict)
    for name in archive.namelist():
        path = PurePosixPath(name)
        match = pattern.fullmatch(path.name)
        if "__MACOSX" in path.parts or match is None:
            continue
        patient = next((item.upper() for item in path.parts if PATIENT.fullmatch(item)), None)
        if patient is None:
            raise ValueError(f"No Sunnybrook patient folder for {name}")
        key = (match[1].upper(), int(match[2]))
        if key in patients[patient]:
            raise ValueError(f"Duplicate {patient} file {key} in {archive.filename}")
        patients[patient][key] = name
    return dict(patients)


def read_frame(archive: ZipFile, name: str):
    dataset = dcmread(io.BytesIO(archive.read(name)))
    image = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
    spacing = np.asarray(dataset.PixelSpacing, dtype=np.float32)
    if image.ndim != 2 or spacing.shape != (2,) or not np.isfinite(image).all():
        raise ValueError(f"Invalid Sunnybrook DICOM image in {name}")
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError(f"Invalid in-plane spacing in {name}")
    return image, spacing, int(dataset.CardiacNumberOfImages)


def rasterize(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize 1-based (column, row) contour coordinates at pixel centers."""
    height, width = shape
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Invalid LV endocardial polygon")
    lower = np.maximum(np.floor(points.min(axis=0)).astype(int), 1)
    upper = np.minimum(np.ceil(points.max(axis=0)).astype(int), (width, height))
    x, y = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
    inside = np.zeros(x.shape, dtype=bool)
    for (x0, y0), (x1, y1) in zip(points, np.roll(points, -1, axis=0)):
        if y0 != y1:
            crosses = (y0 > y) != (y1 > y)
            x_intercept = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < x_intercept)
    mask = np.zeros(shape, dtype=bool)
    mask[lower[1] - 1:upper[1], lower[0] - 1:upper[0]] = inside
    if not mask.any():
        raise ValueError("LV polygon contains no image pixel centers")
    return mask


def resize_phase(images, masks, spacing, size):
    std = images.std(dtype=np.float64)
    images = ((images - images.mean(dtype=np.float64)) / max(std, 1e-6)).astype(np.float32)
    _, height, width = images.shape
    scale = size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    factors = (1, new_height / height, new_width / width)
    if (new_height, new_width) != (height, width):
        images = zoom(images, factors, order=1, mode="nearest", grid_mode=True)
        masks = zoom(masks, factors, order=0, mode="nearest", grid_mode=True)
    pad_height = size - new_height
    pad_width = size - new_width
    padding = ((0, 0), (pad_height // 2, (pad_height + 1) // 2), (pad_width // 2, (pad_width + 1) // 2))
    spacing = (spacing / np.asarray(factors[1:], dtype=np.float32)).astype(np.float32)
    return np.pad(images, padding), np.pad(masks, padding), spacing


def prepare_case(patient, image_names, contour_names, dicom_archive, contour_archive, size, epicardial_names):
    # The challenge files preserve the IM ID of the original short-axis series.
    if missing := set(contour_names) - set(image_names):
        raise ValueError(f"{patient} has endocardial contours without matching DICOM: {sorted(missing)}")
    series = {key[0] for key in contour_names}
    if len(series) != 1:
        raise ValueError(f"{patient} manual annotations reference multiple DICOM series: {sorted(series)}")
    shape_image, _, phase_count = read_frame(dicom_archive, image_names[min(contour_names)])
    if phase_count < 2:
        raise ValueError(f"{patient} cine series has invalid phase count {phase_count}")

    # Official evaluation_code.zip defines image number = slice * phase_count
    # + phase + 1. Its ED/ES definitions use maximum/minimum positive LV volume.
    native_masks = {}
    areas = defaultdict(int)
    for (_, image_number), name in sorted(contour_names.items()):
        points = np.loadtxt(io.BytesIO(contour_archive.read(name)))
        slice_index, native_phase = divmod(image_number - 1, phase_count)
        mask = rasterize(points, shape_image.shape)
        native_masks[slice_index, native_phase] = mask.astype(np.uint8) * LV_LABEL
        areas[native_phase] += int(np.count_nonzero(mask))
    if len(areas) != 2 or len(set(areas.values())) != 2:
        raise ValueError(f"{patient} expected two distinct positive contour phases: {dict(areas)}")
    phase_order = sorted(areas, key=areas.__getitem__, reverse=True)
    if epicardial_names is not None:
        ed_contours = {key for key in contour_names if (key[1] - 1) % phase_count == phase_order[0]}
        if set(epicardial_names) != ed_contours:
            raise ValueError(f"{patient} epicardial contours do not cover exactly the identified ED phase")
        for (_, image_number), name in sorted(epicardial_names.items()):
            slice_index, native_phase = divmod(image_number - 1, phase_count)
            points = np.loadtxt(io.BytesIO(contour_archive.read(name)))
            outer = rasterize(points, shape_image.shape)
            target = native_masks[slice_index, native_phase]
            # Keep the annotated LV blood-pool boundary, including the papillary
            # regions. Define myocardium from the contour ring, without adding
            # auxiliary p1/p2 papillary contours to the ring.
            myocardium = outer & (target == 0)
            if not myocardium.any():
                raise ValueError(f"{patient} lacks epicardial ring pixels in {name}")
            target[myocardium] = MYO_LABEL

    phases = []
    for native_phase in phase_order:
        rows = []
        for (series_id, image_number), name in sorted(image_names.items()):
            native_index, frame = divmod(image_number - 1, phase_count)
            if series_id not in series or frame != native_phase:
                continue
            image, spacing, count = read_frame(dicom_archive, name)
            if image.shape != shape_image.shape or count != phase_count:
                raise ValueError(f"{patient} has inconsistent cine geometry in {name}")
            rows.append((native_index, image, spacing, native_masks.get((native_index, frame))))

        annotated = np.array([row[3] is not None for row in rows])
        native_indices = np.array([row[0] for row in rows], dtype=np.int32)
        volume = np.stack([row[1] for row in rows])
        spacing = np.stack([row[2] for row in rows])
        targets = np.stack([row[3] for row in rows if row[3] is not None])
        images, targets, spacing = resize_phase(volume, targets, spacing, size)
        positive = (targets == LV_LABEL).any(axis=(1, 2))
        validated = int(positive.sum())
        if validated == 0:
            raise ValueError(f"{patient} resize eliminated all positive masks in phase {native_phase + 1}")
        phases.append((images[annotated][positive], targets[positive], native_indices[annotated][positive],
                       spacing[annotated][positive], len(rows), int(annotated.sum())))

    case = Case(
        patient_id=patient,
        images=np.concatenate([row[0] for row in phases]),
        masks=np.concatenate([row[1] for row in phases]),
        phase=np.repeat(np.arange(2, dtype=np.int8), [len(row[0]) for row in phases]),
        slice_index=np.concatenate([row[2] for row in phases]),
        spacing_yx=np.concatenate([row[3] for row in phases]),
    )
    audit = {
        key: {"native_frame_one_based": native_phase + 1, "available_images": row[4],
              "explicit_positive_contours": row[5], "cached_positive_masks": len(row[0]),
              "cached_positive_myo_masks": int(((row[1] == MYO_LABEL).any(axis=(1, 2))).sum())}
        for key, native_phase, row in zip(("ED", "ES"), phase_order, phases, strict=True)
    }
    return case, audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as data:
        for block in iter(lambda: data.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: Path, output: Path, size: int, *, include_ed_myocardium: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    archives = {}
    for part in PARTS:
        dicom_path = root / f"DICOMPart{part}.zip"
        contour_path = root / f"ContoursPart{part}.zip"
        with ZipFile(dicom_path) as dicom, ZipFile(contour_path) as contour:
            images = inventory(dicom, IMAGE)
            outlines = inventory(contour, CONTOUR)
            epicardium = inventory(contour, EPICARDIUM) if include_ed_myocardium else {}
            if set(images) != set(outlines) or len(outlines) != EXPECTED_PATIENTS_PER_PART:
                raise ValueError(f"Unexpected release part {part}: {len(images)} DICOM and {len(outlines)} contour patients")
            if include_ed_myocardium and epicardium.keys() != outlines.keys():
                raise ValueError(f"Epicardial and endocardial patient rosters differ in release part {part}")
            for patient, contour_names in sorted(outlines.items()):
                if patient in reports:
                    raise ValueError(f"Patient appears in two source archives: {patient}")
                case, report = prepare_case(patient, images[patient], contour_names, dicom, contour, size,
                                            epicardium.get(patient))
                _save_case(output, case)
                reports[patient] = {"source_part": part, **report}
        archives[dicom_path.name] = sha256(dicom_path)
        archives[contour_path.name] = sha256(contour_path)
    if unexpected := {path.stem for path in output.glob("*.npz")} - reports.keys():
        raise ValueError(f"Unexpected patient cache files: {sorted(unexpected)}")
    scope = "positive LV endocardium at ED/ES only"
    if include_ed_myocardium:
        scope += ", plus positive ED LV myocardium from matched epicardial contours"
    manifest = {
        "source": SOURCE,
        "source_license": "CC0-1.0",
        "scope": f"{scope}; no negative slice ground truth",
        "include_ed_myocardium": include_ed_myocardium,
        "phase_assignment": "larger positive LV rasterized area = ED, smaller = ES",
        "patients": reports,
        "archive_sha256": archives,
        "size": size,
    }
    (output / "sunnybrook_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    cached = load_cases(output)
    print(f"Prepared {len(cached)} Sunnybrook patients and {sum(len(case.images) for case in cached.values())} positive LV slices.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--size", default=192, type=int)
    parser.add_argument("--download", action="store_true", help="Download the six official SourceForge ZIPs as needed")
    parser.add_argument("--include-ed-myocardium", action="store_true",
                        help="Also include the ED myocardium ring in a separate secondary cache")
    args = parser.parse_args()
    if args.size < 16:
        parser.error("Output size must be at least 16 pixels")
    if args.download:
        args.archives.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs = [workers.submit(fetch, args.archives, kind, part) for kind in ARCHIVE_TYPES for part in PARTS]
            for job in jobs:
                job.result()
    prepare(args.archives, args.out_root, args.size, include_ed_myocardium=args.include_ed_myocardium)


if __name__ == "__main__":
    main()
