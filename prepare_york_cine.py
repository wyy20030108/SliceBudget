"""Prepare the official York pediatric cine-MRI data for a frozen external audit.

York provides LV cavity and myocardium contours only on slices where both
boundaries were visible. Pick the phase extrema from a uniform slice subset
and retain only explicitly paired contours for scoring. Use the external
cohort for positive-region comparisons; it has no verified negative slices
or RV reference. Cite Andreopoulos and Tsotsos, doi:10.1016/j.media.2007.12.003.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import loadmat
from scipy.ndimage import zoom

from acdc_data import Case, _save_case


ORIGIN = "https://data.nvision.eecs.yorku.ca/datasets/MRI"
FILES = {
    "mrimages.tar.gz": "80d9fd3132c4668ea4859e19acd471eb024019f0fed4060d89d695c78200216d",
    "manual_seg.zip": "26d355ae2f5ae40c49b985c5ee5d7201d3a498cc3587f7c20532e06976ed244e",
    "metadata.txt": "05abea97091bd6e811bd9c9697fd7224cb4eb6cd6cc00ac6568773eb55cb37f1",
}
SUBJECTS = set(range(1, 34))
PHASES = 20


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def official_files(directory: Path, download: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = directory / name
        if not path.is_file() and download:
            # An interrupted transfer leaves a distinct file until verified.
            temporary = path.with_name(f"{name}.part")
            digest = hashlib.sha256()
            with urlopen(f"{ORIGIN}/{name}", timeout=120) as response, temporary.open("wb") as stream:
                while data := response.read(1024 * 1024):
                    stream.write(data)
                    digest.update(data)
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded checksum differs for {name}")
            temporary.replace(path)
        elif not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Fetch it from {ORIGIN}/ or use --download.")
        elif file_hash(path) != expected:
            raise ValueError(f"Local checksum differs for {path}")


def read_spacing(path: Path) -> dict[int, tuple[float, float]]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines()[1:]:
        if not line.strip():
            continue
        subject, in_plane, between_slices, *_ = line.split()
        result[int(subject)] = float(in_plane), float(between_slices)
    if set(result) != SUBJECTS or not all(np.isfinite(values).all() and min(values) > 0 for values in result.values()):
        raise ValueError("Unexpected York metadata coverage or spacing")
    return result


def area(points: np.ndarray) -> float:
    x, y = points.T
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def endpoints(cells: np.ndarray) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    if cells.ndim != 2 or cells.shape[1] != PHASES:
        raise ValueError(f"Unexpected contour grid {cells.shape}")
    cavity_area = np.full(cells.shape, np.nan)
    for index, cell in np.ndenumerate(cells):
        if cell.size == 1 and cell.item() == -99999:
            continue
        if cell.shape != (65, 2) or not (cell[32] == 0).all() or not np.isfinite(cell).all():
            raise ValueError(f"Unexpected contour cell at {index}")
        cavity_area[index] = area(cell[:32])

    present = np.isfinite(cavity_area)
    if np.any(cavity_area[present] <= 0):
        raise ValueError("A documented cavity contour has nonpositive area")
    common_slices = present.all(axis=1)
    if not common_slices.any():
        raise ValueError("No slice has cavity contours across the entire cycle")
    volume_proxy = cavity_area[common_slices].sum(axis=0)
    ed, es = int(volume_proxy.argmax()), int(volume_proxy.argmin())
    if ed == es:
        raise ValueError("The cavity area proxy does not distinguish the phases")
    return (ed, es), present, common_slices


def rasterize(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    # Follow the release's MATLAB interpolation and x/y convention before
    # even-odd polygon filling at one-based pixel centers.
    closed = np.concatenate((points, points[:1]))
    sampled = CubicSpline(np.linspace(0, 1, len(closed)), closed)(np.linspace(0, 1, 1000))
    x0, y0 = sampled[:-1].T
    x1, y1 = sampled[1:].T
    mask = np.zeros(shape, dtype=bool)
    columns = np.arange(1, shape[1] + 1)
    for row in range(max(1, int(np.ceil(y0.min()))), min(shape[0], int(np.floor(y0.max()))) + 1):
        crosses = (y0 > row) != (y1 > row)
        boundaries = x0[crosses] + (row - y0[crosses]) * (x1[crosses] - x0[crosses]) / (y1[crosses] - y0[crosses])
        mask[row - 1] = np.searchsorted(np.sort(boundaries), columns, side="right") % 2 == 1
    return mask


def convert(
    subject: int, image_grid: np.ndarray, contour_grid: np.ndarray,
    spacing: tuple[float, float], size: int,
) -> tuple[Case, dict]:
    if image_grid.shape[:2] != (256, 256) or image_grid.shape[2:] != contour_grid.shape:
        raise ValueError(f"York subject {subject} has inconsistent image/contour shapes")
    phase_ids, present, common = endpoints(contour_grid)
    images, masks, phase, indexes = [], [], [], []
    nonnested_pixels = 0
    factors = (1, size / 256, size / 256)
    for name, time_index in enumerate(phase_ids):
        full = image_grid[..., time_index].transpose(2, 0, 1).astype(np.float32)
        if not np.isfinite(full).all():
            raise ValueError(f"Nonfinite York intensities for subject {subject}")
        full = ((full - full.mean(dtype=np.float64)) / max(full.std(dtype=np.float64), 1e-6)).astype(np.float32)
        native = np.flatnonzero(present[:, time_index])
        selected = full[native]
        target = np.empty(selected.shape, dtype=np.uint8)
        for row, slice_index in enumerate(native):
            contour = contour_grid[slice_index, time_index]
            inner = rasterize(contour[:32], full.shape[-2:])
            outer = rasterize(contour[33:], full.shape[-2:])
            nonnested_pixels += int(np.count_nonzero(inner & ~outer))
            # Preserve the supplied endocardial delineation at crossings.
            target[row] = np.where(inner, 3, np.where(outer, 2, 0))
        if size != 256:
            selected = zoom(selected, factors, order=1, mode="nearest", grid_mode=True)
            target = zoom(target, factors, order=0, mode="nearest", grid_mode=True)
        if not np.logical_and(np.any(target == 2, axis=(1, 2)), np.any(target == 3, axis=(1, 2))).all():
            raise ValueError(f"A rasterized York target lost a reference class for subject {subject}")
        images.append(selected)
        masks.append(target)
        phase.append(np.full(len(native), name, dtype=np.int8))
        indexes.append(native.astype(np.int32))

    case = Case(
        f"york{subject:02d}", np.concatenate(images), np.concatenate(masks),
        np.concatenate(phase), np.concatenate(indexes),
        np.full((sum(map(len, phase)), 2), spacing[0] / factors[1], dtype=np.float32),
    )
    coverage = {
        "patient_id": case.patient_id,
        "ed_frame_one_based": phase_ids[0] + 1,
        "es_frame_one_based": phase_ids[1] + 1,
        "uniformly_contoured_slices": int(common.sum()),
        "ed_annotated_slices": len(phase[0]),
        "es_annotated_slices": len(phase[1]),
        "endocardial_pixels_outside_epicardial_polygon": nonnested_pixels,
    }
    return case, coverage


def prepare(raw: Path, output: Path, size: int, download: bool) -> None:
    if not 16 <= size <= 512:
        raise ValueError("Output size must be 16–512 pixels")
    official_files(raw, download)
    spacing = read_spacing(raw / "metadata.txt")
    output.mkdir(parents=True, exist_ok=True)
    expected = {f"york{subject:02d}.npz" for subject in SUBJECTS}
    if unexpected := {p.name for p in output.glob("*.npz")} - expected:
        raise ValueError(f"Unexpected cache files in {output}: {sorted(unexpected)}")
    coverage = []
    seen = set()
    with ZipFile(raw / "manual_seg.zip") as contours, tarfile.open(raw / "mrimages.tar.gz", "r|gz") as images:
        for item in images:
            match = re.fullmatch(r"sol_yxzt_pat(\d+)\.mat", item.name)
            if not item.isfile() or match is None:
                continue
            subject = int(match[1])
            if subject not in SUBJECTS or subject in seen:
                raise ValueError(f"Unexpected or duplicate subject {subject}")
            seen.add(subject)
            image = loadmat(io.BytesIO(images.extractfile(item).read()), variable_names=["sol_yxzt"])["sol_yxzt"]
            entry = contours.read(f"manual_seg/manual_seg_32points_pat{subject}.mat")
            contour = loadmat(io.BytesIO(entry), variable_names=["manual_seg_32points"])["manual_seg_32points"]
            case, summary = convert(subject, image, contour, spacing[subject], size)
            _save_case(output, case)
            coverage.append(summary)
    if seen != SUBJECTS:
        raise ValueError(f"Missing subjects: {sorted(SUBJECTS - seen)}")
    (output / "york_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(coverage)} York subjects and {sum(r['ed_annotated_slices'] + r['es_annotated_slices'] for r in coverage)} annotated slices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--size", type=int, default=192, metavar="PIXELS")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    prepare(args.raw, args.cache, args.size, args.download)
