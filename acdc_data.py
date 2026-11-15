"""Prepare labeled ACDC cine-MRI frames for patient-level annotation replay.

The official training subset is used when the archive has a training folder.
Native short-axis slice order is retained. Each 3D frame is z-scored, then each
slice is resized without stretching its array aspect ratio and center-padded
to a square. Intensity interpolation is linear and masks use nearest neighbor.
The official mask IDs are retained (0 background, 1 RV, 2 myocardium, 3 LV).
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


PATIENT_PATTERN = re.compile(r"patient\d+", re.IGNORECASE)
FRAME_PATTERN = re.compile(r"(patient\d+)_frame(\d+)(_gt)?\.nii(?:\.gz)?", re.IGNORECASE)
TRAIN_DIRECTORIES = {"train", "training"}
TEST_DIRECTORIES = {"test", "testing"}


@dataclass(frozen=True)
class Case:
    patient_id: str
    images: np.ndarray
    masks: np.ndarray
    phase: np.ndarray  # 0 = end-diastole, 1 = end-systole
    slice_index: np.ndarray  # Native short-axis index, not a universal apex marker
    spacing_yx: np.ndarray | None = None  # Transformed in-plane spacing, mm


def _patient_directories(raw_root: Path) -> list[Path]:
    raw_root = raw_root.resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"ACDC directory does not exist: {raw_root}")
    if raw_root.name.lower() in TEST_DIRECTORIES or (
        PATIENT_PATTERN.fullmatch(raw_root.name) and raw_root.parent.name.lower() in TEST_DIRECTORIES
    ):
        raise ValueError(f"Use the official training folder instead of {raw_root}")
    if PATIENT_PATTERN.fullmatch(raw_root.name):
        return [raw_root]

    candidates = []
    training_candidates = []
    for path in raw_root.rglob("patient*"):
        if not PATIENT_PATTERN.fullmatch(path.name) or not path.is_dir():
            continue
        parents = {part.lower() for part in path.relative_to(raw_root).parts[:-1]}
        if parents & TEST_DIRECTORIES:
            continue
        candidates.append(path)
        if parents & TRAIN_DIRECTORIES:
            training_candidates.append(path)
    patients = sorted(training_candidates or candidates)
    if not patients:
        raise FileNotFoundError(f"No ACDC patient directories found in {raw_root}")
    ids = [path.name.lower() for path in patients]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate patient IDs under {raw_root}; choose a single ACDC release")
    return patients


def _frame_pairs(patient_dir: Path) -> dict[int, tuple[Path, Path]]:
    images = {}
    masks = {}
    for path in patient_dir.iterdir():
        match = FRAME_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file() or match[1].lower() != patient_dir.name.lower():
            continue
        frame = int(match[2])
        files = masks if match[3] else images
        if frame in files:
            raise ValueError(f"Ambiguous frame {frame} files in {patient_dir}")
        files[frame] = path

    missing = masks.keys() - images.keys()
    if missing:
        raise ValueError(f"Missing frame images in {patient_dir}: {sorted(missing)}")
    if len(masks) < 2:
        raise ValueError(f"Expected at least two labeled frames in {patient_dir}")
    return {frame: (images[frame], mask) for frame, mask in masks.items()}


def _metadata_phases(patient_dir: Path) -> dict[str, int]:
    config = next((path for path in patient_dir.iterdir() if path.name.lower() == "info.cfg"), None)
    if config is None:
        return {}
    phases = {}
    for line in config.read_text(encoding="utf-8-sig").splitlines():
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key in ("ED", "ES"):
            frame = int(value.strip())
            if key in phases and phases[key] != frame:
                raise ValueError(f"Conflicting {key} definitions in {config}")
            phases[key] = frame
    return phases


def _phase_frames(patient_dir: Path, pairs: dict[int, tuple[Path, Path]]) -> list[int]:
    phases = _metadata_phases(patient_dir)
    if any(frame not in pairs for frame in phases.values()):
        raise ValueError(f"Info.cfg identifies a frame without a labeled pair in {patient_dir}")
    if len(phases) == 2:
        if phases["ED"] == phases["ES"]:
            raise ValueError(f"Identical ED and ES frames in {patient_dir}")
        return [phases["ED"], phases["ES"]]

    if len(pairs) != 2:
        raise ValueError(f"Need Info.cfg ED and ES to resolve {len(pairs)} labeled frames in {patient_dir}")
    if phases:
        known_phase, known_frame = next(iter(phases.items()))
        other_frame = next(frame for frame in pairs if frame != known_frame)
        return [known_frame, other_frame] if known_phase == "ED" else [other_frame, known_frame]

    import nibabel as nib

    # Bare NIfTI exports may omit Info.cfg. Infer the phases only for the two
    # annotated frames, using physical LV cavity volume (class 3).
    volumes = {}
    for frame, (_, mask_path) in pairs.items():
        mask = nib.load(str(mask_path))
        volume_per_voxel = abs(np.linalg.det(mask.affine[:3, :3]))
        volumes[frame] = np.count_nonzero(np.asarray(mask.dataobj) == 3) * volume_per_voxel
    ed, es = sorted(volumes, key=volumes.__getitem__, reverse=True)
    if (
        not np.isfinite([volumes[ed], volumes[es]]).all()
        or volumes[es] <= 0
        or np.isclose(volumes[ed], volumes[es])
    ):
        raise ValueError(f"Cannot infer ED and ES from LV cavity volumes; supply Info.cfg for {patient_dir}")
    warnings.warn(
        f"{patient_dir.name}: Info.cfg lacks phase IDs. Inferred ED={ed}, ES={es} from LV cavity volumes.",
        stacklevel=2,
    )
    return [ed, es]


def _read_frame(image_path: Path, mask_path: Path, size: int) -> tuple[np.ndarray, ...]:
    import nibabel as nib
    from scipy.ndimage import zoom

    image_file = nib.load(str(image_path))
    mask_file = nib.load(str(mask_path))
    if len(image_file.shape) != 3 or image_file.shape != mask_file.shape or min(image_file.shape) < 1:
        raise ValueError(f"Expected matching nonempty 3D image and mask: {image_path} / {mask_path}")
    if not np.allclose(image_file.affine, mask_file.affine, rtol=1e-5, atol=1e-3):
        raise ValueError(f"Image and mask use different voxel grids: {image_path} / {mask_path}")

    # ACDC volumes use the third array axis for the stack. Moving Z first
    # yields (slice, Y, X) while preserving all native slice indices.
    image = image_file.get_fdata(dtype=np.float32).transpose(2, 1, 0)
    mask = np.asarray(mask_file.dataobj).transpose(2, 1, 0)
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite intensities in {image_path}")
    if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 3 or not np.equal(mask, np.rint(mask)).all():
        raise ValueError(f"Unexpected ACDC mask values in {mask_path}")
    mask = mask.astype(np.uint8, copy=False)

    mean = image.mean(dtype=np.float64)
    std = image.std(dtype=np.float64)
    image = ((image - mean) / max(std, 1e-6)).astype(np.float32, copy=False)
    _, height, width = image.shape
    scale = size / max(height, width)
    resized_h = max(1, round(height * scale))
    resized_w = max(1, round(width * scale))
    factors = (1, resized_h / height, resized_w / width)
    if (resized_h, resized_w) != (height, width):
        image = zoom(image, factors, order=1, mode="nearest", grid_mode=True)
        mask = zoom(mask, factors, order=0, mode="nearest", grid_mode=True)

    pad_h = size - resized_h
    pad_w = size - resized_w
    padding = ((0, 0), (pad_h // 2, (pad_h + 1) // 2), (pad_w // 2, (pad_w + 1) // 2))
    voxel_x, voxel_y = map(float, image_file.header.get_zooms()[:2])
    spacing = np.array([voxel_y / factors[1], voxel_x / factors[2]], dtype=np.float32)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"Invalid voxel spacing in {image_path}")
    return np.pad(image, padding), np.pad(mask, padding), spacing


def _prepare_case(patient_dir: Path, size: int) -> Case:
    pairs = _frame_pairs(patient_dir)
    phases = [_read_frame(*pairs[frame], size) for frame in _phase_frames(patient_dir, pairs)]
    counts = [image.shape[0] for image, _, _ in phases]
    return Case(
        patient_id=patient_dir.name.lower(),
        images=np.concatenate([image for image, _, _ in phases]),
        masks=np.concatenate([mask for _, mask, _ in phases]),
        phase=np.repeat(np.arange(2, dtype=np.int8), counts),
        slice_index=np.concatenate([np.arange(count, dtype=np.int32) for count in counts]),
        spacing_yx=np.concatenate([np.tile(spacing, (count, 1)) for (_, _, spacing), count in zip(phases, counts)]),
    )


def _save_case(out_root: Path, case: Case) -> None:
    target = out_root / f"{case.patient_id}.npz"
    with tempfile.NamedTemporaryFile(dir=out_root, suffix=".tmp", delete=False) as file:
        temp_path = Path(file.name)
    try:
        with temp_path.open("wb") as file:
            np.savez_compressed(
                file,
                patient_id=case.patient_id,
                images=case.images,
                masks=case.masks,
                phase=case.phase,
                slice_index=case.slice_index,
                spacing_yx=case.spacing_yx,
            )
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def prepare(raw_root: Path, out_root: Path, size: int = 192) -> dict[str, Case]:
    """Convert ACDC patient folders to NPZ files and return the prepared cases."""
    if size < 16:
        raise ValueError("Output size must be at least 16 pixels")
    patients = _patient_directories(Path(raw_root))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    intended = {path.name.lower() for path in patients}
    unexpected = {path.stem for path in out_root.glob("*.npz")} - intended
    if unexpected:
        raise ValueError(f"Unexpected cached patients in {out_root}: {sorted(unexpected)}")

    cases = {}
    for patient_dir in patients:
        case = _prepare_case(patient_dir, size)
        _save_case(out_root, case)
        cases[case.patient_id] = case
    return cases


def _load_case(path: Path, *, require_both_phases: bool = True) -> Case:
    required = {"images", "masks", "phase", "slice_index"}
    with np.load(path, allow_pickle=False) as data:
        if missing := required - set(data.files):
            raise ValueError(f"Missing arrays: {sorted(missing)}")
        patient_id = str(data["patient_id"].item()) if "patient_id" in data.files else path.stem
        if patient_id != path.stem:
            raise ValueError(f"Patient ID {patient_id!r} differs from the filename")
        images = data["images"]
        masks = data["masks"]
        phase = data["phase"]
        index = data["slice_index"]
        spacing = data["spacing_yx"] if "spacing_yx" in data.files else None

    if images.ndim != 3 or 0 in images.shape or masks.shape != images.shape:
        raise ValueError("Expected same-shaped, nonempty (slice, Y, X) images and masks")
    if images.dtype.kind != "f" or not np.isfinite(images).all() or np.max(np.abs(images)) > np.finfo(np.float32).max:
        raise ValueError("Images must be finite floating-point arrays")
    if masks.dtype.kind not in "ui" or masks.min() < 0 or masks.max() > 3:
        raise ValueError("Masks must have integer labels 0–3")
    valid_phases = set(np.unique(phase))
    if (phase.shape != (len(images),) or phase.dtype.kind not in "ui" or not valid_phases <= {0, 1}
            or (require_both_phases and valid_phases != {0, 1})):
        raise ValueError("Phase must identify both ED (0) and ES (1)")
    if (
        index.shape != phase.shape
        or index.dtype.kind not in "ui"
        or index.min() < 0
        or index.max() > np.iinfo(np.int32).max
    ):
        raise ValueError("Expected nonnegative integer slice indices")
    if len(set(zip(phase.tolist(), index.tolist()))) != len(images):
        raise ValueError("Duplicate phase/slice indices")
    if spacing is not None and (
        spacing.shape != (len(images), 2)
        or spacing.dtype.kind not in "fui"
        or not np.isfinite(spacing).all()
        or np.any(spacing <= 0)
    ):
        raise ValueError("Expected positive finite (slice, Y/X) spacing")

    return Case(
        patient_id,
        images.astype(np.float32, copy=False),
        masks.astype(np.uint8, copy=False),
        phase.astype(np.int8, copy=False),
        index.astype(np.int32, copy=False),
        None if spacing is None else spacing.astype(np.float32, copy=False),
    )


def load_cases(data_dir: Path) -> dict[str, Case]:
    """Read and validate a directory of same-size patient NPZ files."""
    cases = {}
    shape = None
    for path in sorted(Path(data_dir).glob("*.npz")):
        try:
            case = _load_case(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid ACDC cache file {path}: {exc}") from exc
        if shape is not None and case.images.shape[1:] != shape:
            raise ValueError(f"In-plane shape {case.images.shape[1:]} in {path} differs from {shape}")
        shape = case.images.shape[1:]
        cases[case.patient_id] = case
    if not cases:
        raise FileNotFoundError(f"No patient NPZ files in {data_dir}")
    return cases


def make_protocol(cases: Mapping[str, Case], seed: int, *, use_proxy: bool = False) -> dict[str, list[str]]:
    """Create deterministic, disjoint roles from the official training cohort.

    With 100 patients, full replay leaves 40 cases in the formal selection pool.
    The separate proxy diagnostic protocol holds out extra development cases
    and must be selected before observing results.
    """
    patient_ids = sorted(cases)
    if len(patient_ids) < 89:
        raise ValueError(f"Need at least 89 ACDC training patients for the requested roles, found {len(patient_ids)}")
    shuffled = np.random.default_rng(seed).permutation(patient_ids).tolist()
    protocol = {"test": shuffled[:20], "initial": shuffled[20:27]}
    role_counts = (16, 9, 16, 8, 8) if use_proxy else (12, 7, 0, 7, 7)
    names = ("replay", "reference", "proxy", "independent_reference", "validation")
    offset = 27
    for role, count in zip(names, role_counts):
        protocol[role] = shuffled[offset : offset + count]
        offset += count
    protocol["pool"] = shuffled[offset:]
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    args = parser.parse_args()
    cases = prepare(args.raw_root, args.out_root, size=args.size)
    print(f"Prepared {len(cases)} patients and {sum(len(case.images) for case in cases.values())} labeled slices.")


if __name__ == "__main__":
    main()
