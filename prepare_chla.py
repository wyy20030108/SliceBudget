"""Prepare chamber-specific positive-only caches from the original CHLA release."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import zoom


URL = "https://github.com/saeedkarimi/Cardiac_MRI_Segmentation"
DOI = "10.1186/s12968-020-00678-0"
REVISION = "1e0d483045cbcd4d172156152c6ab7e8169528a4"
ARCHIVE_SHA256 = "eb9df013b43f0d1f4178c183476f67e61f583c3044479196ec83cbaef02fdee4"
CHAMBERS = {"RV": 1, "LV": 3}
SUBJECTS = range(1, 65)
ENTRY = re.compile(r"[^/]+/nearest/(LV|RV)/Sbj_(\d+)_(LV|RV)(ED|ES)_(image|mask)\.mat")


def archive_hash(path: Path) -> str:
    with path.open("rb") as source:
        return sha256(source.read()).hexdigest()


def read_archives(path: Path) -> dict:
    """Read only the author’s nearest-neighbor version and preserve its array order."""
    values = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            match = ENTRY.fullmatch(member.name)
            if match is None or not member.isfile():
                continue
            chamber, identifier, repeated, phase, kind = match.groups()
            key = (chamber, int(identifier), phase, kind)
            if chamber != repeated or key in values:
                raise ValueError(f"Duplicate or ambiguous member: {member.name}")
            # Keep the images compact while reading both chambers from the source tarball.
            source = loadmat(io.BytesIO(archive.extractfile(member).read()), variable_names=[kind])[kind]
            if source.ndim != 3 or source.shape[1:] != (128, 128) or not np.isfinite(source).all():
                raise ValueError(f"Unexpected release matrix: {member.name}")
            if kind == "mask" and not np.isin(source, [0, 1]).all():
                raise ValueError(f"Expected binary manual mask: {member.name}")
            if not np.equal(source, source.astype(np.uint8)).all():
                raise ValueError(f"Unexpected release pixel values: {member.name}")
            values[key] = source.astype(np.uint8)
    expected = {(chamber, subject, phase, kind) for chamber in CHAMBERS for subject in SUBJECTS
                for phase in ("ED", "ES") for kind in ("image", "mask")}
    if values.keys() != expected:
        raise ValueError(f"Expected {len(expected)} entries from the pinned release, found {len(values)}")
    return values


def prepare(source: Path, root: Path) -> None:
    checksum = archive_hash(source)
    if checksum != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 differs from the pinned snapshot")
    arrays = read_archives(source)
    for chamber, label in CHAMBERS.items():
        destination = root / chamber.lower()
        destination.mkdir(parents=True, exist_ok=True)
        counts = {"ED": Counter(), "ES": Counter()}
        support = {"ED": 0, "ES": 0}
        for subject in SUBJECTS:
            patient = f"chla{subject:02d}"
            images, masks, phases, indices, native = [], [], [], [], []
            for code, time in enumerate(("ED", "ES")):
                stack = arrays.pop((chamber, subject, time, "image"))
                contour = arrays.pop((chamber, subject, time, "mask"))
                if stack.shape != contour.shape:
                    raise ValueError(f"Reference grid does not match image: {chamber} {subject} {time}")
                areas = np.count_nonzero(contour, axis=(1, 2))
                positive = areas > 0
                if not positive.any():
                    raise ValueError(f"No manual positive slices: {chamber} {subject} {time}")
                counts[time]["released_slices"] += len(areas)
                counts[time]["positive_slices"] += int(positive.sum())
                counts[time]["excluded_blank_slices"] += int((~positive).sum())
                small_count = int((positive & (4 * areas <= areas.max())).sum())
                counts[time]["small_slices"] += small_count
                support[time] += bool(small_count)
                stack = stack.astype(np.float32)
                stack = ((stack - stack.mean(dtype=np.float64)) / max(stack.std(dtype=np.float64), 1e-6)).astype(np.float32)
                images.append(zoom(stack[positive], (1, 1.5, 1.5), order=1, mode="nearest", grid_mode=True))
                target = zoom(contour[positive], (1, 1.5, 1.5), order=0, mode="nearest", grid_mode=True) * label
                if not np.any(target, axis=(1, 2)).all():
                    raise ValueError(f"Resized reference lost a contour: {chamber} {subject} {time}")
                masks.append(target)
                phases.append(np.full(int(positive.sum()), code, dtype=np.int8))
                indices.append(np.flatnonzero(positive).astype(np.int32))
                areas_by_label = np.zeros((int(positive.sum()), 3), dtype=np.int32)
                areas_by_label[:, label - 1] = areas[positive]
                native.append(areas_by_label)
            native_areas = np.concatenate(native)
            with tempfile.NamedTemporaryFile(dir=destination, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, patient_id=patient, images=np.concatenate(images), masks=np.concatenate(masks),
                                    phase=np.concatenate(phases), slice_index=np.concatenate(indices),
                                    native_areas=native_areas, verified_absent=np.zeros_like(native_areas, dtype=bool))
            temporary.replace(destination / f"{patient}.npz")
        manifest = {"cohort": "CHLA pediatric congenital heart disease", "source_url": URL, "source_article_doi": DOI,
                    "release_revision": REVISION, "source_archive_sha256": checksum, "patients": len(SUBJECTS),
                    "labels": [label], "phase_scope": "cine", "released_grid": [128, 128], "model_grid": [192, 192],
                    "eligibility": "Only explicitly contoured positive slices in each released chamber and phase stack",
                    "area_basis": "Counts on the released 128×128 binary mask before resizing",
                    "phases": {time: {**counts[time], "small_supporting_patients": support[time]} for time in ("ED", "ES")}}
        (destination / "coverage.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.out_root)
