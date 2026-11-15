"""Prepare native ED/ES short-axis cine references from the public CMR-MULTI files."""

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

import nibabel as nib
import numpy as np

from prepare_hvsmr import file_sha256, resize_axial


SOURCE = "https://huggingface.co/datasets/TaipingQu/CMR-MULTI"
REVISION = "cfdbfec4b18b245bd96b631b5b330236ae2f238e"
PAPER = "https://arxiv.org/abs/2604.04078"
PATTERN = re.compile(r"CINE_SAX_(\d{3})\.nii\.gz")
SPLITS = (("TR", "sax_slice_info.json", "dataset_train.xlsx", range(1, 106)),
          ("VAL", "id_slice_info_valid.json", "dataset_valid.xlsx", range(106, 121)))
LUT = np.array([0, 2, 3, 1], dtype=np.uint8)
CLASSES = ("RV", "MYO", "LV")
NAMESPACE = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def xlsx_rows(path):
    """Read the first, SAX worksheet of the publisher's metadata workbook."""
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        if workbook.find("s:sheets/s:sheet", NAMESPACE).get("name") != "SAX":
            raise ValueError(f"First worksheet is not SAX: {path.name}")
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        else:
            strings = []
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    for row in sheet.findall("s:sheetData/s:row", NAMESPACE):
        values = {}
        for cell in row.findall("s:c", NAMESPACE):
            text = cell.findtext("s:v", namespaces=NAMESPACE)
            if cell.get("t") == "s":
                text = strings[int(text)]
            elif cell.get("t") == "inlineStr":
                text = cell.findtext("s:is/s:t", namespaces=NAMESPACE)
            column = re.match(r"[A-Z]+", cell.get("r"))[0]
            values[column] = text
        yield values


def ejection_fractions(path, split, expected):
    rows = iter(xlsx_rows(path))
    header = {name.strip().casefold(): col for col, name in next(rows).items() if name}
    id_col, image_col, label_col, ef_col = [header[c] for c in ("patient_id", "image_path", "anno_path", "lvef")]
    results = {}
    for row in rows:
        value = row.get(id_col)
        if not value:
            continue
        case = f"{int(value):03}"
        if case in results:
            raise ValueError(f"Duplicate SAX participant in {path.name}: {case}")
        for column, kind in ((image_col, "image"), (label_col, "anno")):
            actual = str(row.get(column, "")).replace("\\", "/").strip()
            if not actual.endswith(f"/SAX_{split}/{kind}/CINE_SAX_{case}.nii.gz"):
                raise ValueError(f"Metadata pairing does not match source case {case}: {actual}")
        text = (row.get(ef_col) or "").strip()
        ef = float(text[:-1]) / 100 if text.endswith("%") else (float(text) if text else None)
        if ef is not None and (not np.isfinite(ef) or not 0 <= ef <= 1):
            raise ValueError(f"Invalid released ejection fraction for {case}")
        results[case] = ef
    if set(results) != expected:
        raise ValueError(f"Unexpected SAX participant roster in {path.name}")
    return results


def source_pairs(root, split_info):
    split, phase_file, ef_file, numbers = split_info
    expected = {f"{number:03}" for number in numbers}
    phase_counts = json.loads((root / phase_file).read_text(encoding="utf-8-sig"))
    if set(phase_counts) != expected or any(p not in (25, 30, 50) for p in phase_counts.values()):
        raise ValueError(f"Unexpected participants or cine phase counts in {phase_file}")
    efs = ejection_fractions(root / ef_file, split, expected)
    folder = root / f"SAX_{split}"
    entries = {}
    for kind in ("image", "anno"):
        files = {match[1]: path for path in (folder / kind).glob("*.nii.gz")
                 if (match := PATTERN.fullmatch(path.name))}
        if set(files) != expected:
            raise ValueError(f"Unexpected or missing {split} {kind} files: {sorted(set(files) ^ expected)}")
        entries[kind] = files
    return [(case, entries["image"][case], entries["anno"][case], phase_counts[case], efs[case])
            for case in sorted(expected)]


def case_arrays(image_path, mask_path, phases, published_ef, size):
    scan, label = nib.load(str(image_path)), nib.load(str(mask_path))
    if (len(scan.shape) != 3 or scan.shape != label.shape or scan.shape[-1] % phases
            or not np.allclose(scan.affine, label.affine, rtol=1e-5, atol=1e-3)):
        raise ValueError(f"Expected aligned packed SAX volumes: {image_path.name}")
    n_slices = scan.shape[-1] // phases
    if n_slices < 3:
        raise ValueError(f"Too few SAX slices: {image_path.name}")
    raw = np.asanyarray(label.dataobj)
    if not np.isin(raw, (0, 1, 2, 3)).all():
        raise ValueError(f"Unexpected CMR-MULTI SAX labels: {mask_path.name}")

    # The release stores all cardiac phases of native slice 0 followed by
    # all phases of slice 1, and so on. Its third spacing mixes slice and time.
    raw = raw.reshape(*scan.shape[:2], n_slices, phases)
    lv_counts = (raw == 2).sum(axis=(0, 1, 2))
    ed = int(lv_counts.argmax())
    es = int(lv_counts.argmin())
    if lv_counts[es] == 0 or lv_counts[ed] <= lv_counts[es]:
        raise ValueError(f"Incomplete or constant LV reference throughout the cycle: {mask_path.name}")
    inferred_ef = float(1 - lv_counts[es] / lv_counts[ed])
    values = scan.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite SAX intensities: {image_path.name}")
    volume_digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    voxel_spacing = np.asarray(scan.header.get_zooms()[:2][::-1], dtype=np.float32)
    if not np.isfinite(voxel_spacing).all() or (voxel_spacing <= 0).any():
        raise ValueError(f"Invalid in-plane spacing: {image_path.name}")

    image_frames, mask_frames, areas, spacings = [], [], [], []
    for frame in (ed, es):
        image = values[:, :, frame::phases].transpose(2, 1, 0)
        mask = LUT[raw[:, :, :, frame].astype(np.uint8)].transpose(2, 1, 0)
        areas.append(np.stack([(mask == i).sum(axis=(1, 2)) for i in (1, 2, 3)], axis=1))
        image, mask, spacing = resize_axial(image, mask, voxel_spacing, size)
        image_frames.append(image)
        mask_frames.append(mask)
        spacings.append(np.tile(spacing, (n_slices, 1)))
    native_areas = np.concatenate(areas).astype(np.int32)
    # A native zero is counted only on a same-phase plane with independently
    # contoured tissue and where the target has positive support at that phase.
    eligible_absent = []
    for per_phase in areas:
        covered = (per_phase > 0).any(axis=1, keepdims=True)
        target_contoured = (per_phase > 0).any(axis=0, keepdims=True)
        eligible_absent.append((per_phase == 0) & covered & target_contoured)
    verified_absent = np.concatenate(eligible_absent)
    masks = np.concatenate(mask_frames)
    post_resize = np.stack([(masks == i).any(axis=(1, 2)) for i in (1, 2, 3)], axis=1)
    data = {"images": np.concatenate(image_frames), "masks": masks, "native_areas": native_areas,
            "verified_absent": verified_absent, "spacing_yx": np.concatenate(spacings),
            "phase": np.repeat(np.arange(2, dtype=np.int8), n_slices),
            "slice_index": np.tile(np.arange(n_slices, dtype=np.int32), 2)}
    support = {"ED_ES_frames": [ed, es], "frames_in_cycle": phases, "slices_per_phase": n_slices,
               "released_lvef": published_ef, "reference_extrema_lvef": inferred_ef,
               "native_blank_planes": int((native_areas.sum(axis=1) == 0).sum()),
               "native_positive": (native_areas > 0).sum(axis=0).tolist(),
               "verified_absent": verified_absent.sum(axis=0).tolist(),
               "native_positive_lost_at_resize": ((native_areas > 0) & ~post_resize).sum(axis=0).tolist()}
    return data, support, volume_digest


def grid_exclusions(pairs):
    accepted, excluded = [], {}
    for entry in pairs:
        split, case, image, mask, phases, _ = entry
        image_header, mask_header = nib.load(str(image)), nib.load(str(mask))
        if (len(image_header.shape) != 3 or image_header.shape != mask_header.shape
                or not np.allclose(image_header.affine, mask_header.affine, rtol=1e-5, atol=1e-3)):
            raise ValueError(f"Released image and mask grids differ for {case}")
        flat_count = image_header.shape[-1]
        if flat_count % phases:
            excluded[f"case-CINE_SAX_{case}"] = {
                "release_split": split, "packed_plane_count": flat_count, "released_phase_count": phases,
                "reason": "Packed plane count is not divisible by the released phase count.",
                "image_sha256": file_sha256(image), "source_mask_sha256": file_sha256(mask),
            }
        else:
            accepted.append(entry)
    return accepted, excluded


def write_npz(directory, patient, arrays):
    with NamedTemporaryFile(dir=directory, suffix=".npz.tmp", delete=False) as handle:
        temp = Path(handle.name)
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, patient_id=patient, **arrays)
        temp.replace(directory / f"{patient}.npz")
    finally:
        temp.unlink(missing_ok=True)


def prepare(raw, output, size, split_names):
    root = raw / "CINE_MULTI" if (raw / "CINE_MULTI").is_dir() else raw
    output.mkdir(parents=True, exist_ok=True)
    split_info = [item for item in SPLITS if item[0] in split_names]
    pairs = [(item[0], *pair) for item in split_info for pair in source_pairs(root, item)]
    released_counts = {name: sum(pair[0] == name for pair in pairs) for name in split_names}
    pairs, excluded = grid_exclusions(pairs)
    intended = {f"case-CINE_SAX_{case}" for _, case, *_ in pairs}
    unexpected = {path.stem for path in output.glob("*.npz")} - intended
    if unexpected:
        raise ValueError(f"Remove unrelated cached cases: {sorted(unexpected)}")
    coverage, digests = {}, {}
    for split, case, image, mask, phases, ef in pairs:
        patient = f"case-CINE_SAX_{case}"
        data, support, digest = case_arrays(image, mask, phases, ef, size)
        if digest in digests:
            raise ValueError(f"Identical source voxels for {case} and {digests[digest]}")
        digests[digest] = case
        write_npz(output, patient, data)
        coverage[patient] = {"release_split": split, **support,
                             "image_sha256": file_sha256(image), "source_mask_sha256": file_sha256(mask),
                             "cache_sha256": file_sha256(output / f"{patient}.npz")}
        print(f"Prepared {len(coverage)}/{len(pairs)} cases", flush=True)
    discrepancies = [abs(p["released_lvef"] - p["reference_extrema_lvef"]) for p in coverage.values()
                     if p["released_lvef"] is not None]
    summaries = {}
    for name in ("native_positive", "verified_absent", "native_positive_lost_at_resize"):
        counts = np.asarray([c[name] for c in coverage.values()])
        summaries[name] = {"slices": counts.sum(axis=0).tolist(), "patients": (counts > 0).sum(axis=0).tolist()}
    source_metadata = {name: file_sha256(root / name) for _, jf, xf, _ in split_info for name in (jf, xf)}
    result = {"source": SOURCE, "release_revision": REVISION, "paper": PAPER, "classes": CLASSES,
              "released_split_counts": released_counts,
              "evaluated_split_counts": {name: sum(row["release_split"] == name for row in coverage.values())
                                         for name in split_names},
              "geometry_exclusions": excluded,
              "patients": len(coverage), "plane_shape": [size, size],
              "ED_ES_rule": "Earliest maximum and minimum whole-stack native LV-cavity mask area in each cardiac cycle.",
              "annotation_scope": "Three SAX regions independently annotated by two radiologists with senior review.",
              "negative_definition": "Target zero on a native plane where another labeled structure is present,"
                                     " and target present somewhere in the same phase stack.",
              "released_lvef_check": {"patients": len(discrepancies),
                                      "within_four_decimals": int(sum(d <= 0.000051 for d in discrepancies)),
                                      "median_absolute_difference": float(np.median(discrepancies)) if discrepancies else None,
                                      "max_absolute_difference": max(discrepancies, default=None)},
              "planes": sum(2 * row["slices_per_phase"] for row in coverage.values()),
              "excluded_all_blank_negative_planes": sum(row["native_blank_planes"] for row in coverage.values()),
              "support": summaries, "source_metadata_sha256": source_metadata, "cases": coverage}
    (output / "coverage.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--splits", nargs="+", choices=("TR", "VAL"), default=("TR", "VAL"))
    args = parser.parse_args()
    if not 16 <= args.size <= 512 or args.raw.resolve() == args.output.resolve():
        parser.error("Use distinct source and output directories and a size from 16 to 512")
    prepare(args.raw, args.output, args.size, tuple(dict.fromkeys(args.splits)))


if __name__ == "__main__":
    main()
