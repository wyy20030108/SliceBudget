"""Retrieve hash-verified cropped HVSMR-2.0 NIfTI and technical metadata.

The source is Pace et al.'s Figshare version 2, DOI 10.6084/m9.figshare.25226366.v2,
licensed CC BY 4.0. This script uses the pinned third-party mirror because the
original Figshare API and downloader returned HTTP 403 during the study.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import subprocess


SOURCE_DOI = "10.6084/m9.figshare.25226366.v2"
MIRROR_REPO = "MedOtter/HVSMR-2.0"
REVISION = "f98902627611298bf41ae95a68525df73ab3558b"
INVENTORY_SHA256 = "800b14eba1b5eea38f74af4b57c88bb30523ad83b32785ee0aa3031925b18bdf"
TECHNICAL_FILE = {
    "name": "hvsmr_technical.csv", "size": 2222,
    "sha256": "7af95e023f4828fd17d82cf45d3ebec904f14e5c446390b7c76514e24e3dfed0",
}
CURL = ("curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3", "--max-time", "120")


def inventory(endpoint):
    api = f"{endpoint}/api/datasets/{MIRROR_REPO}/tree/{REVISION}/cropped"
    items = json.loads(subprocess.check_output([*CURL, api]))
    expected = {f"cropped/pat{number}_cropped{suffix}.nii.gz" for number in range(60)
                for suffix in ("", "_seg", "_seg_endpoints")}
    if len(items) != len(expected) or {item["path"] for item in items} != expected:
        raise ValueError("Pinned cropped tree does not contain 60 images, masks and endpoint volumes")
    files = sorted([{"name": Path(item["path"]).name, "size": item["size"], "sha256": item["lfs"]["oid"]}
                    for item in items], key=lambda file: file["name"])
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if sha256(serialized).hexdigest() != INVENTORY_SHA256:
        raise ValueError("The data tree differs from the pinned list of names, lengths and hashes")
    return files


def verified(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    checksum = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest() == entry["sha256"]


def download(target, base, entry, relative):
    path = target / entry["name"]
    if verified(path, entry):
        return
    partial = path.with_suffix(path.suffix + ".part")
    try:
        url = f"{base}/{relative}?download=true"
        subprocess.run([*CURL, "--output", str(partial), url], check=True)
        if not verified(partial, entry):
            raise ValueError(f"Downloaded content does not match the pinned checksum: {path.name}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def fetch(target, endpoint, workers):
    files = inventory(endpoint)
    target.mkdir(parents=True, exist_ok=True)
    base = f"{endpoint}/datasets/{MIRROR_REPO}/resolve/{REVISION}"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        combined = [(entry, f"cropped/{entry['name']}") for entry in files]
        combined.append((TECHNICAL_FILE, TECHNICAL_FILE["name"]))
        futures = [pool.submit(download, target, base, entry, relative) for entry, relative in combined]
        for future in futures:
            future.result()
    provenance = {"provider_doi": SOURCE_DOI, "mirror": f"https://huggingface.co/datasets/{MIRROR_REPO}",
                  "revision": REVISION, "inventory_sha256": INVENTORY_SHA256,
                  "technical_csv_sha256": TECHNICAL_FILE["sha256"], "files": files}
    staged = target / "provenance.json.tmp"
    staged.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    staged.replace(target / "provenance.json")
    print(f"Verified {len(files)} NIfTI files and the technical CSV at {REVISION}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"),
                        default="https://huggingface.co")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Specify at least one download worker")
    fetch(args.output, args.endpoint, args.workers)


if __name__ == "__main__":
    main()
