"""Download the annotated ED/ES training volumes from the official ACDC host."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"
TRAINING_FOLDER = "63721d7073e9f0047faa0525"
LICENSE_ITEM = "66e290e0961576b1bad4ee3c"
FILE_PATTERN = re.compile(r"patient\d{3}_frame\d+(?:_gt)?\.nii\.gz")


def get(path):
    for attempt in range(4):
        try:
            with urlopen(f"{API}/{path}", timeout=45) as response:
                return response.read()
        except HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)


def list_resources(path, **filters):
    return json.loads(get(path + "?" + urlencode({**filters, "limit": 200})))


def download_patient(folder, root):
    patient = folder["name"]
    if not re.fullmatch(r"patient\d{3}", patient):
        raise ValueError(f"Unexpected training folder name {patient!r}")
    patient_dir = root / patient
    patient_dir.mkdir(exist_ok=True)
    items = list_resources("item", folderId=folder["_id"])
    items = [item for item in items if item["name"] == "Info.cfg" or FILE_PATTERN.fullmatch(item["name"])]
    if len(items) != 5 or sum(item["name"].endswith("_gt.nii.gz") for item in items) != 2:
        raise ValueError(f"{patient} does not have the expected ED/ES image and mask items")
    for item in items:
        target = patient_dir / item["name"]
        if target.is_file() and target.stat().st_size == item["size"]:
            continue
        contents = get(f"item/{item['_id']}/download")
        if len(contents) != item["size"]:
            raise ValueError(f"Unexpected download size for {target.name}")
        temp = target.with_suffix(target.suffix + ".part")
        temp.write_bytes(contents)
        temp.replace(target)
    return patient


def download(destination, workers):
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "LICENSE_TERMS.md").write_bytes(get(f"item/{LICENSE_ITEM}/download"))
    root = destination / "training"
    root.mkdir(exist_ok=True)
    patients = list_resources("folder", parentType="folder", parentId=TRAINING_FOLDER)
    if len(patients) != 100:
        raise ValueError(f"Expected 100 official training patients, found {len(patients)}")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, patient in enumerate(executor.map(lambda folder: download_patient(folder, root), patients), 1):
            if index % 10 == 0 or index == len(patients):
                print(f"{index}/{len(patients)} through {patient}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 13))
    args = parser.parse_args()
    download(args.destination, args.workers)


if __name__ == "__main__":
    main()
