"""Download and prepare the official 50-case ACDC test release for one frozen audit."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acdc_data import prepare
from download_acdc import LICENSE_ITEM, download_patient, get, list_resources


OFFICIAL_TEST = "6372203a73e9f0047faa117e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    (args.raw / "LICENSE_TERMS.md").write_bytes(get(f"item/{LICENSE_ITEM}/download"))
    patients = list_resources("folder", parentType="folder", parentId=OFFICIAL_TEST)
    if len(patients) != 50:
        raise ValueError(f"Expected 50 official test patients, got {len(patients)}")
    with ThreadPoolExecutor(max_workers=8) as workers:
        for index, _ in enumerate(workers.map(lambda folder: download_patient(folder, args.raw), patients), 1):
            if index % 10 == 0:
                print(f"Fetched {index}/50", flush=True)
    cases = prepare(args.raw, args.cache, size=192)
    print(f"Prepared {len(cases)} official challenge test cases", flush=True)


if __name__ == "__main__":
    main()
