"""Download the pinned organizer-published labeled CMR-MULTI SAX files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1, sha256
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


HUB = "https://huggingface.co"
REPO = "TaipingQu/CMR-MULTI"
REVISION = "cfdbfec4b18b245bd96b631b5b330236ae2f238e"
FILENAME = re.compile(r"CINE_SAX_\d+\.nii\.gz")


def fetch_json(url):
    with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=90) as response:
        return json.load(response)


def tree(relative, recursive=False):
    suffix = "/"+quote(relative, safe="/") if relative else ""
    query = "?recursive=true" if recursive else ""
    return fetch_json(f"{HUB}/api/datasets/{REPO}/tree/{REVISION}{suffix}{query}")


def release():
    info = fetch_json(f"{HUB}/api/datasets/{REPO}")
    if info.get("id") != REPO or info.get("private") or info.get("gated"):
        raise RuntimeError("Organizer repository is unavailable as an anonymous public release")
    files = []
    roster = {}
    for split in ("TR", "VAL"):
        rows = tree(f"CINE_MULTI/SAX_{split}", recursive=True)
        by_kind = {k: {} for k in ("image", "anno")}
        for entry in rows:
            rel = PurePosixPath(entry["path"])
            if entry["type"] == "file" and rel.parent.name in by_kind and FILENAME.fullmatch(rel.name):
                by_kind[rel.parent.name][rel.name] = entry
        common = sorted(set(by_kind["image"]) & set(by_kind["anno"]))
        if not common or (split == "TR" and len(common) != 105):
            raise RuntimeError(f"Unexpected labeled {split} roster at pinned revision: {len(common)}")
        roster[split] = {
            "paired": common,
            "image_only": sorted(set(by_kind["image"]) - set(common)),
            "annotation_only": sorted(set(by_kind["anno"]) - set(common)),
        }
        files.extend(by_kind[k][n] for k in by_kind for n in common)
    files.extend(row for row in tree("CINE_MULTI")
                 if row["type"] == "file" and PurePosixPath(row["path"]).suffix in (".xlsx", ".json"))
    return files, roster


def file_digest(path, entry):
    if path.stat().st_size != entry["size"]:
        return False
    linked = entry.get("lfs")
    h = sha256() if linked else sha1(f"blob {entry['size']}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest() == (linked["oid"] if linked else entry["oid"])


def download(root, entry):
    relative = PurePosixPath(entry["path"])
    target = root.joinpath(*relative.parts)
    if target.exists() and file_digest(target, entry):
        return
    url = f"{HUB}/datasets/{REPO}/resolve/{REVISION}/{quote(entry['path'], safe='/')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(target.suffix+".part")
    for attempt in range(6):
        try:
            with urlopen(url, timeout=100) as response, staged.open("wb") as handle:
                for block in iter(lambda: response.read(1024*1024), b""):
                    handle.write(block)
            if not file_digest(staged, entry):
                raise ValueError("size or organizer digest mismatch")
            staged.replace(target)
            return
        except (OSError, ValueError) as error:
            if attempt == 5 or (isinstance(error, HTTPError) and error.code in (401, 403, 404)):
                raise RuntimeError(f"Could not verify {entry['path']}") from error
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    files, roster = release()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"repo": REPO, "revision": REVISION, "roster": roster, "files": files}
    manifest = args.output/"release_manifest.json"
    (args.output/"release_pending.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(f"Organizer roster: TR={len(roster['TR']['paired'])}, VAL={len(roster['VAL']['paired'])}; "
          f"{len(files)} files, {sum(row['size'] for row in files)} bytes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        jobs = [workers.submit(download, args.output, entry) for entry in sorted(files, key=lambda e: e["size"])]
        for completed, job in enumerate(as_completed(jobs), 1):
            job.result()
            if completed % 20 == 0 or completed == len(files):
                print(f"Verified {completed}/{len(files)} files", flush=True)
    (args.output/"release_pending.json").replace(manifest)


if __name__ == "__main__":
    main()
