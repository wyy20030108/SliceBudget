"""Audit retained ACDC run records without exporting patient-level outcomes."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from acdc_data import make_protocol


PROTOCOL_SEED = 2026
PATIENT = re.compile(r"(?<![A-Za-z0-9])patient[0-9]{3}(?![0-9])", re.IGNORECASE)
DECLARATION = "legacy test roster declaration"
SENSITIVITY = "designated reserved sensitivity assessment"
UNKNOWN = "unclassified reserved patient mention"
GUARD_NAMES = {"holdout.json", "holdout_started.json", ".case_selection_holdout_started.json"}


def patient_ids(value):
    return {match.lower() for match in PATIENT.findall(value)}


def classify(root, path, document, locked, hits, legacy_code_sha256):
    relative = path.relative_to(root).as_posix()
    if path.name == "validation_checks.json" and "held_out_test_used" in document and not hits:
        used = document["held_out_test_used"]
        return {
            "role": "legacy confirmation gate",
            "held_out_test_used": used,
            "failed_checks": sum(check.get("passes") is False for check in document.get("checks", [])),
        }
    if path.name in GUARD_NAMES:
        return {"role": "legacy holdout evaluation marker or outcome"}
    if not hits:
        return {"role": "no reserved ID mention"}
    if path.name == "protocol.json" and set(document.get("test", [])) == locked:
        other_roles = {key: value for key, value in document.items() if key != "test"}
        config = json.loads(path.with_name("run.json").read_bytes())
        protocol_digest = sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        if (not patient_ids(json.dumps(other_roles)) & locked
                and config.get("protocol_seed") == PROTOCOL_SEED
                and config.get("mode") in {"pilot", "validation"}
                and config.get("protocol_sha256") == protocol_digest):
            archived_code = config.get("code_sha256", {}).get("experiment.py")
            return {
                "role": DECLARATION, "field": "test", "run_mode": config["mode"],
                "legacy_code_sha256": archived_code,
                "matches_retained_legacy_code": archived_code == legacy_code_sha256,
            }
    if relative in {"runs/reviewer-audit/fold-0.json", "runs/reviewer-audit/fold-1.json"}:
        arms = document.get("arms", {})
        if len(arms) == 7 and all(set(arm.get("cohorts", {}).get("reserved", {})) == locked
                                  for arm in arms.values()):
            for arm in arms.values():
                del arm["cohorts"]["reserved"]
            if not patient_ids(json.dumps(document)) & locked:
                return {"role": SENSITIVITY, "field": "arms.*.cohorts.reserved", "arms": len(arms)}
    return {"role": UNKNOWN}


def scan(root, archive):
    training = {path.stem for path in (root / "data/acdc-192").glob("patient*.npz")}
    official = {path.stem for path in (root / "data/acdc-challenge-audit-192").glob("patient*.npz")}
    if (training != {f"patient{i:03d}" for i in range(1, 101)}
            or official != {f"patient{i:03d}" for i in range(101, 151)}):
        raise ValueError("Unexpected ACDC cache roster")
    locked = set(make_protocol(dict.fromkeys(training), PROTOCOL_SEED)["test"])
    manifests = set((root / "runs").rglob("*.json")) | set(root.glob("registration-*.json"))
    markers = set((root / "runs").rglob(".case_selection_holdout_started.json"))
    markers.update((root / "data").rglob(".case_selection_holdout_started.json"))
    if not manifests:
        raise ValueError("No retained run manifests or registration records found")
    legacy_code_sha256 = sha256((root / "experiment.py").read_bytes()).hexdigest()
    records = []
    active_roles = Counter()
    for path in sorted(manifests | markers):
        raw = path.read_bytes()
        document = json.loads(raw)
        ids = patient_ids(raw.decode("utf-8-sig"))
        hits = ids & locked
        classification = classify(root, path, document, locked, hits, legacy_code_sha256)
        if path.name == "run.json" and isinstance(document, dict):
            for role in ("training", "calibration", "assessment"):
                roster = document.get(role)
                if isinstance(roster, list):
                    active_roles[role] += 1
        records.append({
            "file": path.relative_to(root).as_posix(),
            "sha256": sha256(raw).hexdigest(),
            "acdc_ids_mentioned": len(ids),
            "reserved_ids_mentioned": len(hits),
            **classification,
        })
    counts = Counter(row["role"] for row in records)
    unresolved = [row["file"] for row in records if row["role"] in {
        UNKNOWN, "legacy holdout evaluation marker or outcome",
    } or (row["role"] == "legacy confirmation gate" and row["held_out_test_used"] is not False)]
    ledger_names = {"run.json", "protocol.json", "roster.json", "audit_plan.json", "source_replay.json"}
    summary = {
        "archive": archive,
        "scope": "All retained runs/**/*.json, root registration-*.json and the legacy holdout-start marker in runs or data.",
        "interpretation": "Checks retained project records next to the ACDC caches. Does not establish use outside these records or independence from prior PRCV experiments.",
        "protocol_seed": PROTOCOL_SEED,
        "reserved_patients": len(locked),
        "reserved_roster_sha256": sha256("\n".join(sorted(locked)).encode()).hexdigest(),
        "training_release_cache_cases": len(training),
        "official_assessment_cache_cases": len(official),
        "reserved_cache_cases": len(training & locked),
        "records_inspected": len(records),
        "named_run_roster_and_audit_ledgers": sum(Path(row["file"]).name in ledger_names for row in records),
        "run_json_files_declaring_roles": dict(active_roles),
        "records_by_role": dict(sorted(counts.items())),
        "retained_legacy_guard_sha256": legacy_code_sha256,
        "legacy_gate_opened": any(row.get("held_out_test_used") is True for row in records),
        "unresolved_records": unresolved,
    }
    return {"summary": summary, "manifest_index": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--archive", choices=("project-copy", "retained-autodl"), required=True)
    args = parser.parse_args()
    print(json.dumps(scan(args.root, args.archive), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
