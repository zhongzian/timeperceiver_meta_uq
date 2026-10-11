#!/usr/bin/env python3
"""Freeze Gate 7A manuscript inputs without overwriting an existing baseline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/gate7a_pre_submission"
SOURCES = [
    "results/final_evidence_registry.json",
    "results/final_claim_evidence_registry.json",
    "results/final_project_status.json",
    "results/gate5a_formal_decision.json",
    "results/gate5b_formal_decision.json",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5a_pairwise_deltas.csv",
    "reports/gate5b_patchtst_probability_results.csv",
    "reports/gate5b_patchtst_paired_deltas.csv",
    "results/gate6_status.json",
    "results/gate6_submission_verification.json",
    "paper/submission/submission_manifest.json",
]
PROTECTED_FORMAL = [
    "results/gate4a_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "results/gate5a_formal_decision.json",
    "results/gate5b_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5a_pairwise_deltas.csv",
    "reports/gate5b_patchtst_probability_results.csv",
    "reports/gate5b_patchtst_paired_deltas.csv",
    "results/final_evidence_registry.json",
]


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_hashes(paths: list[str]) -> str:
    return "".join(f"{sha256(ROOT / path)}  {path}\n" for path in paths)


def verify_existing(path: Path, expected: str) -> None:
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError(f"existing Gate 7A freeze baseline differs: {path.relative_to(ROOT)}")


def main() -> None:
    missing = [path for path in sorted(set(SOURCES + PROTECTED_FORMAL)) if not (ROOT / path).is_file()]
    if missing:
        payload = {
            "status": "BLOCKED_GATE7A_MISSING_SUBMISSION_SOURCE",
            "generated_at": generated_at(),
            "missing": missing,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "source_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(1)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_text = format_hashes(SOURCES)
    formal_text = format_hashes(PROTECTED_FORMAL)
    source_hash_path = OUTPUT / "source_hashes.sha256"
    formal_hash_path = OUTPUT / "formal_source_hashes.sha256"
    frozen_existing = source_hash_path.is_file() or formal_hash_path.is_file()
    if frozen_existing:
        if not source_hash_path.is_file() or not formal_hash_path.is_file():
            raise RuntimeError("incomplete existing Gate 7A freeze baseline")
        verify_existing(source_hash_path, source_text)
        verify_existing(formal_hash_path, formal_text)
    else:
        source_hash_path.write_text(source_text, encoding="utf-8")
        formal_hash_path.write_text(formal_text, encoding="utf-8")

    rows = []
    for path in SOURCES:
        file_path = ROOT / path
        rows.append(
            {
                "file": path,
                "sha256": sha256(file_path),
                "size": file_path.stat().st_size,
                "role": "gate7a_submission_input",
                "read_only": True,
            }
        )
    payload = {
        "status": "PASS",
        "generated_at": generated_at(),
        "baseline_created": not frozen_existing,
        "source_count": len(SOURCES),
        "protected_formal_count": len(PROTECTED_FORMAL),
        "sources": rows,
        "source_hashes": "artifacts/gate7a_pre_submission/source_hashes.sha256",
        "formal_source_hashes": "artifacts/gate7a_pre_submission/formal_source_hashes.sha256",
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "source_count": len(SOURCES), "protected_formal_count": len(PROTECTED_FORMAL), "baseline_created": not frozen_existing}, sort_keys=True))


if __name__ == "__main__":
    main()
