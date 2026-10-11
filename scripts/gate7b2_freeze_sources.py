#!/usr/bin/env python3
"""Freeze Gate 7B1 inputs before creating private Gate 7B2 metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts/gate7b2_pre_metadata"
HASH_FILE = OUTPUT_DIR / "source_hashes.sha256"
MANIFEST_FILE = OUTPUT_DIR / "source_manifest.json"

REQUIRED_SOURCES = [
    "results/gate7b1_jof_status.json",
    "results/gate7b1_jof_guideline_audit.json",
    "paper/venues/journal_of_forecasting/submission/manuscript.pdf",
    "paper/venues/journal_of_forecasting/submission/manuscript_anonymous.pdf",
    "paper/venues/journal_of_forecasting/submission/supplementary.pdf",
    "paper/venues/journal_of_forecasting/submission/title_page.pdf",
    "paper/venues/journal_of_forecasting/submission/cover_letter_draft.pdf",
    "paper/venues/journal_of_forecasting/metadata/submission_metadata.json",
    "paper/venues/journal_of_forecasting/journal_of_forecasting_submission_package.tar.gz",
    "results/final_evidence_registry.json",
    "results/gate5a_formal_decision.json",
    "results/gate5b_formal_decision.json",
]

FORMAL_PROTECTED_ASSETS = [
    "results/gate4a_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "results/gate5a_formal_decision.json",
    "results/gate5b_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5b_patchtst_probability_results.csv",
    "results/final_evidence_registry.json",
    "paper/manuscript/build/manuscript.pdf",
    "paper/final_submission_package.tar.gz",
    "paper/venues/journal_of_forecasting/submission/manuscript.pdf",
    "paper/venues/journal_of_forecasting/submission/manuscript_anonymous.pdf",
    "paper/venues/journal_of_forecasting/submission/supplementary.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required Gate 7B2 input is missing: {relative}")
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def verify_existing(manifest: dict[str, object]) -> None:
    for group in ("required_sources", "formal_protected_assets"):
        entries = manifest.get(group)
        if not isinstance(entries, list):
            raise RuntimeError(f"Existing manifest is missing {group}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise RuntimeError(f"Malformed manifest entry in {group}")
            relative = str(entry["path"])
            actual = record(relative)
            if actual["sha256"] != entry.get("sha256"):
                raise RuntimeError(f"Frozen source changed before Gate 7B2 rerun: {relative}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MANIFEST_FILE.exists() or HASH_FILE.exists():
        if not MANIFEST_FILE.is_file() or not HASH_FILE.is_file():
            raise RuntimeError("Gate 7B2 freeze is incomplete; refusing to replace it")
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        verify_existing(manifest)
        print("Gate 7B2 pre-metadata freeze already exists and is unchanged.")
        return 0

    required = [record(relative) for relative in REQUIRED_SOURCES]
    protected = [record(relative) for relative in FORMAL_PROTECTED_ASSETS]
    timestamp = datetime.now(timezone.utc).astimezone().isoformat()
    manifest = {
        "stage": "Gate 7B2 pre-metadata freeze",
        "created_at": timestamp,
        "required_source_count": len(required),
        "formal_protected_asset_count": len(protected),
        "required_sources": required,
        "formal_protected_assets": protected,
    }
    HASH_FILE.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in required),
        encoding="utf-8",
    )
    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen {len(required)} Gate 7B1 inputs and {len(protected)} protected assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
