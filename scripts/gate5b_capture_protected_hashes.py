#!/usr/bin/env python3
"""Capture Gate 5b protected formal-source hashes without modifying sources."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import dump, sha256_file, write_report  # noqa: E402


PROTECTED = (
    "results/gate4a_formal_decision.json",
    "results/gate4b_expanded_formal_decision.json",
    "results/gate4b_cross_dataset_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "results/gate5a_formal_decision.json",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "reports/gate4b_multiseed_dataset_summary.csv",
    "reports/gate5a_distribution_results.csv",
    "reports/gate5a_pairwise_deltas.csv",
)


def main():
    rows = []
    for relative in PROTECTED:
        path = ROOT / relative
        rows.append({
            "path": relative, "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    payload = {
        "status": "PASS" if all(row["exists"] for row in rows) else "FAILED",
        "purpose": "Gate 5b pre-formal protected-asset baseline",
        "protected_assets": rows,
    }
    dump(ROOT / "results/gate5b_protected_assets_preflight.json", payload)
    write_report(ROOT / "reports/gate5b_protected_assets_preflight.md", "Gate 5b Protected Assets Preflight", payload)
    print(json.dumps({"status": payload["status"], "asset_count": len(rows)}, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

