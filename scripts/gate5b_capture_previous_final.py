#!/usr/bin/env python3
"""Capture the pre-Gate-5b final-synthesis hash before rebuilding it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import dump, sha256_file  # noqa: E402


def main():
    source = ROOT / "reports/final_project_synthesis.md"
    target = ROOT / "results/gate5b_previous_final_synthesis_hash.json"
    if not source.is_file():
        raise RuntimeError("pre-Gate-5b final synthesis is missing")
    if target.is_file():
        payload = json.loads(target.read_text())
        print(json.dumps(payload, indent=2))
        return 0
    payload = {
        "status": "PASS",
        "path": "reports/final_project_synthesis.md",
        "sha256": sha256_file(source),
        "purpose": "Pre-Gate-5b final-synthesis snapshot; the source may change only during the verified Gate 5b rebuild.",
    }
    dump(target, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
