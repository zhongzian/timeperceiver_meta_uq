#!/usr/bin/env python
"""Verify Gate 3a missingness assets."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = {"etth1", "weather", "electricity"}
EXPECTED_MASK_TYPES = {"random", "block", "channel", "mixed"}
EXPECTED_RATES = {"0.1", "0.3", "0.5", "0.7"}
EXPECTED_SEEDS = {"2025"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def main() -> int:
    inventory_path = ROOT / "reports" / "gate3a_mask_inventory.csv"
    errors: List[str] = []
    warnings: List[str] = []
    checked = 0
    seen = set()

    if not inventory_path.exists():
        errors.append(f"Missing inventory: {inventory_path}")
    else:
        with inventory_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                checked += 1
                key = (row["dataset"], row["mask_type"], str(row["rate"]), str(row["seed"]))
                seen.add(key)
                npz_path = Path(row["npz_path"])
                meta_path = Path(row["meta_path"])
                if not npz_path.exists():
                    errors.append(f"Missing npz: {npz_path}")
                    continue
                if not meta_path.exists():
                    errors.append(f"Missing metadata: {meta_path}")
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                actual_hash = sha256_file(npz_path)
                if meta.get("sha256") != actual_hash:
                    errors.append(f"sha256 mismatch: {npz_path}")
                loaded = np.load(npz_path, allow_pickle=False)
                mask = loaded["mask"]
                if list(mask.shape) != list(meta.get("shape")):
                    errors.append(f"shape mismatch: {npz_path} {mask.shape} != {meta.get('shape')}")
                if mask.dtype != np.uint8:
                    errors.append(f"dtype mismatch: {npz_path} {mask.dtype}")
                values = set(np.unique(mask).tolist())
                if not values.issubset({0, 1}):
                    errors.append(f"non-binary values: {npz_path} {values}")
                if meta.get("mask_semantics") != "1=observed/keep, 0=missing/mask":
                    errors.append(f"mask semantics mismatch: {meta_path}")
                if meta.get("applies_to") != "test input/history window seq_x only":
                    errors.append(f"scope mismatch: {meta_path}")
                if meta.get("warning"):
                    warnings.append(f"{meta_path}: {meta['warning']}")

    expected = {
        (dataset, mask_type, rate, seed)
        for dataset in EXPECTED_DATASETS
        for mask_type in EXPECTED_MASK_TYPES
        for rate in EXPECTED_RATES
        for seed in EXPECTED_SEEDS
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        errors.append(f"missing expected masks: {missing}")
    if extra:
        errors.append(f"unexpected masks in inventory: {extra}")

    status: Dict[str, Any] = {
        "gate": "Gate 3a verify",
        "timestamp": iso_now(),
        "verified": not errors,
        "checked": checked,
        "expected": len(expected),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "sha256": "OK" if not any("sha256" in e for e in errors) else "FAILED",
            "shape": "OK" if not any("shape" in e for e in errors) else "FAILED",
            "dtype": "OK" if not any("dtype" in e for e in errors) else "FAILED",
            "binary_values": "OK" if not any("non-binary" in e for e in errors) else "FAILED",
            "metadata": "OK" if not any("metadata" in e or "semantics" in e or "scope" in e for e in errors) else "FAILED",
        },
    }
    out = ROOT / "reports" / "gate3a_verify_status.json"
    out.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
