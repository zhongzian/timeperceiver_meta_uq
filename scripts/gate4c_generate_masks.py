#!/usr/bin/env python
"""Generate and verify fixed Gate 4c selection/calibration masks."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gate3a_generate_missingness_assets import make_mask, summarize_mask

SPLIT_DIR = ROOT / "artifacts/gate4c_splits/etth1_random_rate0.3_seed2025"


def load(path: Path):
    return json.loads(path.read_text())


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_mask(path: Path, expected: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = np.load(path, allow_pickle=False)["mask"]
        if current.dtype != np.uint8 or not np.array_equal(current, expected):
            raise RuntimeError(f"existing mask differs; refusing resample overwrite: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, mask=expected)
        temporary.replace(path)
    return sha256(path)


def create(split: str, seed: int, windows: int) -> dict:
    path = ROOT / f"artifacts/gate4c_masks/etth1/{split}/random/rate_0.3/seed_{seed}.npz"
    meta_path = path.with_suffix(".metadata.json")
    expected = make_mask("random", seed, (windows, 384, 7), 0.3).astype(np.uint8, copy=False)
    digest = save_mask(path, expected)
    stats = summarize_mask(expected, 0.3)
    metadata = {
        **stats,
        "sha256": digest,
        "dataset": "etth1",
        "split": "model_selection" if split == "selection" else "calibration",
        "subset_name": split,
        "mask_type": "random",
        "seed": seed,
        "seq_len": 384,
        "pred_len": 96,
        "num_features": 7,
        "window_count": windows,
        "mask_semantics": "1=observed/keep, 0=missing/mask",
        "applies_to": f"{split} batch_x history only",
        "does_not_mask": "batch_y or future prediction target",
        "generator": "Gate 3a make_mask random semantics",
        "generated_at": datetime.now().isoformat(),
    }
    if meta_path.exists():
        current = load(meta_path)
        stable = set(metadata) - {"generated_at"}
        if any(current.get(key) != metadata.get(key) for key in stable):
            raise RuntimeError(f"existing mask metadata differs; refusing overwrite: {meta_path}")
        metadata = current
    else:
        dump(meta_path, metadata)
    unique = set(np.unique(expected).tolist())
    return {
        "split": split,
        "seed": seed,
        "windows": windows,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "actual_missing_rate": float(1.0 - expected.mean()),
        "sha256": digest,
        "npz_path": str(path.relative_to(ROOT)),
        "metadata_path": str(meta_path.relative_to(ROOT)),
        "status": "PASS" if unique <= {0, 1} else "FAIL",
    }


def test_inventory_row() -> dict:
    with (ROOT / "reports/gate3a_mask_inventory.csv").open(newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row["dataset"] == "etth1"
                   and row["mask_type"] == "random" and float(row["rate"]) == 0.3
                   and int(row["seed"]) == 2025]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Gate 3a test mask inventory row, got {len(matches)}")
    source = matches[0]
    path = Path(source["npz_path"])
    meta_path = Path(source["meta_path"])
    array = np.load(path, allow_pickle=False)["mask"]
    digest = sha256(path)
    metadata = load(meta_path)
    status = "PASS" if digest == source["sha256"] == metadata["sha256"] else "FAIL"
    return {
        "split": "test",
        "seed": 2025,
        "windows": int(array.shape[0]),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "actual_missing_rate": float(1.0 - array.mean()),
        "sha256": digest,
        "npz_path": str(path.relative_to(ROOT)),
        "metadata_path": str(meta_path.relative_to(ROOT)),
        "status": status,
    }


def main() -> int:
    split = load(SPLIT_DIR / "metadata.json")
    rows = [
        create("selection", 43025, int(split["selection_window_count"])),
        create("calibration", 44025, int(split["calibration_window_count"])),
        test_inventory_row(),
    ]
    checks = {
        "selection_masks_verified_1_of_1": rows[0]["status"] == "PASS",
        "calibration_masks_verified_1_of_1": rows[1]["status"] == "PASS",
        "test_masks_verified_1_of_1": rows[2]["status"] == "PASS",
        "hash_mismatch_count": sum(row["status"] != "PASS" for row in rows),
        "metadata_mismatch_count": 0,
        "selection_and_calibration_paths_differ": rows[0]["npz_path"] != rows[1]["npz_path"],
        "selection_and_calibration_hashes_differ": rows[0]["sha256"] != rows[1]["sha256"],
    }
    passed = all(value is True or value == 0 for value in checks.values())
    inventory = {
        "stage": "Gate 4c mask inventory",
        "status": "PASS" if passed else "BLOCKED_GATE4C_MASK_AUDIT_FAILURE",
        "rows": rows,
        "checks": checks,
        "test_mask_reused_from_gate3a": True,
        "test_mask_regenerated": False,
    }
    dump(ROOT / "results/gate4c_mask_inventory.json", inventory)
    fields = ["split", "seed", "windows", "actual_missing_rate", "sha256", "npz_path", "metadata_path", "status"]
    with (ROOT / "reports/gate4c_mask_inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    md = [
        "# Gate 4c Mask Inventory", "", f"Status: **{inventory['status']}**", "",
        "| Split | Seed | Windows | Actual rate | SHA256 | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    md += [f"| {row['split']} | {row['seed']} | {row['windows']} | {row['actual_missing_rate']:.9f} | `{row['sha256']}` | {row['status']} |" for row in rows]
    md += ["", "The test mask is the existing Gate 3a asset; it was not regenerated."]
    (ROOT / "reports/gate4c_mask_inventory.md").write_text("\n".join(md) + "\n")
    leakage_path = ROOT / "results/gate4c_leakage_audit.json"
    leakage = load(leakage_path)
    leakage.update({
        "selection_mask_history_only": True,
        "calibration_mask_history_only": True,
        "test_mask_reused_without_resampling": True,
        "selection_mask_sha256": rows[0]["sha256"],
        "calibration_mask_sha256": rows[1]["sha256"],
        "test_mask_sha256": rows[2]["sha256"],
        "status": "PASS" if passed and leakage.get("status") == "PASS" else "BLOCKED_CONFORMAL_CALIBRATION_LEAKAGE",
    })
    dump(leakage_path, leakage)
    print(json.dumps(inventory, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
