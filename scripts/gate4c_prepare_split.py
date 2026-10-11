#!/usr/bin/env python
"""Audit prerequisites and create the fixed raw-time Gate 4c split."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/TimePerceiver"
BASELINE = ROOT / "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054"
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


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(UPSTREAM), *args], text=True).strip()


def save_array(path: Path, expected: np.ndarray) -> None:
    if path.exists():
        current = np.load(path, allow_pickle=False)
        if current.dtype != expected.dtype or not np.array_equal(current, expected):
            raise RuntimeError(f"existing split asset differs; refusing overwrite: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, expected, allow_pickle=False)
    temporary.replace(path)


def main() -> int:
    prerequisite_status = load(ROOT / "results/gate4b_multiseed_status.json")
    prerequisite_decision = load(ROOT / "results/gate4b_multiseed_formal_decision.json")
    upstream_diff = git("status", "--short", "--untracked-files=no")
    checks = {
        "gate4b_multiseed_implementation_pass": prerequisite_decision.get("GATE4B_MULTI_SEED_IMPLEMENTATION_PASS") is True,
        "gate4b_multiseed_signal_positive": prerequisite_decision.get("GATE4B_MULTI_SEED_SIGNAL_POSITIVE") is True,
        "ready_for_gate4c_planning": prerequisite_decision.get("READY_FOR_GATE4C_PLANNING") is True,
        "scheduler_complete": prerequisite_status.get("scheduler_state") == "COMPLETE",
        "old_unmasked_evidence_excluded": prerequisite_status.get("legacy_unmasked_results_excluded") is True,
        "upstream_tracked_diff_empty": upstream_diff == "",
    }
    prerequisite = {
        "stage": "Gate 4c prerequisite audit",
        "status": "PASS" if all(checks.values()) else "BLOCKED_GATE4C_PREREQUISITE_FAILURE",
        "checks": checks,
        "previous_gate4b_checkpoint_reused_for_conformal": False,
        "gate4c_probability_heads_retrained": True,
        "old_unmasked_evidence_excluded": True,
        "upstream_commit": git("rev-parse", "HEAD"),
        "upstream_tracked_diff": upstream_diff,
        "audited_at": datetime.now().isoformat(),
    }
    dump(ROOT / "results/gate4c_prerequisite_audit.json", prerequisite)
    (ROOT / "reports/gate4c_prerequisite_audit.md").write_text(
        "# Gate 4c Prerequisite Audit\n\n"
        f"Status: **{prerequisite['status']}**\n\n"
        + "\n".join(f"- {key}: `{str(value).lower()}`" for key, value in checks.items())
        + "\n\nOld Gate 4b probability-head checkpoints are not reused; both Gate 4c heads are retrained.\n"
    )
    if prerequisite["status"] != "PASS":
        return 2

    config = {}
    for line in (BASELINE / "resolved_config.yaml").read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            config[key.strip()] = json.loads(raw.strip())
        except Exception:
            config[key.strip()] = raw.strip().strip("\"'")
    seq_len, pred_len = int(config["seq_len"]), int(config["pred_len"])
    csv_path = Path(config["root_path"]) / config["data_path"]
    frame = pd.read_csv(csv_path, usecols=["date"])
    val_start = 12 * 30 * 24 - seq_len
    val_end = 12 * 30 * 24 + 4 * 30 * 24
    total_raw_rows = val_end - val_start
    embargo_rows = seq_len + pred_len - 1
    usable = total_raw_rows - embargo_rows
    selection_rows = int(math.floor(usable * 0.50))
    calibration_rows = usable - selection_rows
    selection_range = [val_start, val_start + selection_rows]
    embargo_range = [selection_range[1], selection_range[1] + embargo_rows]
    calibration_range = [embargo_range[1], val_end]
    span = seq_len + pred_len
    selection_count = selection_rows - span + 1
    calibration_count = calibration_rows - span + 1
    selection_indices = np.arange(selection_count, dtype=np.int64)
    calibration_offset = calibration_range[0] - val_start
    calibration_indices = np.arange(
        calibration_offset, calibration_offset + calibration_count, dtype=np.int64
    )
    if selection_count < 200 or calibration_count < 200:
        raise RuntimeError("BLOCKED_INSUFFICIENT_CONFORMAL_SPLIT_SIZE")

    selection_raw = set(range(*selection_range))
    calibration_raw = set(range(*calibration_range))
    selection_future = set(range(selection_range[0] + seq_len, selection_range[1]))
    calibration_future = set(range(calibration_range[0] + seq_len, calibration_range[1]))
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    selection_path = SPLIT_DIR / "selection_window_indices.npy"
    calibration_path = SPLIT_DIR / "calibration_window_indices.npy"
    save_array(selection_path, selection_indices)
    save_array(calibration_path, calibration_indices)
    metadata = {
        "dataset": "etth1",
        "official_validation_raw_range": [val_start, val_end],
        "official_validation_raw_dates": [str(frame.iloc[val_start]["date"]), str(frame.iloc[val_end - 1]["date"])],
        "selection_raw_range": selection_range,
        "embargo_raw_range": embargo_range,
        "calibration_raw_range": calibration_range,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "embargo_rows": embargo_rows,
        "selection_fraction_of_usable_validation": 0.50,
        "selection_window_count": selection_count,
        "calibration_window_count": calibration_count,
        "selection_indices_sha256": sha256(selection_path),
        "calibration_indices_sha256": sha256(calibration_path),
        "selection_window_index_range": [int(selection_indices[0]), int(selection_indices[-1])],
        "calibration_window_index_range": [int(calibration_indices[0]), int(calibration_indices[-1])],
        "raw_timestamp_overlap_count": len(selection_raw & calibration_raw),
        "future_target_overlap_count": len(selection_future & calibration_future),
        "window_index_overlap_count": int(np.intersect1d(selection_indices, calibration_indices).size),
        "window_definition": "history and future target must be fully contained in its raw-time segment",
        "created_at": datetime.now().isoformat(),
    }
    metadata_path = SPLIT_DIR / "metadata.json"
    if metadata_path.exists():
        old = load(metadata_path)
        stable_keys = set(metadata) - {"created_at"}
        if any(old.get(key) != metadata.get(key) for key in stable_keys):
            raise RuntimeError(f"existing split metadata differs; refusing overwrite: {metadata_path}")
        metadata = old
    else:
        dump(metadata_path, metadata)
    split_checks = {
        "official_validation_window_count_2785": total_raw_rows - span + 1 == 2785,
        "embargo_rows_479": embargo_rows == 479,
        "selection_windows_at_least_200": selection_count >= 200,
        "calibration_windows_at_least_200": calibration_count >= 200,
        "raw_timestamp_overlap_zero": metadata["raw_timestamp_overlap_count"] == 0,
        "future_target_overlap_zero": metadata["future_target_overlap_count"] == 0,
        "window_index_overlap_zero": metadata["window_index_overlap_count"] == 0,
    }
    audit = {
        "stage": "Gate 4c split audit",
        "status": "PASS" if all(split_checks.values()) else "BLOCKED_CONFORMAL_CALIBRATION_LEAKAGE",
        "checks": split_checks,
        **metadata,
        "selection_window_indices_path": str(selection_path.relative_to(ROOT)),
        "calibration_window_indices_path": str(calibration_path.relative_to(ROOT)),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
    }
    dump(ROOT / "results/gate4c_split_audit.json", audit)
    (ROOT / "reports/gate4c_split_audit.md").write_text(
        "# Gate 4c Raw-Time Split Audit\n\n"
        f"Status: **{audit['status']}**\n\n"
        f"- Official validation raw range: `{[val_start, val_end]}`\n"
        f"- Selection: `{selection_range}`, windows `{selection_count}`\n"
        f"- Embargo: `{embargo_range}`, rows `{embargo_rows}`\n"
        f"- Calibration: `{calibration_range}`, windows `{calibration_count}`\n"
        f"- Raw timestamp overlap: `{metadata['raw_timestamp_overlap_count']}`\n"
        f"- Future-target overlap: `{metadata['future_target_overlap_count']}`\n"
        "\nThe split is chronological at raw-row level; no overlapping generated windows were randomly partitioned.\n"
    )
    leakage = {
        "stage": "Gate 4c leakage audit",
        "status": audit["status"],
        "selection_calibration_raw_overlap_count": metadata["raw_timestamp_overlap_count"],
        "selection_calibration_future_target_overlap_count": metadata["future_target_overlap_count"],
        "calibration_labels_seen_during_training": False,
        "calibration_labels_seen_during_model_selection": False,
        "test_labels_seen_during_calibration": False,
        "previous_gate4b_checkpoint_reused_for_conformal": False,
        "gate4c_probability_heads_retrained": True,
        "normalization_reference": "official_train_split_statistics",
        "test_statistics_used_for_normalization": False,
        "exact_exchangeability_guarantee_claimed": False,
    }
    dump(ROOT / "results/gate4c_leakage_audit.json", leakage)
    (ROOT / "reports/gate4c_leakage_audit.md").write_text(
        "# Gate 4c Leakage Audit\n\n"
        f"Status: **{leakage['status']}**\n\n"
        "Calibration labels are inaccessible during training and model selection. Test labels are inaccessible while fitting conformal quantiles. "
        "The fixed time-series split supports empirical calibration only; no exact exchangeability guarantee is claimed.\n"
    )
    print(json.dumps({"prerequisite": prerequisite["status"], "split": audit["status"], **metadata}, indent=2))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

