#!/usr/bin/env python
"""Validate one Gate 4c formal task, then invoke the leakage-safe trainer."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate(task):
    errors = []
    exact = {
        "stage": "gate4c", "dataset": "etth1", "missing_type": "random", "missing_rate": 0.3,
        "seed": 2025, "experiment_seed": 2025, "seq_len": 384, "pred_len": 96,
        "train_split": "official_train", "train_mask_base_seed": 42025,
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 0.001,
        "weight_decay": 0.0001, "gradient_clip": 1.0, "max_epochs": 20,
        "patience": 3, "early_stopping_metric": "selection_nll",
        "calibration_used_for_selection": False, "test_used_for_selection": False,
        "retrain_probability_head": True, "reuse_gate4b_probability_checkpoint": False,
        "condition_source": "historical_input_mask_only", "condition_bins": 3,
        "bin_boundaries_fit_split": "calibration_masks_only", "smoke_only": False,
    }
    for key, value in exact.items():
        if task.get(key) != value:
            errors.append(f"{key}: expected {value!r}, got {task.get(key)!r}")
    if task.get("variant") not in {"gaussian_head", "student_t_head"}:
        errors.append("variant must be gaussian_head or student_t_head")
    if list(map(float, task.get("coverage_levels", []))) != [0.9, 0.95]:
        errors.append("coverage_levels must be [0.90, 0.95]")
    if task.get("conformal_methods") != ["marginal", "mask_severity_conditioned"]:
        errors.append("conformal_methods mismatch")
    for key in ("max_train_batches", "max_selection_batches", "max_calibration_batches", "max_test_batches"):
        if task.get(key) is not None:
            errors.append(f"{key} must be null/unlimited")
    path_hash_keys = {
        "source_checkpoint": "source_checkpoint_sha256",
        "selection_split_path": "selection_split_sha256",
        "calibration_split_path": "calibration_split_sha256",
        "selection_mask_path": "selection_mask_sha256",
        "calibration_mask_path": "calibration_mask_sha256",
        "test_mask_path": "test_mask_sha256",
    }
    for path_key, hash_key in path_hash_keys.items():
        path = resolve(task.get(path_key, ""))
        if not path.is_file():
            errors.append(f"missing {path_key}: {path}")
        elif task.get(hash_key) != sha256(path):
            errors.append(f"hash mismatch for {path_key}")
    split = load(ROOT / "results/gate4c_split_audit.json")
    if split.get("status") != "PASS" or split.get("raw_timestamp_overlap_count") != 0:
        errors.append("split audit is not leakage-safe")
    if split.get("future_target_overlap_count") != 0 or split.get("embargo_rows") != 479:
        errors.append("split future overlap or embargo mismatch")
    if int(task.get("required_free_mib", 0)) < 4096:
        errors.append("required_free_mib below 4096")
    forbidden_true = (
        "gate4a_adapter", "film", "online_conformal", "adaptive_conformal",
        "test_time_adaptation", "official_baseline_retrained",
    )
    for key in forbidden_true:
        if task.get(key) is True:
            errors.append(f"forbidden option enabled: {key}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    task = load(args.task_file)
    errors = validate(task)
    if args.validate_only:
        print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
        return 0 if not errors else 2
    if errors:
        print(json.dumps({"status": "CONFIG_ERROR", "errors": errors}, indent=2))
        return 2
    if not args.run_dir:
        raise SystemExit("--run-dir is required")
    command = [
        sys.executable, "-u", str(ROOT / "scripts/gate4c_train_and_calibrate.py"),
        "--task-file", args.task_file, "--run-dir", args.run_dir, "--device", "cuda",
    ]
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

