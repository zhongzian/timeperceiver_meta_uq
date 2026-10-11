#!/usr/bin/env python
"""Validate a Gate 4b task, then invoke the shared probabilistic trainer."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {("random", 0.1), ("random", 0.3), ("random", 0.5),
              ("block", 0.3), ("channel", 0.3), ("mixed", 0.3)}
DATASETS = {"etth1", "weather", "electricity"}


def load(path):
    return json.loads(Path(path).read_text())


def resolve(task, allow_smoke=False):
    errors = []
    dataset = str(task.get("dataset", "")).lower()
    stage = str(task.get("stage", ""))
    multiseed = stage.startswith("gate4b_multiseed")
    gate5a = stage.startswith("gate5a")
    expected = {
        "seq_len": 384, "pred_len": 96,
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "gradient_clip": 1.0, "patience": 3,
        "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
    }
    for key, value in expected.items():
        if task.get(key) != value:
            errors.append(f"{key}: expected {value!r}, got {task.get(key)!r}")
    experiment_seed = int(task.get("experiment_seed", task.get("seed", -1)))
    allowed_seeds = {2021, 2022, 2023, 2025} if multiseed else {2025}
    if task.get("seed") not in allowed_seeds or experiment_seed != task.get("seed"):
        errors.append(
            f"seed protocol mismatch: seed={task.get('seed')}, experiment_seed={experiment_seed}"
        )
    if multiseed and task.get("train_mask_base_seed") != 40000 + experiment_seed:
        errors.append("multi-seed train_mask_base_seed must equal 40000 + experiment_seed")
    if multiseed and task.get("backbone_seed_scope") != "fixed_official_checkpoint":
        errors.append("multi-seed backbone_seed_scope mismatch")
    if dataset not in DATASETS:
        errors.append(f"invalid dataset: {dataset}")
    condition = (task.get("missing_type"), float(task.get("missing_rate", -1)))
    allowed_conditions = {("random", 0.3)} if gate5a else CONDITIONS
    if condition not in allowed_conditions:
        errors.append(f"invalid condition: {condition}")
    allowed_variants = (
        {"gaussian_head", "laplace_head", "student_t_fixed_df8_head", "student_t_head"}
        if gate5a else {"gaussian_head", "student_t_head"}
    )
    if task.get("variant") not in allowed_variants:
        errors.append("invalid variant")
    if gate5a:
        if task.get("train_mask_base_seed") != 42025:
            errors.append("Gate 5a train_mask_base_seed must be 42025")
        if task.get("initial_scale") != 0.5 or task.get("scale_floor") != 1e-4:
            errors.append("Gate 5a scale initialization protocol mismatch")
        if task.get("variant") == "student_t_fixed_df8_head" and task.get("fixed_t_df") != 8.0:
            errors.append("Gate 5a fixed Student-t must use df=8.0")
        if task.get("use_gate4a_adapter") or task.get("use_second_backbone"):
            errors.append("Gate 5a forbids Gate 4a adapters and a second backbone")
    smoke = bool(task.get("smoke_only"))
    if smoke and not allow_smoke:
        errors.append("smoke task not allowed")
    if smoke:
        if task.get("max_epochs") != 1 or task.get("max_train_batches") != 4:
            errors.append("protocol smoke train limits mismatch")
        if task.get("max_val_batches") != 2 or task.get("max_test_batches") != 2:
            errors.append("protocol smoke evaluation limits mismatch")
    else:
        electricity_floor = 18000 if multiseed or gate5a else 10000
        minimum_memory = {"etth1": 4096, "weather": 7000, "electricity": electricity_floor}.get(
            dataset, 10**9
        )
        if task.get("max_epochs") != 20 or int(task.get("required_free_mib", 0)) < minimum_memory:
            errors.append("formal epoch/resource protocol mismatch")
        for key in ("max_train_batches", "max_val_batches", "max_test_batches"):
            if task.get(key) is not None:
                errors.append(f"{key} must be null/unlimited")
    for key in ("source_checkpoint", "validation_mask", "test_mask", "config"):
        if not task.get(key) or not (ROOT / task[key]).is_file():
            errors.append(f"missing path {key}: {task.get(key)}")
    if task.get("conformal_calibrated") or task.get("posthoc_calibration"):
        errors.append("calibration is forbidden in Gate 4b")
    if not task.get("training_protocol_hash") or not task.get("metric_protocol_hash"):
        errors.append("protocol hashes required")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    task = load(args.task_file)
    errors = resolve(task, args.allow_smoke)
    if args.validate_only:
        print(json.dumps({"status": "PASS" if not errors else "FAIL", "errors": errors}, indent=2))
        return 0 if not errors else 2
    if errors:
        print(json.dumps({"status": "CONFIG_ERROR", "errors": errors}, indent=2))
        return 2
    if not args.run_dir:
        raise SystemExit("--run-dir required for execution")
    command = [sys.executable, "-u", str(ROOT / "scripts/gate4b_train_probabilistic.py"),
               "--task-file", args.task_file, "--run-dir", args.run_dir, "--device", "cuda"]
    env = os.environ.copy()
    env["GATE4B_TRAINING_PROTOCOL_HASH"] = task["training_protocol_hash"]
    env["GATE4B_METRIC_PROTOCOL_HASH"] = task["metric_protocol_hash"]
    return subprocess.call(command, env=env, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
