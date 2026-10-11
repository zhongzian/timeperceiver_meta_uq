#!/usr/bin/env python3
"""Strict static and runner validation for both sequential Gate 5b queues."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import DATASETS, PYTHON, dump, load, write_report  # noqa: E402


def runner_validate(script, task, allow_placeholder=False):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=ROOT / "runtime") as handle:
        json.dump(task, handle)
        path = Path(handle.name)
    command = [str(PYTHON), str(ROOT / "scripts" / script), "--task-file", str(path), "--validate-only"]
    if allow_placeholder:
        command.append("--allow-manifest-placeholder")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path.unlink(missing_ok=True)
    return {"pass": completed.returncode == 0, "exit_code": completed.returncode, "output": completed.stdout.strip()}


def validate_backbone(queue):
    errors = []
    tasks = queue.get("tasks", [])
    if len(tasks) != 3 or queue.get("logical_task_count") != 3:
        errors.append("backbone logical task count must be 3")
    if {task.get("dataset") for task in tasks} != set(DATASETS):
        errors.append("backbone datasets are not exactly ETTh1/Weather/Electricity")
    runner_rows = []
    for task in tasks:
        expected = {
            "backbone": "patchtst", "missingness": "none", "experiment_seed": 2025,
            "seq_len": 384, "pred_len": 96, "test_used_for_model_selection": False,
            "smoke_only": False,
        }
        errors.extend(f"{task.get('task_id')} {key} mismatch" for key, value in expected.items() if task.get(key) != value)
        runner_rows.append({"task_id": task.get("task_id"), **runner_validate("gate5b_train_backbone.py", task)})
    if not all(row["pass"] for row in runner_rows):
        errors.append("one or more backbone --validate-only commands failed")
    return errors, runner_rows


def validate_probability(queue):
    errors = []
    tasks = queue.get("tasks", [])
    if len(tasks) != 6 or queue.get("logical_task_count") != 6:
        errors.append("probability logical task count must be 6")
    expected_pairs = {(dataset, distribution) for dataset in DATASETS for distribution in ("gaussian", "fixed_t8")}
    if {(task.get("dataset"), task.get("distribution")) for task in tasks} != expected_pairs:
        errors.append("probability dataset/head matrix is not exactly 3x2")
    runner_rows = []
    for task in tasks:
        expected = {
            "backbone": "patchtst", "missing_type": "random", "missing_rate": 0.3,
            "experiment_seed": 2025, "seq_len": 384, "pred_len": 96,
            "freeze_backbone": True, "test_used_for_model_selection": False,
            "learnable_df": False, "conformal": False, "posthoc_calibration": False,
            "smoke_only": False,
        }
        errors.extend(f"{task.get('task_id')} {key} mismatch" for key, value in expected.items() if task.get(key) != value)
        if task.get("distribution") == "fixed_t8" and task.get("fixed_t_df") != 8.0:
            errors.append(f"{task.get('task_id')} fixed df mismatch")
        prohibited = ("laplace", "adapter", "conformal", "learnable_df")
        if any(token in str(task.get("variant", "")).lower() for token in prohibited[:3]):
            errors.append(f"{task.get('task_id')} prohibited variant")
        runner_rows.append({
            "task_id": task.get("task_id"),
            **runner_validate("gate5b_train_probability.py", task, allow_placeholder=True),
        })
    if not all(row["pass"] for row in runner_rows):
        errors.append("one or more probability --validate-only commands failed")
    return errors, runner_rows


def main():
    backbone_path = ROOT / "configs/queues/gate5b_patchtst_backbones.yaml"
    probability_path = ROOT / "configs/queues/gate5b_patchtst_probability.yaml"
    if not backbone_path.is_file() or not probability_path.is_file():
        payload = {"status": "BLOCKED_GATE5B_QUEUE_VALIDATION", "error": "queue files are missing"}
        dump(ROOT / "results/gate5b_queue_validation.json", payload)
        write_report(ROOT / "reports/gate5b_queue_validation.md", "Gate 5b Queue Validation", payload)
        print(json.dumps(payload, indent=2))
        return 2
    backbone = load(backbone_path)
    probability = load(probability_path)
    backbone_errors, backbone_runners = validate_backbone(backbone)
    probability_errors, probability_runners = validate_probability(probability)
    placeholder_pending = all(
        task["source_backbone_checkpoint"] == "__FROM_GATE5B0_MANIFEST__" for task in probability["tasks"]
    )
    manifest = load(ROOT / "results/gate5b_patchtst_backbone_manifest.json") \
        if (ROOT / "results/gate5b_patchtst_backbone_manifest.json").is_file() else {}
    distribution = load(ROOT / "results/gate5b_distribution_validation.json") \
        if (ROOT / "results/gate5b_distribution_validation.json").is_file() else {}
    pairing = load(ROOT / "results/gate5b_pairing_validation.json") \
        if (ROOT / "results/gate5b_pairing_validation.json").is_file() else {}
    smoke = load(ROOT / "results/gate5b_probability_smoke.json") \
        if (ROOT / "results/gate5b_probability_smoke.json").is_file() else {}
    probability_prerequisites = all((
        manifest.get("status") == "PASS", distribution.get("status") == "PASS",
        pairing.get("status") == "PASS", pairing.get("pairs_pass") == "3/3",
        smoke.get("status") == "PASS", smoke.get("probability_smoke_pass") == "6/6",
        not placeholder_pending,
    ))
    launch_flag_valid = probability.get("launch_enabled") is False or probability_prerequisites
    passed = not backbone_errors and not probability_errors and launch_flag_valid
    payload = {
        "status": "PASS" if passed else "BLOCKED_GATE5B_QUEUE_VALIDATION",
        "backbone_queue_validation": "PASS" if not backbone_errors else "FAIL",
        "backbone_launch_ready": not backbone_errors and backbone.get("launch_enabled") is True,
        "backbone_errors": backbone_errors, "backbone_runner_validation": backbone_runners,
        "probability_queue_structural_validation": "PASS" if not probability_errors else "FAIL",
        "probability_checkpoint_manifest": manifest.get("status", "NOT_RUN"),
        "probability_distribution_validation": distribution.get("status", "NOT_RUN"),
        "probability_pairing_validation": (
            "PENDING_GATE5B0_MANIFEST" if placeholder_pending else pairing.get("status", "NOT_RUN")
        ),
        "probability_smoke_validation": smoke.get("status", "NOT_RUN"),
        "probability_prerequisites_pass": probability_prerequisites,
        "probability_launch_ready": (
            not probability_errors and probability_prerequisites and probability.get("launch_enabled") is True
        ),
        "probability_errors": probability_errors,
        "probability_runner_validation": probability_runners,
        "probability_queue_launch_enabled": probability.get("launch_enabled"),
        "sequential_queues": True, "queues_concurrently_started": False,
    }
    dump(ROOT / "results/gate5b_queue_validation.json", payload)
    write_report(ROOT / "reports/gate5b_queue_validation.md", "Gate 5b Queue Validation", payload)
    print(json.dumps({"status": payload["status"], "backbone": payload["backbone_queue_validation"],
                      "probability": payload["probability_queue_structural_validation"],
                      "probability_launch_ready": payload["probability_launch_ready"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
