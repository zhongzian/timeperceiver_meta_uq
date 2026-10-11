#!/usr/bin/env python3
"""Strictly validate the Gate 5a queue without loading CUDA."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_scheduler import valid_run
from gate5a_common import (
    DATASETS,
    HEADS,
    RESOURCE_FLOORS,
    corrected_peak_mib,
    dump,
    load,
    write_json_markdown,
)


PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
QUEUE = ROOT / "configs/queues/gate5a_distribution_ablation.yaml"


def main():
    queue = load(QUEUE)
    tasks = queue.get("tasks", [])
    smoke = load(ROOT / "results/gate5a_protocol_smoke.json")
    smoke_peaks = {
        dataset: max(float(row["peak_gpu_memory_mib"]) for row in smoke["rows"] if row["dataset"] == dataset)
        for dataset in DATASETS
    }
    expected_required = {
        dataset: max(
            RESOURCE_FLOORS[dataset],
            math.ceil(max(smoke_peaks[dataset], corrected_peak_mib(dataset)) * 1.5) + 512,
        )
        for dataset in DATASETS
    }
    task_pairs = Counter((task.get("dataset"), task.get("variant")) for task in tasks)
    reused_tasks = [task for task in tasks if task.get("reuse_allowed") is True]
    active_tasks = [task for task in tasks if task.get("reuse_allowed") is False]
    common_checks = {
        "datasets_exact": set(task.get("dataset") for task in tasks) == set(DATASETS),
        "heads_exact": set(task.get("variant") for task in tasks) == set(HEADS),
        "each_dataset_head_once": all(task_pairs[(dataset, head)] == 1 for dataset in DATASETS for head in HEADS),
        "logical_tasks_12": len(tasks) == queue.get("logical_task_count") == 12,
        "reused_6": len(reused_tasks) == queue.get("reused_task_count") == 6,
        "active_6": len(active_tasks) == queue.get("active_task_count") == 6,
        "random_rate_0_3_only": all(task.get("missing_type") == "random" and task.get("missing_rate") == 0.3 for task in tasks),
        "seed_2025_only": all(task.get("seed") == 2025 and task.get("experiment_seed") == 2025 for task in tasks),
        "train_mask_base_seed_42025": all(task.get("train_mask_base_seed") == 42025 for task in tasks),
        "frozen_backbone": all(task.get("freeze_backbone") is True for task in tasks),
        "full_training": all(
            task.get("max_epochs") == 20
            and task.get("max_train_batches") is None
            and task.get("max_val_batches") is None
            and task.get("max_test_batches") is None
            and task.get("smoke_only") is False
            for task in tasks
        ),
        "optimization_protocol": all(
            task.get("optimizer") == "AdamW"
            and task.get("learning_rate") == 1e-3
            and task.get("weight_decay") == 1e-4
            and task.get("gradient_clip") == 1.0
            and task.get("patience") == 3
            and task.get("early_stopping_metric") == "validation_nll"
            and task.get("test_used_for_model_selection") is False
            for task in tasks
        ),
        "distribution_initialization": all(
            task.get("initial_scale") == 0.5
            and task.get("scale_floor") == 1e-4
            and (task.get("variant") != "student_t_fixed_df8_head" or task.get("fixed_t_df") == 8.0)
            for task in tasks
        ),
        "fixed_assets_present": all(
            (ROOT / task["source_checkpoint"]).is_file()
            and (ROOT / task["validation_mask"]).is_file()
            and (ROOT / task["test_mask"]).is_file()
            for task in tasks
        ),
        "no_forbidden_modules": all(
            task.get("use_gate4a_adapter") is False
            and task.get("conformal_calibrated") is False
            and task.get("posthoc_calibration") is False
            and task.get("use_second_backbone") is False
            and task.get("official_backbone_retrained") is False
            for task in tasks
        ),
        "resource_formula_exact": queue.get("dataset_required_free_mib") == expected_required,
        "resource_floors_respected": all(
            int(task.get("required_free_mib", 0)) >= RESOURCE_FLOORS[task["dataset"]] for task in tasks
        ),
        "one_task_per_gpu": queue.get("max_tasks_per_gpu") == 1,
        "bounded_concurrency": queue.get("max_concurrent_total") == 2,
        "old_unmasked_reuse_count_zero": queue.get("old_unmasked_reuse_count") == 0,
    }
    reused_rows = []
    for task in reused_tasks:
        run = ROOT / task["reuse_run_dir"]
        passed = valid_run(run, task)
        reused_rows.append({
            "task_id": task["task_id"], "run_dir": task["reuse_run_dir"],
            "status": "PASS" if passed else "FAIL",
        })
    active_rows = []
    for task in active_tasks:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir="/tmp") as handle:
            json.dump(task, handle)
            task_path = Path(handle.name)
        result = subprocess.run(
            [PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"), "--task-file", str(task_path), "--validate-only"],
            cwd=str(ROOT), text=True, capture_output=True,
        )
        task_path.unlink(missing_ok=True)
        active_rows.append({
            "task_id": task["task_id"], "status": "PASS" if result.returncode == 0 else "FAIL",
            "return_code": result.returncode, "output": result.stdout.strip(), "stderr": result.stderr.strip(),
        })
    checks = {
        **common_checks,
        "all_reuse_valid": len(reused_rows) == 6 and all(row["status"] == "PASS" for row in reused_rows),
        "all_active_validate_only_pass": len(active_rows) == 6 and all(row["status"] == "PASS" for row in active_rows),
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "BLOCKED_GATE5A_QUEUE_VALIDATION_FAILURE",
        "queue": str(QUEUE.relative_to(ROOT)),
        "logical_tasks": len(tasks),
        "reused_tasks": len(reused_tasks),
        "active_tasks": len(active_tasks),
        "required_free_mib": queue.get("dataset_required_free_mib"),
        "expected_required_free_mib": expected_required,
        "checks": checks,
        "reused_validation": reused_rows,
        "active_validate_only": active_rows,
    }
    dump(ROOT / "results/gate5a_queue_validation.json", payload)
    write_json_markdown(ROOT / "reports/gate5a_queue_validation.md", "Gate 5a Queue Validation", payload)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
