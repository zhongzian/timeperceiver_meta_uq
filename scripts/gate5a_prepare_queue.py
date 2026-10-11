#!/usr/bin/env python3
"""Create the Gate 5a logical queue only after all prerequisite gates pass."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import (
    DATASETS,
    HEADS,
    NEW_HEADS,
    RESOURCE_FLOORS,
    REUSED_HEADS,
    REUSE_RUNS,
    corrected_peak_mib,
    dump,
    load,
    task_config,
)


def require_pass(path, expected="PASS"):
    payload = load(ROOT / path)
    if payload.get("status") != expected:
        raise RuntimeError(f"prerequisite not PASS: {path}: {payload.get('status')}")
    return payload


def main():
    require_pass("results/gate5a_architecture_audit.json")
    require_pass("results/gate5a_distribution_validation.json")
    reuse = require_pass("results/gate5a_reuse_inventory.json")
    require_pass("results/gate5a_pairing_validation.json")
    smoke = require_pass("results/gate5a_protocol_smoke.json")
    if smoke.get("passed") != 6 or smoke.get("failed") != 0:
        raise RuntimeError("BLOCKED_GATE5A_PROTOCOL_SMOKE_FAILURE")
    reuse_map = {(row["dataset"], row["variant"]): row for row in reuse["rows"]}
    smoke_peaks = {
        dataset: max(
            float(row["peak_gpu_memory_mib"])
            for row in smoke["rows"]
            if row["dataset"] == dataset
        )
        for dataset in DATASETS
    }
    required = {
        dataset: max(
            RESOURCE_FLOORS[dataset],
            math.ceil(max(smoke_peaks[dataset], corrected_peak_mib(dataset)) * 1.5) + 512,
        )
        for dataset in DATASETS
    }
    tasks = []
    suffix = {"laplace_head": "laplace", "student_t_fixed_df8_head": "fixed_t8"}
    for dataset in DATASETS:
        for variant in HEADS:
            task = task_config(dataset, variant, smoke=False)
            task["required_free_mib"] = required[dataset]
            task["task_id"] = f"{dataset}__random_rate0.3__{variant}"
            if variant in REUSED_HEADS:
                row = reuse_map[(dataset, variant)]
                task.update({
                    "reuse_allowed": True,
                    "reuse_run_dir": row["run_dir"],
                    "reuse_schema_audit_pass": row["schema_audit_pass"],
                    "reuse_training_protocol_hash": row["training_protocol_hash"],
                    "reuse_metric_protocol_hash": row["metric_protocol_hash"],
                })
            else:
                task.update({"reuse_allowed": False, "reuse_run_dir": ""})
                config_path = ROOT / f"configs/gate5a/{dataset}_random0.3_{suffix[variant]}_seed2025.yaml"
                dump(config_path, task)
            tasks.append(task)
    queue = {
        "stage": "Gate 5a Distribution-Head Ablation",
        "schema_version": 1,
        "dataset_count": 3,
        "head_count": 4,
        "logical_task_count": 12,
        "reused_task_count": 6,
        "active_task_count": 6,
        "max_concurrent_total": 2,
        "max_tasks_per_gpu": 1,
        "experiment_stage_label": "gate5a",
        "state_file": "runtime/gate5a_distribution_ablation_scheduler_state.json",
        "pid_file": "runtime/gate5a_distribution_ablation_scheduler.pid",
        "lock_file": "runtime/locks/gate5a_distribution_ablation_scheduler.lock",
        "dataset_required_free_mib": required,
        "resource_formula": "max(dataset_floor, ceil(max(new_head_smoke_peak, corrected_formal_peak)*1.5)+512)",
        "dataset_smoke_peak_mib": smoke_peaks,
        "dataset_corrected_formal_peak_mib": {dataset: corrected_peak_mib(dataset) for dataset in DATASETS},
        "tasks": tasks,
        "official_baseline_retrained": False,
        "gate4a_adapter": False,
        "missingness_film": False,
        "conformal": False,
        "calibration_fitting": False,
        "meta_learning": False,
        "test_time_adaptation": False,
        "second_backbone": False,
        "extra_datasets": False,
        "extra_missing_types": False,
        "extra_rates": False,
        "extra_seeds": False,
        "old_unmasked_reuse_count": 0,
    }
    output = ROOT / "configs/queues/gate5a_distribution_ablation.yaml"
    dump(output, queue)
    status = load(ROOT / "results/gate5a_status.json")
    status.update({
        "status": "QUEUE_CREATED_PENDING_VALIDATION",
        "distribution_validation": "PASS",
        "protocol_smoke": "PASS",
        "formal_queue": "CREATED_PENDING_VALIDATION",
        "queue": str(output.relative_to(ROOT)),
        "logical_tasks": 12,
        "reused_tasks": 6,
        "active_tasks": 6,
        "required_free_mib": required,
        "scheduler": "NOT_STARTED",
    })
    dump(ROOT / "results/gate5a_status.json", status)
    print(json.dumps({
        "status": "PASS", "queue": str(output), "logical": 12, "reused": 6,
        "active": 6, "required_free_mib": required,
    }, indent=2))


if __name__ == "__main__":
    main()
