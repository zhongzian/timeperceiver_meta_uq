#!/usr/bin/env python3
"""Create sequential Gate 5b queues after backbone smokes pass."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import (  # noqa: E402
    DATASETS,
    RESOURCE_FLOORS,
    backbone_config,
    dump,
    load,
    probability_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--electricity-no-floor-user-override", action="store_true")
    cli = parser.parse_args()
    smoke = load(ROOT / "results/gate5b_backbone_smoke.json")
    if smoke.get("status") != "PASS" or smoke.get("backbone_smoke_pass") != "3/3":
        raise SystemExit("Gate 5b0 backbone smoke is not 3/3 PASS")
    smoke_rows = {row["dataset"]: row for row in smoke["datasets"]}
    required = {}
    effective_floors = dict(RESOURCE_FLOORS)
    if cli.electricity_no_floor_user_override:
        effective_floors["electricity"] = 0
    for dataset in DATASETS:
        peak = max(float(smoke_rows[dataset]["peak_allocated_mib"]), float(smoke_rows[dataset]["peak_reserved_mib"]))
        required[dataset] = max(effective_floors[dataset], math.ceil(peak * 1.5) + 1024)
    task_order = ("electricity", "etth1", "weather") if cli.electricity_no_floor_user_override else DATASETS
    backbone_tasks = []
    for dataset in task_order:
        task = backbone_config(dataset, smoke=False)
        task["required_free_mib"] = required[dataset]
        task["resource_floor_original_mib"] = RESOURCE_FLOORS[dataset]
        task["resource_floor_effective_mib"] = effective_floors[dataset]
        task["resource_floor_user_override"] = bool(
            dataset == "electricity" and cli.electricity_no_floor_user_override
        )
        task["task_id"] = f"{dataset}__patchtst_clean_seed2025"
        task["reuse_allowed"] = True
        backbone_tasks.append(task)
    backbone_queue = {
        "stage": "Gate 5b0 PatchTST Clean Backbones", "schema_version": 1,
        "logical_task_count": 3, "active_task_count": 3, "max_concurrent_total": 2,
        "max_tasks_per_gpu": 1, "bounded_oom_retry": 3,
        "experiment_stage_label": "gate5b0_backbone", "launch_enabled": True,
        "state_file": "runtime/gate5b_patchtst_backbones_scheduler_state.json",
        "pid_file": "runtime/gate5b_patchtst_backbones_scheduler.pid",
        "lock_file": "runtime/locks/gate5b_patchtst_backbones_scheduler.lock",
        "stop_file": "runtime/gate5b_patchtst_backbones_scheduler.stop",
        "resource_formula": "max(effective_dataset_floor, ceil(max(smoke_allocated,smoke_reserved)*1.5)+1024)",
        "dataset_original_floor_mib": RESOURCE_FLOORS,
        "dataset_effective_floor_mib": effective_floors,
        "electricity_floor_override_authorized_by_user": cli.electricity_no_floor_user_override,
        "scheduling_order": list(task_order),
        "dataset_required_free_mib": required, "tasks": backbone_tasks,
    }
    probability_tasks = []
    for dataset in task_order:
        for distribution in ("gaussian", "fixed_t8"):
            task = probability_config(dataset, distribution, smoke=False)
            task["required_free_mib"] = required[dataset]
            task["resource_floor_original_mib"] = RESOURCE_FLOORS[dataset]
            task["resource_floor_effective_mib"] = effective_floors[dataset]
            task["resource_floor_user_override"] = bool(
                dataset == "electricity" and cli.electricity_no_floor_user_override
            )
            task["task_id"] = f"{dataset}__{distribution}__random_rate0.3_seed2025"
            task["reuse_allowed"] = False
            probability_tasks.append(task)
    probability_queue = {
        "stage": "Gate 5b1 PatchTST Probability Replication", "schema_version": 1,
        "logical_task_count": 6, "active_task_count": 6, "max_concurrent_total": 2,
        "max_tasks_per_gpu": 1, "bounded_oom_retry": 3,
        "experiment_stage_label": "gate5b1_probability", "launch_enabled": False,
        "blocked_reason": "PENDING_GATE5B0_MANIFEST_DISTRIBUTION_PAIRING_AND_PROBABILITY_SMOKE",
        "state_file": "runtime/gate5b_patchtst_probability_scheduler_state.json",
        "pid_file": "runtime/gate5b_patchtst_probability_scheduler.pid",
        "lock_file": "runtime/locks/gate5b_patchtst_probability_scheduler.lock",
        "stop_file": "runtime/gate5b_patchtst_probability_scheduler.stop",
        "resource_formula": "recomputed after Gate 5b1 probability smoke 6/6",
        "dataset_original_floor_mib": RESOURCE_FLOORS,
        "dataset_effective_floor_mib": effective_floors,
        "electricity_floor_override_authorized_by_user": cli.electricity_no_floor_user_override,
        "scheduling_order": list(task_order),
        "dataset_required_free_mib": required, "tasks": probability_tasks,
    }
    dump(ROOT / "configs/queues/gate5b_patchtst_backbones.yaml", backbone_queue)
    dump(ROOT / "configs/queues/gate5b_patchtst_probability.yaml", probability_queue)
    print(json.dumps({"status": "PASS", "backbone_required_free_mib": required,
                      "probability_queue_launch_enabled": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
