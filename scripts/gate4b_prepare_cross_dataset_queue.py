#!/usr/bin/env python
"""Create the formal four-task Gate 4b cross-dataset queue after smoke PASS."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def main():
    smoke = load(ROOT / "results/gate4b_cross_dataset_protocol_smoke.json")
    if smoke.get("status") != "PASS" or smoke.get("passed") != 4:
        raise RuntimeError("BLOCKED_CROSS_DATASET_PROTOCOL_SMOKE_FAILURE")
    reuse = load(ROOT / "results/gate4b_cross_dataset_reuse_inventory.json")
    reuse_map = {(row["dataset"], row["variant"]): row for row in reuse["rows"]}
    peaks = {dataset: max(row["peak_gpu_memory_mib"] for row in smoke["rows"] if row["dataset"] == dataset)
             for dataset in ("weather", "electricity")}
    required = {
        "weather": max(7000, math.ceil(peaks["weather"] * 1.5) + 512),
        "electricity": max(10000, math.ceil(peaks["electricity"] * 1.5) + 512),
    }
    tasks = []
    for dataset in ("weather", "electricity"):
        for variant in ("gaussian_head", "student_t_head"):
            path = ROOT / f"configs/gate4b_cross_dataset/{dataset}_{variant}_random_rate0.3_seed2025.json"
            task = load(path)
            task["required_free_mib"] = required[dataset]
            task["task_id"] = f"{dataset}__random_rate0.3__{variant}"
            task["command"] = ["gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"]
            task["reuse_run_dir"] = reuse_map[(dataset, variant)]["run_dir"]
            path.write_text(json.dumps(task, indent=2) + "\n")
            tasks.append(task)
    reused = sum(bool(task["reuse_run_dir"]) for task in tasks)
    queue = {
        "stage": "Gate 4b cross-dataset", "schema_version": 4,
        "dataset_count": 2, "logical_task_count": 4, "result_row_count_target": 6,
        "reused_task_count": reused, "active_task_count": 4 - reused,
        "max_concurrent_total": 1, "experiment_stage_label": "gate4b_cross_dataset",
        "state_file": "runtime/gate4b_cross_dataset_scheduler_state.json",
        "lock_file": "runtime/locks/gate4b_cross_dataset_scheduler.lock",
        "dataset_required_free_mib": required, "tasks": tasks,
        "official_baseline_retrained": False, "gate4a_adapter": False,
        "missingness_film": False, "conformal": False, "calibration_fitting": False,
        "meta_learning": False, "test_time_adaptation": False, "gate4c": False,
        "etth1_active_task": False, "extra_missing_types": False, "extra_rates": False, "extra_seeds": False,
    }
    output = ROOT / "configs/queues/gate4b_cross_dataset.yaml"
    output.write_text(json.dumps(queue, indent=2) + "\n")
    status_path = ROOT / "results/gate4b_cross_dataset_status.json"
    status = load(status_path)
    status.update({"status": "QUEUE_CREATED_PENDING_VALIDATION", "queue": str(output.relative_to(ROOT)),
                   "reused_tasks": reused, "active_tasks": 4 - reused,
                   "required_free_mib": required, "scheduler": "NOT_STARTED"})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "queue": str(output), "logical_tasks": 4,
                      "reused": reused, "active": 4 - reused, "required_free_mib": required}, indent=2))


if __name__ == "__main__":
    main()
