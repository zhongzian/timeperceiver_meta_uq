#!/usr/bin/env python
"""Create the 12-logical-task Gate 4b expanded ETTh1 queue."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (("random", 0.1), ("random", 0.3), ("random", 0.5),
              ("block", 0.3), ("channel", 0.3), ("mixed", 0.3))


def load(path):
    return json.loads(Path(path).read_text())


def main():
    prerequisite = load(ROOT / "results/gate4b_corrected_minimal_decision.json")
    smoke = load(ROOT / "results/gate4b_expanded_protocol_smoke.json")
    reuse = load(ROOT / "results/gate4b_expanded_reuse_inventory.json")
    if not prerequisite.get("ENTER_GATE4B_EXPANDED_MATRIX") or smoke.get("status") != "PASS":
        raise RuntimeError("expanded prerequisites have not passed")
    reuse_map = {(row["condition"], row["variant"]): row for row in reuse["rows"]}
    tasks = []
    for mask_type, rate in CONDITIONS:
        for variant in ("gaussian_head", "student_t_head"):
            path = ROOT / f"configs/gate4b_expanded/etth1_{mask_type}_rate{rate}_{variant}_seed2025.json"
            task = load(path)
            task["task_id"] = f"{mask_type}_rate{rate}__{variant}"
            row = reuse_map[(f"{mask_type}_rate{rate}", variant)]
            task["reuse_run_dir"] = row["run_dir"] if row["reusable"] else ""
            tasks.append(task)
    peaks = []
    for path in (ROOT / "runs").glob("timeperceiver__etth1__gate4b_corrected_minimal__*/metrics.json"):
        metrics = load(path)
        if metrics.get("peak_gpu_memory_mib") is not None:
            peaks.append(float(metrics["peak_gpu_memory_mib"]))
    required = max(4096, math.ceil(max(peaks) * 1.5) + 512)
    active = sum(not task["reuse_run_dir"] for task in tasks)
    queue = {
        "stage": "Gate 4b expanded ETTh1", "schema_version": 3,
        "condition_count": 6, "logical_task_count": 12, "result_row_count_target": 18,
        "reused_task_count": 12 - active, "active_task_count": active, "max_concurrent_total": 1,
        "required_free_mib": required, "experiment_stage_label": "gate4b_expanded",
        "state_file": "runtime/gate4b_expanded_etth1_scheduler_state.json",
        "lock_file": "runtime/locks/gate4b_expanded_etth1_scheduler.lock",
        "tasks": tasks, "official_baseline_retrained": False, "gate4a_adapter": False,
        "missingness_film": False, "conformal": False, "calibration_fitting": False,
        "meta_learning": False, "test_time_adaptation": False, "gate4c": False,
    }
    output = ROOT / "configs/queues/gate4b_expanded_etth1.yaml"
    output.write_text(json.dumps(queue, indent=2) + "\n")
    status_path = ROOT / "results/gate4b_expanded_status.json"
    status = load(status_path)
    status.update({"expanded_queue": "CREATED_PENDING_VALIDATION", "logical_tasks": 12,
                   "reused_tasks": 12 - active, "active_tasks": active,
                   "required_free_mib": required, "scheduler": "NOT_STARTED"})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "queue": str(output), "logical_tasks": 12,
                      "reused": 12 - active, "active": active, "required_free_mib": required}, indent=2))


if __name__ == "__main__":
    main()
