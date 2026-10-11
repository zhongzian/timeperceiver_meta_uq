#!/usr/bin/env python
"""Create the two-task corrected random-0.3 queue after the input-mask repair."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text())


def main():
    smoke = load(ROOT / "results/gate4b_expanded_protocol_smoke.json")
    if smoke.get("status") != "PASS":
        raise RuntimeError("corrected protocol smoke has not passed")
    tasks = []
    for variant in ("gaussian_head", "student_t_head"):
        path = ROOT / f"configs/gate4b_expanded/etth1_random_rate0.3_{variant}_seed2025.json"
        task = load(path)
        task["stage"] = "gate4b_corrected_minimal"
        task["task_id"] = f"random_rate0.3__{variant}"
        task["reuse_run_dir"] = ""
        tasks.append(task)
    queue = {
        "stage": "Gate 4b corrected minimal random 0.3", "schema_version": 3,
        "logical_task_count": 2, "active_task_count": 2, "max_concurrent_total": 1,
        "required_free_mib": 4096, "experiment_stage_label": "gate4b_corrected_minimal",
        "state_file": "runtime/gate4b_corrected_minimal_scheduler_state.json",
        "lock_file": "runtime/locks/gate4b_corrected_minimal_scheduler.lock",
        "tasks": tasks, "official_baseline_retrained": False, "gate4a_adapter": False,
        "conformal": False, "gate4c": False,
    }
    output = ROOT / "configs/queues/gate4b_corrected_minimal_random0.3.yaml"
    output.write_text(json.dumps(queue, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "queue": str(output), "tasks": 2}, indent=2))


if __name__ == "__main__":
    main()
