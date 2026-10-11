#!/usr/bin/env python
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_minimal_probabilistic.yaml"
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"


def main():
    queue = json.loads(QUEUE.read_text())
    tasks, errors = queue.get("tasks", []), []
    if queue.get("logical_task_count") != 2:
        errors.append("logical_task_count != 2")
    if not 0 < queue.get("active_task_count", 0) <= 2:
        errors.append("active_task_count outside 1..2")
    if {task.get("variant") for task in tasks} != {"gaussian_head", "student_t_head"}:
        errors.append("variant set mismatch")
    if queue.get("required_free_mib", 0) < 4096:
        errors.append("required_free_mib below 4096")
    if len({task.get("training_protocol_hash") for task in tasks}) != 1:
        errors.append("training protocol hashes differ")
    if len({task.get("metric_protocol_hash") for task in tasks}) != 1:
        errors.append("metric protocol hashes differ")
    for index, task in enumerate(tasks):
        task_file = ROOT / "runtime" / f"gate4b_validate_task_{index}.json"
        task_file.parent.mkdir(exist_ok=True)
        task_file.write_text(json.dumps(task, indent=2) + "\n")
        result = subprocess.run([PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"),
                                 "--task-file", str(task_file), "--validate-only"],
                                text=True, capture_output=True)
        if result.returncode:
            errors.append(f"{task.get('variant')}: {result.stdout.strip()}")
        expected = ["gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"]
        if task.get("command") != expected:
            errors.append(f"{task.get('variant')}: executable command mismatch")
        if "smoke" in json.dumps(task).lower():
            errors.append(f"{task.get('variant')}: smoke marker present")
    for key in ("official_baseline_retrained", "gate4a_adapter", "missingness_film", "conformal",
                "calibration_fitting", "meta_learning", "test_time_adaptation", "gate4c"):
        if queue.get(key):
            errors.append(f"forbidden enabled: {key}")
    status = "PASS" if not errors else "BLOCKED_FORMAL_QUEUE_VALIDATION_FAILURE"
    output = {"status": status, "queue": str(QUEUE), "logical_tasks": queue.get("logical_task_count"),
              "active_tasks": queue.get("active_task_count"), "required_free_mib": queue.get("required_free_mib"),
              "training_protocol_hash": queue.get("training_protocol_hash"),
              "metric_protocol_hash": queue.get("metric_protocol_hash"), "errors": errors}
    (ROOT / "results/gate4b_formal_queue_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4b_formal_queue_validation.md").write_text(
        f"# Gate 4b Formal Queue Validation\n\nStatus: **{status}**\n\nErrors: {errors or 'none'}\n"
    )
    print(json.dumps(output, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
