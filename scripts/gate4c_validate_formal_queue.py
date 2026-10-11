#!/usr/bin/env python
"""Dry-validate the two-task Gate 4c formal queue without loading a GPU."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4c_minimal_conformal.yaml"


def load(path):
    return json.loads(Path(path).read_text())


def main():
    queue = load(QUEUE)
    tasks = queue.get("tasks", [])
    errors = []
    if queue.get("logical_task_count") != 2 or len(tasks) != 2:
        errors.append("logical tasks must equal 2")
    if queue.get("active_task_count") != 2:
        errors.append("active tasks must equal 2")
    if queue.get("required_free_mib", 0) < 4096:
        errors.append("queue required_free_mib below 4096")
    if queue.get("allow_reuse") is not False:
        errors.append("Gate 4c formal queue must forbid reuse")
    if {task.get("variant") for task in tasks} != {"gaussian_head", "student_t_head"}:
        errors.append("variant set mismatch")
    task_results = []
    for task in tasks:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(task, handle)
            task_path = Path(handle.name)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gate4c_formal_runner.py"),
             "--task-file", str(task_path), "--validate-only"],
            cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        task_path.unlink(missing_ok=True)
        try:
            payload = json.loads(result.stdout)
        except Exception:
            payload = {"status": "FAIL", "errors": [result.stdout]}
        task_results.append({"task_id": task.get("task_id"), "return_code": result.returncode, **payload})
        if result.returncode != 0 or payload.get("status") != "PASS":
            errors.extend(payload.get("errors", ["runner validate-only failed"]))
    checks = {
        "logical_tasks_2": queue.get("logical_task_count") == 2 and len(tasks) == 2,
        "active_tasks_2": queue.get("active_task_count") == 2,
        "dataset_etth1_only": {task.get("dataset") for task in tasks} == {"etth1"},
        "variants_exact": {task.get("variant") for task in tasks} == {"gaussian_head", "student_t_head"},
        "selection_calibration_raw_overlap_zero": load(ROOT / "results/gate4c_split_audit.json").get("raw_timestamp_overlap_count") == 0,
        "future_target_overlap_zero": load(ROOT / "results/gate4c_split_audit.json").get("future_target_overlap_count") == 0,
        "embargo_rows_479": load(ROOT / "results/gate4c_split_audit.json").get("embargo_rows") == 479,
        "full_training_no_batch_limits": all(all(task.get(key) is None for key in (
            "max_train_batches", "max_selection_batches", "max_calibration_batches", "max_test_batches"
        )) for task in tasks),
        "retrain_heads": all(task.get("retrain_probability_head") is True for task in tasks),
        "old_gate4b_head_reuse_false": all(task.get("reuse_gate4b_probability_checkpoint") is False for task in tasks),
        "runner_validate_only_2_of_2": all(row.get("status") == "PASS" for row in task_results),
        "required_free_mib_valid": queue.get("required_free_mib", 0) >= 4096,
        "no_smoke_ids": all("smoke" not in str(task.get("task_id", "")) for task in tasks),
    }
    passed = not errors and all(checks.values())
    output = {
        "stage": "Gate 4c formal queue validation",
        "status": "PASS" if passed else "BLOCKED_GATE4C_QUEUE_VALIDATION_FAILURE",
        "queue": str(QUEUE.relative_to(ROOT)),
        "logical_tasks": len(tasks), "active_tasks": queue.get("active_task_count"),
        "required_free_mib": queue.get("required_free_mib"),
        "checks": checks, "task_validate_only_results": task_results, "errors": errors,
    }
    (ROOT / "results/gate4c_formal_queue_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4c_formal_queue_validation.md").write_text(
        "# Gate 4c Formal Queue Validation\n\n"
        f"Status: **{output['status']}**\n\n"
        + "\n".join(f"- {key}: `{str(value).lower()}`" for key, value in checks.items())
        + ("\n\nErrors:\n" + "\n".join(f"- {error}" for error in errors) if errors else "")
        + "\n"
    )
    planning_status = {
        "stage": "Gate 4c", "status": "PLANNING_PASS" if passed else output["status"],
        "prerequisite": load(ROOT / "results/gate4c_prerequisite_audit.json").get("status"),
        "split_audit": load(ROOT / "results/gate4c_split_audit.json").get("status"),
        "mask_inventory": load(ROOT / "results/gate4c_mask_inventory.json").get("status"),
        "conformal_validation": load(ROOT / "results/gate4c_conformal_validation.json").get("status"),
        "data_path_parity": load(ROOT / "results/gate4c_data_path_parity.json").get("status"),
        "protocol_smoke": load(ROOT / "results/gate4c_training_smoke.json").get("status"),
        "formal_queue_validation": output["status"], "scheduler": "NOT_STARTED",
        "GATE4C_IMPLEMENTATION_PASS": False, "GATE4C_SIGNAL_POSITIVE": False,
        "ENTER_GATE4C_EXPANDED_MATRIX": False,
    }
    (ROOT / "results/gate4c_status.json").write_text(json.dumps(planning_status, indent=2) + "\n")
    (ROOT / "reports/gate4c_status.md").write_text(
        f"# Gate 4c Status\n\nStatus: **{planning_status['status']}**\n\nFormal queue validated; scheduler not yet started.\n"
    )
    print(json.dumps(output, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
