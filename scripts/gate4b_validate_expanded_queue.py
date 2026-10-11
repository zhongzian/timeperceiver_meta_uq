#!/usr/bin/env python
"""Hard validator for the Gate 4b expanded ETTh1 queue."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_expanded_etth1.yaml"
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
EXPECTED = {("random", 0.1), ("random", 0.3), ("random", 0.5),
            ("block", 0.3), ("channel", 0.3), ("mixed", 0.3)}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    queue = json.loads(QUEUE.read_text())
    tasks = queue.get("tasks", [])
    errors = []
    if queue.get("condition_count") != 6:
        errors.append("condition_count != 6")
    if queue.get("logical_task_count") != 12 or len(tasks) != 12:
        errors.append("logical task count != 12")
    if queue.get("result_row_count_target") != 18:
        errors.append("result row target != 18")
    active = [task for task in tasks if not task.get("reuse_run_dir")]
    reused = [task for task in tasks if task.get("reuse_run_dir")]
    if len(active) + len(reused) != 12 or len(active) > 10:
        errors.append("reused + active mismatch or active > 10")
    if queue.get("active_task_count") != len(active) or queue.get("reused_task_count") != len(reused):
        errors.append("queue active/reused metadata mismatch")
    conditions = {(task.get("missing_type"), float(task.get("missing_rate", -1))) for task in tasks}
    if conditions != EXPECTED:
        errors.append(f"condition set mismatch: {conditions}")
    for condition in EXPECTED:
        subset = [task for task in tasks if (task["missing_type"], float(task["missing_rate"])) == condition]
        if {task["variant"] for task in subset} != {"gaussian_head", "student_t_head"}:
            errors.append(f"variant mismatch: {condition}")
        if len({task["training_protocol_hash"] for task in subset}) != 1:
            errors.append(f"training hashes differ: {condition}")
    if len({task.get("metric_protocol_hash") for task in tasks}) != 1:
        errors.append("metric protocol differs across conditions")
    expected_values = {
        "dataset": "etth1", "seq_len": 384, "pred_len": 96, "seed": 2025,
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "gradient_clip": 1.0, "max_epochs": 20,
        "patience": 3, "early_stopping_metric": "validation_nll",
        "test_used_for_model_selection": False, "crps_samples": 64, "crps_seed": 52025,
        "conformal_calibrated": False, "posthoc_calibration": False,
    }
    checkpoint_hashes = set()
    for index, task in enumerate(tasks):
        for key, value in expected_values.items():
            if task.get(key) != value:
                errors.append(f"{task.get('task_id')} {key} mismatch")
        for key in ("max_train_batches", "max_val_batches", "max_test_batches"):
            if task.get(key) is not None:
                errors.append(f"{task.get('task_id')} contains smoke limit {key}")
        for path_key, hash_key in (("source_checkpoint", "source_checkpoint_sha256"),
                                   ("validation_mask", "validation_mask_sha256"),
                                   ("test_mask", "test_mask_sha256")):
            path = ROOT / task[path_key]
            if not path.is_file():
                errors.append(f"missing {path_key}: {path}")
            elif sha256(path) != task[hash_key]:
                errors.append(f"hash mismatch {path_key}: {task.get('task_id')}")
        checkpoint_hashes.add(task.get("source_checkpoint_sha256"))
        if task in active:
            task_file = ROOT / "runtime" / f"gate4b_expanded_validate_{index}.json"
            task_file.parent.mkdir(exist_ok=True)
            task_file.write_text(json.dumps(task, indent=2) + "\n")
            result = subprocess.run([PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"),
                                     "--task-file", str(task_file), "--validate-only"],
                                    cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                errors.append(f"dry parse failed {task.get('task_id')}: {result.stdout.strip()}")
    if len(checkpoint_hashes) != 1:
        errors.append("source checkpoint hashes differ")
    if queue.get("required_free_mib", 0) < 4096:
        errors.append("required_free_mib below 4096")
    for key in ("official_baseline_retrained", "gate4a_adapter", "missingness_film", "conformal",
                "calibration_fitting", "meta_learning", "test_time_adaptation", "gate4c"):
        if queue.get(key):
            errors.append(f"forbidden enabled: {key}")
    status = "PASS" if not errors else "BLOCKED_EXPANDED_QUEUE_VALIDATION_FAILURE"
    output = {"status": status, "queue": str(QUEUE), "condition_count": len(conditions),
              "logical_probability_tasks": len(tasks), "result_row_target": 18,
              "reused": len(reused), "active": len(active),
              "required_free_mib": queue.get("required_free_mib"), "errors": errors}
    (ROOT / "results/gate4b_expanded_queue_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4b_expanded_queue_validation.md").write_text(
        f"# Gate 4b Expanded Queue Validation\n\nStatus: **{status}**\n\n"
        f"Logical tasks: `{len(tasks)}`; reused: `{len(reused)}`; active: `{len(active)}`.\n\nErrors: `{errors or 'none'}`\n")
    status_path = ROOT / "results/gate4b_expanded_status.json"
    expanded = json.loads(status_path.read_text())
    expanded["expanded_queue"] = status
    if errors:
        expanded["status"] = status
    status_path.write_text(json.dumps(expanded, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
