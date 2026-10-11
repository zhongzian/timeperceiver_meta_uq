#!/usr/bin/env python
"""Hard validation for the four-task Gate 4b cross-dataset queue."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_cross_dataset.yaml"
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"


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
    expected_pairs = {(dataset, variant) for dataset in ("weather", "electricity")
                      for variant in ("gaussian_head", "student_t_head")}
    if queue.get("dataset_count") != 2 or queue.get("logical_task_count") != 4 or len(tasks) != 4:
        errors.append("dataset/task count mismatch")
    if queue.get("result_row_count_target") != 6:
        errors.append("result row target mismatch")
    pairs = {(task.get("dataset"), task.get("variant")) for task in tasks}
    if pairs != expected_pairs:
        errors.append(f"dataset/variant matrix mismatch: {pairs}")
    active = [task for task in tasks if not task.get("reuse_run_dir")]
    reused = [task for task in tasks if task.get("reuse_run_dir")]
    if len(active) + len(reused) != 4 or queue.get("active_task_count") != len(active):
        errors.append("active/reused mismatch")
    expected = {
        "missing_type": "random", "missing_rate": 0.3, "seed": 2025,
        "seq_len": 384, "pred_len": 96, "freeze_backbone": True,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "gradient_clip": 1.0, "max_epochs": 20, "patience": 3,
        "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
        "crps_method": "empirical_monte_carlo", "crps_samples": 64, "crps_seed": 52025,
        "conformal_calibrated": False, "posthoc_calibration": False,
        "mask_application_verified": True, "legacy_unmasked_invalid": False,
    }
    metric_hashes = set()
    for index, task in enumerate(tasks):
        for key, value in expected.items():
            if task.get(key) != value:
                errors.append(f"{task.get('task_id')} {key} mismatch")
        if task["dataset"] == "weather" and int(task.get("required_free_mib", 0)) < 7000:
            errors.append("Weather memory threshold below 7000")
        if task["dataset"] == "electricity" and int(task.get("required_free_mib", 0)) < 10000:
            errors.append("Electricity memory threshold below 10000")
        for key in ("max_train_batches", "max_val_batches", "max_test_batches"):
            if task.get(key) is not None:
                errors.append(f"{task.get('task_id')} contains smoke limit {key}")
        if "smoke" in task.get("task_id", "") or task.get("smoke_only"):
            errors.append(f"smoke task in formal queue: {task.get('task_id')}")
        for path_key, hash_key in (("source_checkpoint", "source_checkpoint_sha256"),
                                   ("validation_mask", "validation_mask_sha256"),
                                   ("test_mask", "test_mask_sha256")):
            path = ROOT / task[path_key]
            if not path.is_file():
                errors.append(f"missing {path_key}: {path}")
            elif sha256(path) != task[hash_key]:
                errors.append(f"hash mismatch {path_key}: {task.get('task_id')}")
        config = ROOT / task["config"]
        if not config.is_file():
            errors.append(f"missing dataset config: {config}")
        metric_hashes.add(task["metric_protocol_hash"])
        if task in active:
            task_file = ROOT / "runtime" / f"gate4b_cross_dataset_validate_{index}.json"
            task_file.parent.mkdir(exist_ok=True)
            task_file.write_text(json.dumps(task, indent=2) + "\n")
            result = subprocess.run([PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"),
                                     "--task-file", str(task_file), "--validate-only"],
                                    cwd=ROOT, text=True, capture_output=True)
            if result.returncode:
                errors.append(f"dry parse failed {task.get('task_id')}: {result.stdout.strip()}")
    for dataset in ("weather", "electricity"):
        hashes = {task["training_protocol_hash"] for task in tasks if task["dataset"] == dataset}
        if len(hashes) != 1:
            errors.append(f"training protocol differs within {dataset}")
    if len(metric_hashes) != 1:
        errors.append("metric protocol differs across datasets")
    for key in ("official_baseline_retrained", "gate4a_adapter", "missingness_film", "conformal",
                "calibration_fitting", "meta_learning", "test_time_adaptation", "gate4c",
                "etth1_active_task", "extra_missing_types", "extra_rates", "extra_seeds"):
        if queue.get(key):
            errors.append(f"forbidden enabled: {key}")
    status = "PASS" if not errors else "BLOCKED_CROSS_DATASET_QUEUE_VALIDATION_FAILURE"
    output = {"status": status, "queue": str(QUEUE), "dataset_count": len({t["dataset"] for t in tasks}),
              "logical_probability_tasks": len(tasks), "result_row_target": 6,
              "reused": len(reused), "active": len(active),
              "required_free_mib": queue.get("dataset_required_free_mib"), "errors": errors}
    (ROOT / "results/gate4b_cross_dataset_queue_validation.json").write_text(json.dumps(output, indent=2) + "\n")
    (ROOT / "reports/gate4b_cross_dataset_queue_validation.md").write_text(
        f"# Gate 4b Cross-Dataset Queue Validation\n\nStatus: **{status}**\n\n"
        f"Logical tasks: `{len(tasks)}`; reused: `{len(reused)}`; active: `{len(active)}`.\n\n"
        f"Errors: `{errors or 'none'}`\n")
    status_path = ROOT / "results/gate4b_cross_dataset_status.json"
    stage = json.loads(status_path.read_text())
    stage["queue_validation"] = status
    stage["status"] = "QUEUE_VALIDATION_PASS" if not errors else status
    status_path.write_text(json.dumps(stage, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
