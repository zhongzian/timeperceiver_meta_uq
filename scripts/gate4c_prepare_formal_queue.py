#!/usr/bin/env python
"""Create the Gate 4c formal configs and two-task queue after all launch gates pass."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from gate4c_train_and_calibrate import default_task


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    gates = {
        "prerequisite": load(ROOT / "results/gate4c_prerequisite_audit.json").get("status") == "PASS",
        "split": load(ROOT / "results/gate4c_split_audit.json").get("status") == "PASS",
        "masks": load(ROOT / "results/gate4c_mask_inventory.json").get("status") == "PASS",
        "conformal_validation": load(ROOT / "results/gate4c_conformal_validation.json").get("status") == "PASS",
        "data_path_parity": load(ROOT / "results/gate4c_data_path_parity.json").get("status") == "PASS",
        "protocol_smoke": load(ROOT / "results/gate4c_training_smoke.json").get("status") == "PASS",
    }
    if not all(gates.values()):
        raise RuntimeError(f"Gate 4c queue creation blocked: {gates}")
    required = int(load(ROOT / "results/gate4c_training_smoke.json")["required_free_mib"])
    if required < 4096:
        raise RuntimeError("required_free_mib below protocol floor")
    config_dir = ROOT / "configs/gate4c"
    config_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for variant in ("gaussian_head", "student_t_head"):
        task = default_task(variant, False)
        task.update({
            "task_id": f"etth1__random_rate0.3__{variant}__seed2025",
            "required_free_mib": required,
            "source_checkpoint_sha256": sha256(ROOT / task["source_checkpoint"]),
            "selection_split_sha256": sha256(ROOT / task["selection_split_path"]),
            "calibration_split_sha256": sha256(ROOT / task["calibration_split_path"]),
            "selection_mask_sha256": sha256(ROOT / task["selection_mask_path"]),
            "calibration_mask_sha256": sha256(ROOT / task["calibration_mask_path"]),
            "test_mask_sha256": sha256(ROOT / task["test_mask_path"]),
            "selection_calibration_raw_overlap_count": 0,
            "selection_calibration_future_target_overlap_count": 0,
            "embargo_raw_rows": 479,
            "selection_mask_seed": 43025,
            "calibration_mask_seed": 44025,
            "test_mask_seed": 2025,
            "reuse_allowed": False,
            "gate4a_adapter": False,
            "film": False,
            "online_conformal": False,
            "adaptive_conformal": False,
            "test_time_adaptation": False,
            "official_baseline_retrained": False,
            "extra_datasets_types_rates_seeds": False,
            "command": ["gate4c_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
        })
        filename = (
            "etth1_random_rate0.3_gaussian_seed2025.yaml"
            if variant == "gaussian_head"
            else "etth1_random_rate0.3_student_t_seed2025.yaml"
        )
        config_path = config_dir / filename
        task["formal_config_path"] = str(config_path.relative_to(ROOT))
        config_path.write_text(json.dumps(task, indent=2) + "\n")
        tasks.append(task)
    queue = {
        "stage": "Gate 4c minimal conformal",
        "schema_version": 1,
        "logical_task_count": 2,
        "active_task_count": 2,
        "max_concurrent_total": 2,
        "max_tasks_per_gpu": 1,
        "required_free_mib": required,
        "bounded_oom_retry": 3,
        "allow_reuse": False,
        "experiment_stage_label": "gate4c",
        "pid_file": "runtime/gate4c_minimal_conformal_scheduler.pid",
        "state_file": "runtime/gate4c_minimal_conformal_scheduler_state.json",
        "lock_file": "runtime/locks/gate4c_minimal_conformal_scheduler.lock",
        "retrain_probability_head": True,
        "reuse_gate4b_probability_checkpoint": False,
        "coverage_levels": [0.90, 0.95],
        "conformal_methods": ["marginal", "mask_severity_conditioned"],
        "condition_bins": 3,
        "tasks": tasks,
    }
    queue_path = ROOT / "configs/queues/gate4c_minimal_conformal.yaml"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "launch_gates": gates,
        "configs": [task["formal_config_path"] for task in tasks],
        "queue": str(queue_path.relative_to(ROOT)), "logical_tasks": 2,
        "required_free_mib": required, "scheduler_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
