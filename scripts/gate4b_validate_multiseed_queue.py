#!/usr/bin/env python
"""Hard validation for the Gate 4b multi-seed formal queue."""
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_multiseed_common import (
    CANONICAL_METRIC_HASH,
    DATASETS,
    RESOURCE_FLOORS,
    SEEDS,
    VARIANTS,
    dump,
    load,
    load_resolved,
    required_free_mib,
    resolve,
    sha256_file,
    validate_reuse,
    write_json_markdown,
)

PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
QUEUE_PATH = ROOT / "configs/queues/gate4b_multiseed_confirmation.yaml"


def check(condition, message, errors):
    if not condition:
        errors.append(message)


def main():
    queue = load(QUEUE_PATH)
    tasks = queue.get("tasks", [])
    errors = []
    active_validation = []
    check(queue.get("scope") == "PROBABILISTIC_HEAD_TRAINING_SEEDS", "scope mismatch", errors)
    check(queue.get("full_end_to_end_multi_seed") is False, "full_end_to_end_multi_seed must be false", errors)
    check(queue.get("logical_task_count") == 24 and len(tasks) == 24, "logical task count must be 24", errors)
    check(queue.get("reused_task_count") == 6, "reused task count must be 6", errors)
    check(queue.get("active_task_count") == 18, "active task count must be 18", errors)
    check(queue.get("max_concurrent_total") <= 2, "max concurrency exceeds 2", errors)
    check(queue.get("max_tasks_per_gpu") == 1, "one-task-per-GPU gate failed", errors)
    check(queue.get("state_file") == "runtime/gate4b_multiseed_confirmation_scheduler_state.json",
          "queue-specific state path mismatch", errors)
    check(queue.get("pid_file") == "runtime/gate4b_multiseed_confirmation_scheduler.pid",
          "queue-specific PID path mismatch", errors)
    check(queue.get("lock_file") == "runtime/locks/gate4b_multiseed_confirmation_scheduler.lock",
          "queue-specific lock path mismatch", errors)
    check({task.get("dataset") for task in tasks} == set(DATASETS), "dataset set mismatch", errors)
    check({int(task.get("experiment_seed", -1)) for task in tasks} == set(SEEDS), "seed set mismatch", errors)
    check({task.get("variant") for task in tasks} == set(VARIANTS), "variant set mismatch", errors)
    key_counts = Counter((task.get("dataset"), int(task.get("experiment_seed", -1)), task.get("variant"))
                         for task in tasks)
    check(len(key_counts) == 24 and all(count == 1 for count in key_counts.values()),
          "dataset/seed/variant Cartesian product mismatch", errors)
    metric_hashes = {task.get("metric_protocol_hash") for task in tasks}
    check(metric_hashes == {CANONICAL_METRIC_HASH}, "metric protocol hash is not canonical and uniform", errors)

    pairs = defaultdict(dict)
    reuse_count = 0
    active_count = 0
    for index, task in enumerate(tasks):
        label = task.get("task_id", f"task_{index}")
        dataset = task.get("dataset")
        seed = int(task.get("experiment_seed", -1))
        variant = task.get("variant")
        pairs[(dataset, seed)][variant] = task
        check(task.get("stage") == "gate4b_multiseed", f"{label}: stage mismatch", errors)
        check(task.get("seed") == seed and seed in SEEDS, f"{label}: seed mismatch", errors)
        check(task.get("probability_head_training_seed") == seed, f"{label}: probability seed mismatch", errors)
        check(task.get("backbone_seed_scope") == "fixed_official_checkpoint",
              f"{label}: backbone scope mismatch", errors)
        check(task.get("train_mask_base_seed") == 40000 + seed, f"{label}: train mask base mismatch", errors)
        check(task.get("validation_mask_seed") == 43025, f"{label}: validation seed mismatch", errors)
        check(task.get("test_mask_seed") == 2025, f"{label}: test seed mismatch", errors)
        check(task.get("variant_in_mask_seed") is False, f"{label}: variant leaked into mask seed", errors)
        check(task.get("missing_type") == "random" and math.isclose(float(task.get("missing_rate", -1)), 0.3),
              f"{label}: missingness mismatch", errors)
        check(task.get("seq_len") == 384 and task.get("pred_len") == 96,
              f"{label}: sequence protocol mismatch", errors)
        check(task.get("freeze_backbone") is True, f"{label}: backbone not frozen", errors)
        check(task.get("optimizer") == "AdamW" and task.get("learning_rate") == 1e-3,
              f"{label}: optimizer mismatch", errors)
        check(task.get("weight_decay") == 1e-4 and task.get("gradient_clip") == 1.0,
              f"{label}: optimization settings mismatch", errors)
        check(task.get("max_epochs") == 20 and task.get("patience") == 3,
              f"{label}: training budget mismatch", errors)
        check(task.get("early_stopping_metric") == "validation_nll",
              f"{label}: early stopping mismatch", errors)
        check(task.get("test_used_for_model_selection") is False,
              f"{label}: test used for selection", errors)
        check(all(task.get(key) is None for key in (
            "max_train_batches", "max_val_batches", "max_test_batches"
        )), f"{label}: smoke batch limit present", errors)
        check(task.get("smoke_only") is False and "smoke" not in label.lower(),
              f"{label}: smoke task in formal queue", errors)
        check(task.get("input_mask_applied") is True and task.get("mask_application_verified") is True,
              f"{label}: input mask gate failed", errors)
        check(task.get("batch_y_modified") is False, f"{label}: batch_y gate failed", errors)
        check(task.get("legacy_unmasked_invalid") is False, f"{label}: legacy invalid evidence", errors)
        check(task.get("conformal_calibrated") is False and task.get("posthoc_calibration") is False,
              f"{label}: calibration forbidden", errors)
        check(task.get("crps_samples") == 64 and task.get("crps_seed") == 52025,
              f"{label}: CRPS protocol mismatch", errors)
        check(task.get("required_free_mib", 0) >= RESOURCE_FLOORS[dataset],
              f"{label}: below resource floor", errors)
        check(task.get("required_free_mib") == required_free_mib(dataset),
              f"{label}: profiled resource requirement mismatch", errors)
        check(task.get("command") == [
            "gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"
        ], f"{label}: command mismatch", errors)
        for path_key, hash_key in (
            ("source_checkpoint", "source_checkpoint_sha256"),
            ("validation_mask", "validation_mask_sha256"),
            ("test_mask", "test_mask_sha256"),
        ):
            path = resolve(task.get(path_key, ""))
            check(path.is_file(), f"{label}: missing {path_key}", errors)
            if path.is_file():
                check(sha256_file(path) == task.get(hash_key), f"{label}: {hash_key} mismatch", errors)
        config_path = resolve(task.get("config", ""))
        check(config_path.is_file(), f"{label}: config missing", errors)
        if config_path.is_file():
            config = load_resolved(config_path)
            check(str(config.get("dataset", "")).lower() == dataset, f"{label}: resolved dataset mismatch", errors)
            check(int(config.get("seq_len", -1)) == 384 and int(config.get("pred_len", -1)) == 96,
                  f"{label}: resolved sequence mismatch", errors)
            check(int(config.get("enc_in", -1)) == int(task.get("target_channels", -2)),
                  f"{label}: target channel mismatch", errors)
            check(int(config.get("batch_size", -1)) == int(task.get("batch_size", -2)),
                  f"{label}: batch size mismatch", errors)
        if task.get("reuse_run_dir"):
            reuse_count += 1
            audit = validate_reuse(dataset, variant)
            check(seed == 2025 and audit["reusable"], f"{label}: illegal reuse", errors)
        else:
            active_count += 1
            validation_file = ROOT / "runtime/gate4b_multiseed_validation" / f"{label}.json"
            dump(validation_file, task)
            command = [
                PYTHON, str(ROOT / "scripts/gate4b_formal_runner.py"),
                "--task-file", str(validation_file), "--validate-only",
            ]
            result = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
            active_validation.append({
                "task_id": label,
                "return_code": result.returncode,
                "output": result.stdout.strip(),
            })
            check(result.returncode == 0, f"{label}: runner --validate-only failed", errors)

    for key, variants in pairs.items():
        check(set(variants) == set(VARIANTS), f"{key}: incomplete pair", errors)
        if set(variants) != set(VARIANTS):
            continue
        gaussian, student = variants["gaussian_head"], variants["student_t_head"]
        for field in (
            "source_checkpoint_sha256", "validation_mask_sha256", "test_mask_sha256",
            "training_protocol_hash", "metric_protocol_hash", "train_mask_base_seed",
            "train_mask_seed_rule", "batch_size", "max_epochs", "patience",
            "learning_rate", "weight_decay", "gradient_clip", "initial_scale", "scale_floor",
        ):
            check(gaussian.get(field) == student.get(field), f"{key}: paired {field} mismatch", errors)
    check(reuse_count == 6 and active_count == 18 and reuse_count + active_count == 24,
          "reuse/active accounting mismatch", errors)
    check(len(active_validation) == 18 and all(row["return_code"] == 0 for row in active_validation),
          "active runner validation count mismatch", errors)
    reuse_inventory = load(ROOT / "results/gate4b_multiseed_reuse_inventory.json")
    check(reuse_inventory.get("legacy_unmasked_reuse_count") == 0,
          "legacy unmasked reuse contamination", errors)
    smoke = load(ROOT / "results/gate4b_multiseed_seed_smoke.json")
    check(smoke.get("status") == "PASS" and smoke.get("successful_variants") == 2,
          "required seed smoke did not pass", errors)
    payload = {
        "status": "PASS" if not errors else "BLOCKED_MULTI_SEED_QUEUE_VALIDATION_FAILURE",
        "queue": str(QUEUE_PATH.relative_to(ROOT)),
        "logical_tasks": len(tasks),
        "reused_tasks": reuse_count,
        "active_tasks": active_count,
        "dataset_count": len({task["dataset"] for task in tasks}),
        "seed_count": len({task["experiment_seed"] for task in tasks}),
        "variant_count": len({task["variant"] for task in tasks}),
        "pair_count": len(pairs),
        "active_validate_only_pass": sum(row["return_code"] == 0 for row in active_validation),
        "old_unmasked_reuse_count": reuse_inventory.get("legacy_unmasked_reuse_count"),
        "canonical_metric_protocol_hash": CANONICAL_METRIC_HASH,
        "dataset_required_free_mib": queue.get("dataset_required_free_mib"),
        "errors": errors,
        "active_validation": active_validation,
        "scheduler_started": False,
    }
    dump(ROOT / "results/gate4b_multiseed_queue_validation.json", payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_queue_validation.md",
                        "Gate 4b Multi-Seed Queue Validation", payload)
    status = load(ROOT / "results/gate4b_multiseed_status.json")
    status["queue_validation"] = payload["status"]
    status["status"] = "QUEUE_VALIDATION_PASS_SCHEDULER_NOT_STARTED" if not errors else payload["status"]
    status["queue"] = "VALIDATED" if not errors else "VALIDATION_FAILED"
    dump(ROOT / "results/gate4b_multiseed_status.json", status)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_status.md",
                        "Gate 4b Multi-Seed Status", status)
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
