#!/usr/bin/env python
"""Run the required paired GPU smoke for the modified multi-seed path."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_multiseed_common import VARIANTS, dump, load, relative, write_json_markdown

PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"


def gpu_cards():
    output = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    cards = []
    for line in output.splitlines():
        index, uuid, name, used, free, utilization = [part.strip() for part in line.split(",", 5)]
        cards.append({
            "index": int(index), "uuid": uuid, "name": name, "used_mib": int(used),
            "free_mib": int(free), "utilization_percent": int(utilization),
        })
    return sorted(cards, key=lambda card: card["free_mib"], reverse=True)


def main():
    cards = gpu_cards()
    eligible = [card for card in cards if card["free_mib"] >= 4096]
    if not eligible:
        payload = {
            "status": "WAITING_RESOURCE",
            "required_free_mib": 4096,
            "gpu_snapshot": cards,
            "completed": 0,
            "expected": 2,
        }
        dump(ROOT / "results/gate4b_multiseed_seed_smoke.json", payload)
        write_json_markdown(ROOT / "reports/gate4b_multiseed_seed_smoke.md",
                            "Gate 4b Multi-Seed Seed Smoke", payload)
        print(json.dumps(payload, indent=2))
        return 3
    card = eligible[0]
    rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for variant in VARIANTS:
        task_file = ROOT / "configs/gate4b_multiseed_smoke" / f"etth1_{variant}_random_rate0.3_seed2021.json"
        run_id = (
            f"timeperceiver__etth1__gate4b_multiseed_seed_smoke__{variant}__"
            f"random_rate0.3__seed2021__{timestamp}"
        )
        run_dir = ROOT / "runs" / run_id
        command = [
            PYTHON, "-u", str(ROOT / "scripts/gate4b_formal_runner.py"),
            "--task-file", str(task_file), "--run-dir", str(run_dir), "--allow-smoke",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = card["uuid"]
        started = time.time()
        return_code = subprocess.call(command, cwd=str(ROOT), env=environment)
        status = load(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
        metrics = load(run_dir / "metrics.json") if (run_dir / "metrics.json").is_file() else {}
        rows.append({
            "variant": variant,
            "status": status.get("status", "FAILED"),
            "return_code": return_code,
            "run_dir": relative(run_dir),
            "experiment_id": run_id,
            "experiment_seed": metrics.get("experiment_seed"),
            "train_mask_base_seed": metrics.get("train_mask_base_seed"),
            "train_mask_seed_rule": metrics.get("train_mask_seed_rule"),
            "first_batch_window_indices_sha256": metrics.get("first_batch_window_indices_sha256"),
            "actual_train_missing_rate": metrics.get("actual_train_missing_rate"),
            "train_batches": metrics.get("train_batches"),
            "val_batches": metrics.get("val_batches"),
            "test_batches": metrics.get("test_batches"),
            "input_mask_applied": metrics.get("input_mask_applied"),
            "mask_application_verified": metrics.get("mask_application_verified"),
            "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
            "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
            "checkpoint_strict_reload": metrics.get("checkpoint_strict_reload"),
            "test_metrics_finite": metrics.get("test_metrics_finite"),
            "batch_y_modified": metrics.get("batch_y_modified"),
            "test_mask_used_during_training": metrics.get("test_mask_used_during_training"),
            "training_protocol_hash": metrics.get("training_protocol_hash"),
            "metric_protocol_hash": metrics.get("metric_protocol_hash"),
            "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
            "validation_mask_sha256": metrics.get("validation_mask_sha256"),
            "test_mask_sha256": metrics.get("test_mask_sha256"),
            "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
            "elapsed_seconds": time.time() - started,
            "error": metrics.get("error", ""),
        })
        if return_code != 0:
            break
    paired = len(rows) == 2 and all(
        rows[0].get(key) == rows[1].get(key) for key in (
            "first_batch_window_indices_sha256", "actual_train_missing_rate",
            "training_protocol_hash", "metric_protocol_hash", "source_checkpoint_sha256",
            "validation_mask_sha256", "test_mask_sha256",
        )
    )
    run_checks = []
    for row in rows:
        run_checks.append(
            row["status"] == "SUCCESS"
            and row["return_code"] == 0
            and row["experiment_seed"] == 2021
            and row["train_mask_base_seed"] == 42021
            and row["train_batches"] == 4
            and row["val_batches"] == 2
            and row["test_batches"] == 2
            and row["input_mask_applied"] is True
            and row["mask_application_verified"] is True
            and row["backbone_non_null_gradient_count"] == 0
            and row["optimizer_backbone_parameter_count"] == 0
            and row["checkpoint_strict_reload"] is True
            and row["test_metrics_finite"] is True
            and row["batch_y_modified"] is False
            and row["test_mask_used_during_training"] is False
        )
    passed = len(rows) == 2 and all(run_checks) and paired
    payload = {
        "status": "PASS" if passed else "BLOCKED_MULTI_SEED_SEED_SMOKE_FAILURE",
        "dataset": "etth1",
        "experiment_seed": 2021,
        "expected_variants": 2,
        "successful_variants": sum(run_checks),
        "paired_first_batch_protocol": paired,
        "gpu": card,
        "rows": rows,
        "smoke_results_excluded_from_formal_evidence": True,
    }
    dump(ROOT / "results/gate4b_multiseed_seed_smoke.json", payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_seed_smoke.md",
                        "Gate 4b Multi-Seed Seed Smoke", payload)
    current = load(ROOT / "results/gate4b_multiseed_status.json")
    current["seed_gpu_smoke"] = "PASS 2/2" if passed else payload["status"]
    current["status"] = "SEED_SMOKE_PASS_QUEUE_NOT_CREATED" if passed else payload["status"]
    dump(ROOT / "results/gate4b_multiseed_status.json", current)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_status.md",
                        "Gate 4b Multi-Seed Status", current)
    print(json.dumps(payload, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
