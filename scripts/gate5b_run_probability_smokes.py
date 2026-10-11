#!/usr/bin/env python3
"""Run and summarize the six resource-gated Gate 5b1 protocol smokes."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import PYTHON, dump, load, write_report  # noqa: E402


def gpu_cards():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print(f"GPU_QUERY_UNAVAILABLE rc={exc.returncode} output={exc.output.strip()}", flush=True)
        return []
    cards = []
    for row in output.splitlines():
        index, uuid, name, total, used, free, utilization = [item.strip() for item in row.split(",", 6)]
        cards.append({
            "index": int(index), "uuid": uuid, "name": name, "total_mib": int(total),
            "used_mib": int(used), "free_mib": int(free), "utilization": int(utilization),
        })
    return sorted(cards, key=lambda item: item["free_mib"], reverse=True)


def smoke_task(formal):
    task = copy.deepcopy(formal)
    task.update({
        "stage": "gate5b1_probability_smoke", "max_epochs": 1,
        "max_train_batches": 4, "max_val_batches": 2, "max_test_batches": 2,
        "smoke_only": True,
    })
    return task


def pattern(task):
    return (
        f"patchtst__{task['dataset']}__gate5b1_probability_smoke__{task['distribution']}__"
        "random_rate0.3__seed2025__*/status.json"
    )


def valid_smoke(run, task):
    try:
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        return all((
            status.get("status") == "SUCCESS", metrics.get("smoke_only") is True,
            metrics.get("dataset") == task["dataset"], metrics.get("distribution") == task["distribution"],
            metrics.get("train_batches") == 4, metrics.get("val_batches") == 2,
            metrics.get("test_batches") == 2, metrics.get("checkpoint_strict_reload") is True,
            metrics.get("source_backbone_checkpoint_sha256") == task["source_backbone_checkpoint_sha256"],
            metrics.get("validation_mask_sha256") == task["validation_mask_sha256"],
            metrics.get("test_mask_sha256") == task["test_mask_sha256"],
            metrics.get("input_mask_applied") is True, metrics.get("batch_y_modified") is False,
            metrics.get("backbone_non_null_gradient_count") == 0,
            metrics.get("optimizer_backbone_parameter_count") == 0,
            metrics.get("test_used_for_model_selection") is False,
        ))
    except Exception:
        return False


def latest_valid(task):
    candidates = [path.parent for path in (ROOT / "runs").glob(pattern(task)) if valid_smoke(path.parent, task)]
    return sorted(candidates)[-1] if candidates else None


def summarize(queue, waiting=None):
    waiting = waiting or {}
    rows = []
    for formal in queue["tasks"]:
        task = smoke_task(formal)
        key = task["task_id"]
        run = latest_valid(task)
        if run:
            metrics = load(run / "metrics.json")
            resource = load(run / "resource_usage.json")
            rows.append({
                "task_id": key, "dataset": task["dataset"], "distribution": task["distribution"],
                "status": "PASS", "experiment_id": run.name, "run_dir": str(run.relative_to(ROOT)),
                "train_batches": metrics["train_batches"], "val_batches": metrics["val_batches"],
                "test_batches": metrics["test_batches"], "train_nll_finite": metrics["train_nll_finite"],
                "validation_nll_finite": metrics["validation_nll_finite"],
                "test_mean_nll": metrics["test_mean_nll"], "test_crps": metrics["test_crps"],
                "loc_nonzero_gradient": metrics["loc_nonzero_gradient"],
                "scale_nonzero_gradient": metrics["scale_nonzero_gradient"],
                "backbone_non_null_gradient_count": metrics["backbone_non_null_gradient_count"],
                "optimizer_backbone_parameter_count": metrics["optimizer_backbone_parameter_count"],
                "fixed_df_parameter_count": metrics["fixed_df_parameter_count"],
                "checkpoint_strict_reload": metrics["checkpoint_strict_reload"],
                "input_mask_applied": metrics["input_mask_applied"],
                "batch_y_modified": metrics["batch_y_modified"], "no_leakage": metrics["no_leakage"],
                "peak_allocated_mib": resource["peak_allocated_mib"],
                "peak_reserved_mib": resource["peak_reserved_mib"],
            })
        else:
            rows.append({
                "task_id": key, "dataset": task["dataset"], "distribution": task["distribution"],
                "status": waiting.get(key, "PENDING"),
            })
    passed = sum(row["status"] == "PASS" for row in rows)
    any_waiting = any(row["status"] == "WAITING_RESOURCE" for row in rows)
    payload = {
        "status": "PASS" if passed == 6 else "WAITING_RESOURCE" if any_waiting else "INCOMPLETE",
        "probability_smoke_pass": f"{passed}/6", "smoke_results_are_formal": False,
        "tasks": rows,
    }
    dump(ROOT / "results/gate5b_probability_smoke.json", payload)
    write_report(ROOT / "reports/gate5b_probability_smoke.md", "Gate 5b1 Probability Smoke", payload)
    return payload


def update_queue_resources(queue, summary):
    rows = summary["tasks"]
    required = {}
    smoke_peaks = {}
    for dataset in {task["dataset"] for task in queue["tasks"]}:
        selected = [row for row in rows if row["dataset"] == dataset]
        peak = max(max(float(row["peak_allocated_mib"]), float(row["peak_reserved_mib"])) for row in selected)
        floor = int(queue["dataset_effective_floor_mib"][dataset])
        smoke_peaks[dataset] = peak
        required[dataset] = max(floor, math.ceil(peak * 1.5) + 1024)
    for task in queue["tasks"]:
        task["required_free_mib"] = required[task["dataset"]]
        task["probability_smoke_peak_mib"] = smoke_peaks[task["dataset"]]
    queue["dataset_probability_smoke_peak_mib"] = smoke_peaks
    queue["dataset_required_free_mib"] = required
    queue["resource_formula"] = (
        "max(effective_dataset_floor, ceil(max(gaussian_smoke_peak,fixed_t8_smoke_peak)*1.5)+1024)"
    )
    queue["probability_smoke_status"] = "PASS"
    queue["launch_enabled"] = False
    queue["blocked_reason"] = "PENDING_FINAL_QUEUE_VALIDATION"
    dump(ROOT / "configs/queues/gate5b_patchtst_probability.yaml", queue)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=0)
    cli = parser.parse_args()
    queue_path = ROOT / "configs/queues/gate5b_patchtst_probability.yaml"
    queue = load(queue_path)
    deadline = time.time() + max(0, cli.wait_seconds)
    waiting = {}
    task_dir = ROOT / "runtime/gate5b_probability_smoke_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for formal in queue["tasks"]:
        task = smoke_task(formal)
        key = task["task_id"]
        if latest_valid(task):
            continue
        task_path = task_dir / f"{key}.json"
        dump(task_path, task)
        while True:
            candidates = [card for card in gpu_cards() if card["free_mib"] >= int(task["required_free_mib"])]
            if candidates:
                card = candidates[0]
                lock_path = ROOT / f"runtime/locks/gpu{card['index']}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    refreshed = next((item for item in gpu_cards() if item["index"] == card["index"]), None)
                    if refreshed is None or refreshed["free_mib"] < int(task["required_free_mib"]):
                        continue
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run = ROOT / "runs" / (
                        f"patchtst__{task['dataset']}__gate5b1_probability_smoke__{task['distribution']}__"
                        f"random_rate0.3__seed2025__{timestamp}"
                    )
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
                    command = [
                        str(PYTHON), "-u", str(ROOT / "scripts/gate5b_train_probability.py"),
                        "--task-file", str(task_path), "--run-dir", str(run),
                    ]
                    completed = subprocess.run(command, cwd=str(ROOT), env=env)
                if completed.returncode != 0:
                    payload = summarize(queue, waiting)
                    payload["status"] = "FAILED"
                    payload["failed_task"] = key
                    dump(ROOT / "results/gate5b_probability_smoke.json", payload)
                    write_report(ROOT / "reports/gate5b_probability_smoke.md", "Gate 5b1 Probability Smoke", payload)
                    print(json.dumps(payload, indent=2))
                    return 2
                break
            if time.time() >= deadline:
                waiting[key] = "WAITING_RESOURCE"
                break
            time.sleep(30)
    payload = summarize(queue, waiting)
    if payload["status"] == "PASS":
        update_queue_resources(queue, payload)
    print(json.dumps({"status": payload["status"], "probability_smoke_pass": payload["probability_smoke_pass"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 3 if payload["status"] == "WAITING_RESOURCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

