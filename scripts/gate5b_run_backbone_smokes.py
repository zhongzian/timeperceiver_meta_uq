#!/usr/bin/env python3
"""Resource-aware launcher and summarizer for the three Gate 5b0 smokes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import (  # noqa: E402
    DATASETS,
    PATCHTST_COMMIT,
    PYTHON,
    RESOURCE_FLOORS,
    backbone_config,
    dump,
    load,
    write_report,
)


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


def valid_smoke(run, dataset):
    try:
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        return all((
            status.get("status") == "SUCCESS", metrics.get("smoke_only") is True,
            metrics.get("dataset") == dataset, metrics.get("patchtst_upstream_commit") == PATCHTST_COMMIT,
            metrics.get("train_batches") == 4, metrics.get("val_batches") == 2,
            metrics.get("test_batches") == 2, metrics.get("checkpoint_strict_reload") is True,
            metrics.get("test_used_for_model_selection") is False, metrics.get("no_leakage") is True,
        ))
    except Exception:
        return False


def latest_valid(dataset):
    pattern = f"patchtst__{dataset}__gate5b0_backbone_smoke__clean__seed2025__*/status.json"
    runs = [path.parent for path in (ROOT / "runs").glob(pattern) if valid_smoke(path.parent, dataset)]
    return sorted(runs)[-1] if runs else None


def summarize(waiting=None):
    waiting = waiting or {}
    rows = []
    for dataset in DATASETS:
        run = latest_valid(dataset)
        if run:
            metrics = load(run / "metrics.json")
            resource = load(run / "resource_usage.json")
            rows.append({
                "dataset": dataset, "status": "PASS", "experiment_id": run.name,
                "run_dir": str(run.relative_to(ROOT)), "train_batches": metrics["train_batches"],
                "val_batches": metrics["val_batches"], "test_batches": metrics["test_batches"],
                "forward_finite": metrics["forward_finite"], "loss_finite": metrics["loss_finite"],
                "backward_finite": metrics["backward_finite"], "gradient_finite": metrics["gradient_finite"],
                "checkpoint_strict_reload": metrics["checkpoint_strict_reload"],
                "test_used_for_model_selection": metrics["test_used_for_model_selection"],
                "peak_allocated_mib": resource["peak_allocated_mib"],
                "peak_reserved_mib": resource["peak_reserved_mib"],
            })
        else:
            rows.append({"dataset": dataset, "status": waiting.get(dataset, "PENDING")})
    passed = sum(row["status"] == "PASS" for row in rows)
    any_waiting = any(row["status"] == "WAITING_RESOURCE" for row in rows)
    payload = {
        "status": "PASS" if passed == len(DATASETS) else "WAITING_RESOURCE" if any_waiting else "INCOMPLETE",
        "backbone_smoke_pass": f"{passed}/{len(DATASETS)}", "smoke_results_are_formal": False,
        "datasets": rows,
    }
    dump(ROOT / "results/gate5b_backbone_smoke.json", payload)
    write_report(ROOT / "reports/gate5b_backbone_smoke.md", "Gate 5b0 Backbone Smoke", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-seconds", type=int, default=0)
    cli = parser.parse_args()
    deadline = time.time() + max(0, cli.wait_seconds)
    waiting = {}
    for dataset in DATASETS:
        if latest_valid(dataset):
            continue
        task = backbone_config(dataset, smoke=True)
        task_dir = ROOT / "runtime/gate5b_backbone_smoke_tasks"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_path = task_dir / f"{dataset}.json"
        dump(task_path, task)
        while True:
            candidates = [card for card in gpu_cards() if card["free_mib"] >= RESOURCE_FLOORS[dataset]]
            if candidates:
                card = candidates[0]
                lock_path = ROOT / f"runtime/locks/gpu{card['index']}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open("w") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    refreshed = next((item for item in gpu_cards() if item["index"] == card["index"]), None)
                    if refreshed is None or refreshed["free_mib"] < RESOURCE_FLOORS[dataset]:
                        continue
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run = ROOT / "runs" / f"patchtst__{dataset}__gate5b0_backbone_smoke__clean__seed2025__{timestamp}"
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
                    command = [
                        str(PYTHON), "-u", str(ROOT / "scripts/gate5b_train_backbone.py"),
                        "--task-file", str(task_path), "--run-dir", str(run),
                    ]
                    completed = subprocess.run(command, cwd=str(ROOT), env=env)
                if completed.returncode != 0:
                    payload = summarize(waiting)
                    payload["status"] = "BLOCKED_GATE5B_PREDECLARED_CONFIG_OOM" if (
                        (run / "status.json").is_file()
                        and load(run / "status.json").get("error_type") == "OOM"
                    ) else "FAILED"
                    dump(ROOT / "results/gate5b_backbone_smoke.json", payload)
                    write_report(ROOT / "reports/gate5b_backbone_smoke.md", "Gate 5b0 Backbone Smoke", payload)
                    print(json.dumps(payload, indent=2))
                    return 2
                break
            if time.time() >= deadline:
                waiting[dataset] = "WAITING_RESOURCE"
                break
            time.sleep(30)
    payload = summarize(waiting)
    print(json.dumps({"status": payload["status"], "backbone_smoke_pass": payload["backbone_smoke_pass"]}, indent=2))
    return 0 if payload["status"] == "PASS" else 3 if payload["status"] == "WAITING_RESOURCE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
