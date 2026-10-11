#!/usr/bin/env python3
"""Run or wait for the six Gate 5a GPU protocol smokes."""

from __future__ import annotations

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

from gate5a_common import DATASETS, NEW_HEADS, RESOURCE_FLOORS, dump, load, write_json_markdown


PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
STATE = ROOT / "runtime/gate5a_protocol_smoke_state.json"
LOCK = ROOT / "runtime/locks/gate5a_protocol_smoke.lock"
PID_FILE = ROOT / "runtime/gate5a_protocol_smoke_runner.pid"
RESULT = ROOT / "results/gate5a_protocol_smoke.json"
REPORT = ROOT / "reports/gate5a_protocol_smoke.md"
SUFFIX = {"laplace_head": "laplace", "student_t_fixed_df8_head": "fixed_t8"}
CONFIGS = tuple(
    ROOT / f"configs/gate5a_smoke/{dataset}_random0.3_{SUFFIX[variant]}_seed2025.json"
    for dataset in DATASETS
    for variant in NEW_HEADS
)


def save_state(value):
    dump(STATE, value)


def task_key(task):
    return f"{task['dataset']}__{task['variant']}"


def gpu_cards():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, name, free = [part.strip() for part in line.split(",", 3)]
        rows.append((int(free), int(index), uuid, name))
    return sorted(rows, reverse=True)


def valid_smoke(run, task):
    try:
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        fixed = task["variant"] == "student_t_fixed_df8_head"
        expected_shape = [int(task["target_channels"])] if False else None
        checks = (
            status.get("status") == "SUCCESS",
            metrics.get("stage") == "gate5a_protocol_smoke",
            metrics.get("smoke_only") is True,
            metrics.get("dataset") == task["dataset"],
            metrics.get("variant") == task["variant"],
            metrics.get("train_batches") == 4,
            metrics.get("val_batches") == 2,
            metrics.get("test_batches") == 2,
            metrics.get("train_nll_finite") is True,
            metrics.get("validation_nll_finite") is True,
            metrics.get("test_metrics_finite") is True,
            metrics.get("all_new_gradients_finite") is True,
            metrics.get("loc_nonzero_gradient") is True,
            metrics.get("scale_nonzero_gradient") is True,
            metrics.get("raw_df_parameter_present") is False,
            metrics.get("raw_df_nonzero_gradient") is False,
            metrics.get("df_gradient_requirement_pass") is True,
            (not fixed or metrics.get("fixed_df_requirement_pass") is True),
            (not fixed or metrics.get("fixed_df_exact") is True),
            metrics.get("backbone_non_null_gradient_count") == 0,
            metrics.get("optimizer_backbone_parameter_count") == 0,
            metrics.get("batch_y_modified") is False,
            metrics.get("test_mask_used_during_training") is False,
            metrics.get("checkpoint_save") is True,
            metrics.get("checkpoint_strict_reload") is True,
            metrics.get("evaluation_after_reload") is True,
            metrics.get("input_mask_applied") is True,
            metrics.get("mask_application_verified") is True,
            metrics.get("legacy_unmasked_invalid") is False,
            metrics.get("test_used_for_model_selection") is False,
            metrics.get("training_protocol_hash") == task["training_protocol_hash"],
            metrics.get("metric_protocol_hash") == task["metric_protocol_hash"],
            metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"],
            metrics.get("validation_mask_sha256") == task["validation_mask_sha256"],
            metrics.get("test_mask_sha256") == task["test_mask_sha256"],
            float(metrics.get("actual_train_missing_rate")) > 0,
            float(metrics.get("scale_min")) > 0,
        )
        del expected_shape
        return all(checks)
    except Exception:
        return False


def prior_valid_run(task):
    pattern = (
        f"timeperceiver__{task['dataset']}__gate5a_smoke__{task['variant']}__"
        "random_rate0.3__seed2025__*/status.json"
    )
    candidates = [path.parent for path in (ROOT / "runs").glob(pattern)]
    valid = [run for run in candidates if valid_smoke(run, task)]
    return sorted(valid)[-1] if valid else None


def summarize(state):
    rows = []
    for key, item in state["tasks"].items():
        row = {"task_id": key, **item}
        run_dir = item.get("run_dir")
        if run_dir and (Path(run_dir) / "metrics.json").is_file():
            metrics = load(Path(run_dir) / "metrics.json")
            row.update({
                "train_batches": metrics.get("train_batches"),
                "val_batches": metrics.get("val_batches"),
                "test_batches": metrics.get("test_batches"),
                "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
                "test_mean_nll": metrics.get("test_mean_nll"),
                "test_crps": metrics.get("test_crps"),
                "test_mse": metrics.get("test_mse"),
                "fixed_df_exact": metrics.get("fixed_df_exact"),
                "raw_df_parameter_present": metrics.get("raw_df_parameter_present"),
            })
        rows.append(row)
    passed = sum(row["status"] == "SUCCEEDED" for row in rows)
    failed = sum(row["status"] == "FAILED" for row in rows)
    overall = "PASS" if passed == 6 else "BLOCKED_GATE5A_PROTOCOL_SMOKE_FAILURE" if failed else state["status"]
    payload = {
        "status": overall,
        "expected": 6,
        "passed": passed,
        "failed": failed,
        "waiting": sum(row["status"] == "WAITING_RESOURCE" for row in rows),
        "running": sum(row["status"] == "RUNNING" for row in rows),
        "resource_floors_mib": RESOURCE_FLOORS,
        "rows": rows,
    }
    dump(RESULT, payload)
    write_json_markdown(
        REPORT, "Gate 5a Protocol Smoke", payload,
        "Smoke metrics are engineering validation only and are excluded from formal evidence.",
    )
    return payload


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE_GATE5A_SMOKE_RUNNER_BLOCKED", flush=True)
        return 3
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n")
    tasks = [load(path) for path in CONFIGS]
    state = {
        "status": "RUNNING",
        "runner_pid": os.getpid(),
        "max_concurrent": 2,
        "one_task_per_gpu": True,
        "tasks": {},
        "updated_at": datetime.now().isoformat(),
    }
    pending = []
    for config_path, task in zip(CONFIGS, tasks):
        key = task_key(task)
        old = prior_valid_run(task)
        state["tasks"][key] = {
            "status": "SUCCEEDED" if old else "PENDING",
            "dataset": task["dataset"],
            "variant": task["variant"],
            "config": str(config_path.relative_to(ROOT)),
            "required_free_mib": RESOURCE_FLOORS[task["dataset"]],
            "run_dir": str(old) if old else None,
            "reused_smoke": bool(old),
            "retry_count": 0,
        }
        if not old:
            pending.append((config_path, task, key))
    save_state(state)
    summarize(state)
    running = {}
    failed = False
    while True:
        for key, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            run = item["run"]
            task = item["task"]
            if return_code == 0 and valid_smoke(run, task):
                state["tasks"][key].update({"status": "SUCCEEDED", "run_pid": None, "exit_code": 0})
                print(f"SMOKE_SUCCESS {key} run={run}", flush=True)
            else:
                error_type = ""
                try:
                    error_type = load(run / "status.json").get("error_type", "")
                except Exception:
                    pass
                retry = state["tasks"][key]["retry_count"]
                if error_type == "OOM" and retry < 3:
                    retry += 1
                    state["tasks"][key].update({
                        "status": "WAITING_RESOURCE", "run_pid": None, "retry_count": retry,
                        "next_eligible_time": time.time() + 120 * retry,
                    })
                    print(f"SMOKE_OOM_RETRY {key} retry={retry}", flush=True)
                else:
                    state["tasks"][key].update({
                        "status": "FAILED", "run_pid": None, "exit_code": return_code,
                        "error_type": error_type or "UNKNOWN_ERROR",
                    })
                    failed = True
                    print(f"SMOKE_FAILED {key} rc={return_code} type={error_type or 'UNKNOWN_ERROR'}", flush=True)
            del running[key]
            state["updated_at"] = datetime.now().isoformat()
            save_state(state)
            summarize(state)

        statuses = [item["status"] for item in state["tasks"].values()]
        if all(value == "SUCCEEDED" for value in statuses):
            state["status"] = "COMPLETE"
            state["updated_at"] = datetime.now().isoformat()
            save_state(state)
            summarize(state)
            print("GATE5A_SMOKES_COMPLETE", flush=True)
            return 0
        if failed and not running:
            state["status"] = "FAILED"
            state["updated_at"] = datetime.now().isoformat()
            save_state(state)
            summarize(state)
            return 2

        occupied = {item["gpu_index"] for item in running.values()}
        try:
            cards = [card for card in gpu_cards() if card[1] not in occupied]
        except Exception as exc:
            print(f"GPU_QUERY_ERROR {exc!r}", flush=True)
            cards = []
        launched = False
        while len(running) < 2:
            choice = None
            for config_path, task, key in pending:
                item = state["tasks"][key]
                if item["status"] not in {"PENDING", "WAITING_RESOURCE"}:
                    continue
                if time.time() < float(item.get("next_eligible_time", 0)):
                    continue
                candidates = [card for card in cards if card[0] >= RESOURCE_FLOORS[task["dataset"]]]
                if candidates:
                    choice = config_path, task, key, candidates[0]
                    break
                item["status"] = "WAITING_RESOURCE"
            if choice is None:
                break
            config_path, task, key, card = choice
            free_mib, gpu_index, gpu_uuid, gpu_name = card
            gpu_lock_path = ROOT / f"runtime/locks/gpu{gpu_index}.lock"
            gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
            with gpu_lock_path.open("w") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                refreshed = {row[1]: row for row in gpu_cards()}.get(gpu_index)
                if refreshed is None or refreshed[0] < RESOURCE_FLOORS[task["dataset"]]:
                    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                    cards = [row for row in cards if row[1] != gpu_index]
                    continue
                free_mib, gpu_index, gpu_uuid, gpu_name = refreshed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_id = (
                    f"timeperceiver__{task['dataset']}__gate5a_smoke__{task['variant']}__"
                    f"random_rate0.3__seed2025__{timestamp}"
                )
                run = ROOT / "runs" / run_id
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                command = [
                    PYTHON, "-u", str(ROOT / "scripts/gate4b_formal_runner.py"),
                    "--task-file", str(config_path), "--run-dir", str(run), "--allow-smoke",
                ]
                process = subprocess.Popen(command, cwd=str(ROOT), env=env)
                gpu_lock.write(json.dumps({
                    "experiment_id": run_id, "pid": process.pid, "gpu_uuid": gpu_uuid,
                    "timestamp": datetime.now().isoformat(),
                }) + "\n")
                gpu_lock.flush()
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            running[key] = {
                "process": process, "run": run, "task": task, "gpu_index": gpu_index,
            }
            state["tasks"][key].update({
                "status": "RUNNING", "run_pid": process.pid, "run_dir": str(run),
                "experiment_id": run_id, "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
                "gpu_name": gpu_name, "free_mib_at_launch": free_mib,
            })
            print(f"SMOKE_START {key} pid={process.pid} gpu={gpu_index} free={free_mib}", flush=True)
            cards = [row for row in cards if row[1] != gpu_index]
            launched = True
            save_state(state)
            summarize(state)

        state["status"] = "RUNNING" if running else "WAITING_RESOURCE"
        state["updated_at"] = datetime.now().isoformat()
        save_state(state)
        summarize(state)
        if not running:
            waiting = sum(item["status"] == "WAITING_RESOURCE" for item in state["tasks"].values())
            print(f"WAITING_RESOURCE tasks={waiting}", flush=True)
        time.sleep(1 if launched else 10 if running else 60)


if __name__ == "__main__":
    raise SystemExit(main())
