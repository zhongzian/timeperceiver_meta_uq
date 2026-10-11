#!/usr/bin/env python3
"""Queue-specific GPU scheduler for Gate 5b backbone and probability tasks."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import PYTHON, dump, load, resolve  # noqa: E402


def gpu_cards():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
    except Exception:
        return []
    cards = []
    for row in output.splitlines():
        index, uuid, name, total, used, free = [item.strip() for item in row.split(",", 5)]
        cards.append({
            "index": int(index), "uuid": uuid, "name": name, "total_mib": int(total),
            "used_mib": int(used), "free_mib": int(free),
        })
    return sorted(cards, key=lambda item: item["free_mib"], reverse=True)


def task_id(task):
    return task["task_id"]


def valid_run(run, task):
    try:
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        common = all((
            status.get("status") == "SUCCESS", metrics.get("smoke_only") is False,
            metrics.get("dataset") == task["dataset"], metrics.get("backbone") == "patchtst",
            int(metrics.get("experiment_seed", -1)) == int(task["experiment_seed"]),
            metrics.get("training_protocol_hash") == task["training_protocol_hash"],
            metrics.get("test_used_for_model_selection") is False,
        ))
        if task["substage"] == "deterministic_clean_backbone":
            return common and metrics.get("checkpoint_strict_reload") is True and metrics.get("no_leakage") is True
        return all((
            common, metrics.get("variant") == task["variant"],
            metrics.get("source_backbone_checkpoint_sha256") == task["source_backbone_checkpoint_sha256"],
            metrics.get("validation_mask_sha256") == task["validation_mask_sha256"],
            metrics.get("test_mask_sha256") == task["test_mask_sha256"],
            metrics.get("metric_protocol_hash") == task["metric_protocol_hash"],
            metrics.get("input_mask_applied") is True, metrics.get("batch_y_modified") is False,
            metrics.get("backbone_non_null_gradient_count") == 0,
            metrics.get("optimizer_backbone_parameter_count") == 0,
        ))
    except Exception:
        return False


def reusable_run(queue, task):
    if task.get("reuse_allowed") is False:
        return None
    stage_label = queue["experiment_stage_label"]
    pattern = f"patchtst__{task['dataset']}__{stage_label}__*/status.json"
    candidates = [path.parent for path in (ROOT / "runs").glob(pattern) if valid_run(path.parent, task)]
    return sorted(candidates)[-1] if candidates else None


def experiment_id(queue, task, retry):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retry_suffix = f"__retry{retry}" if retry else ""
    if task["substage"] == "deterministic_clean_backbone":
        detail = "clean"
    else:
        detail = f"{task['distribution']}__random_rate0.3"
    return (
        f"patchtst__{task['dataset']}__{queue['experiment_stage_label']}__{detail}__"
        f"seed2025__{timestamp}{retry_suffix}"
    )


def runner_for(task):
    if task["substage"] == "deterministic_clean_backbone":
        return ROOT / "scripts/gate5b_train_backbone.py"
    if task["substage"] == "probability_replication":
        return ROOT / "scripts/gate5b_train_probability.py"
    raise RuntimeError(f"unsupported Gate 5b substage: {task['substage']}")


def process_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def main(queue_argument):
    queue_path = resolve(queue_argument)
    queue = load(queue_path)
    if queue.get("launch_enabled") is not True:
        print(f"QUEUE_NOT_LAUNCHABLE reason={queue.get('blocked_reason', 'launch_enabled=false')}", flush=True)
        return 4
    state_path = resolve(queue["state_file"])
    pid_path = resolve(queue["pid_file"])
    lock_path = resolve(queue["lock_file"])
    stop_path = resolve(queue["stop_file"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if stop_path.exists():
        stop_path.unlink()
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE_SCHEDULER_BLOCKED", flush=True)
        return 3
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n")
    max_concurrent = min(2, int(queue.get("max_concurrent_total", 2)))
    tasks = queue["tasks"]
    state = {
        "queue": str(queue_path), "scheduler_pid": os.getpid(), "status": "RUNNING",
        "max_concurrent_total": max_concurrent, "max_tasks_per_gpu": 1,
        "tasks": {
            task_id(task): {
                "status": "PENDING", "dataset": task["dataset"],
                "substage": task["substage"], "required_free_mib": task["required_free_mib"],
                "oom_retry_count": 0,
            } for task in tasks
        },
        "updated_at": datetime.now().isoformat(),
    }
    dump(state_path, state)
    retries = {task_id(task): {"count": 0, "last_oom_free": -1, "eligible": 0.0} for task in tasks}
    for task in tasks:
        run = reusable_run(queue, task)
        if run:
            state["tasks"][task_id(task)].update({"status": "SUCCEEDED", "run_dir": str(run), "reused": True})
            dump(state_path, state)
            print(f"REUSE {task_id(task)} run={run}", flush=True)
    running = {}
    deterministic_failure = False
    last_wait = 0.0

    while True:
        for key, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            task = item["task"]
            run = item["run"]
            error_type = "UNKNOWN_ERROR"
            error = ""
            if (run / "status.json").is_file():
                result = load(run / "status.json")
                error_type = result.get("error_type") or error_type
                error = result.get("error_summary") or ""
            if return_code == 0 and valid_run(run, task):
                state["tasks"][key].update({
                    "status": "SUCCEEDED", "run_pid": None, "run_dir": str(run), "exit_code": 0,
                })
                print(f"END {run.name} rc=0", flush=True)
            elif error_type == "OOM" and retries[key]["count"] < int(queue.get("bounded_oom_retry", 3)):
                retries[key]["count"] += 1
                retries[key]["last_oom_free"] = item["free_mib_at_launch"]
                retries[key]["eligible"] = time.time() + 120 * retries[key]["count"]
                state["tasks"][key].update({
                    "status": "WAITING_RESOURCE", "run_pid": None,
                    "oom_retry_count": retries[key]["count"], "last_oom_time": datetime.now().isoformat(),
                    "last_oom_free_mib": item["free_mib_at_launch"], "last_error": error,
                    "next_eligible_time": retries[key]["eligible"],
                })
                print(f"OOM_RETRY {key} count={retries[key]['count']}", flush=True)
            else:
                state["tasks"][key].update({
                    "status": "FAILED", "run_pid": None, "exit_code": return_code,
                    "error_type": error_type, "last_error": error,
                })
                deterministic_failure = True
                print(f"FAILED {key} rc={return_code} type={error_type}", flush=True)
            del running[key]
            state["updated_at"] = datetime.now().isoformat()
            dump(state_path, state)

        statuses = [entry["status"] for entry in state["tasks"].values()]
        if all(value == "SUCCEEDED" for value in statuses):
            state["status"] = "COMPLETE"
            state["updated_at"] = datetime.now().isoformat()
            dump(state_path, state)
            print("QUEUE_COMPLETE", flush=True)
            return 0
        if deterministic_failure and not running:
            state["status"] = "FAILED"
            state["updated_at"] = datetime.now().isoformat()
            dump(state_path, state)
            return 2
        if stop_path.exists() and not running:
            state["status"] = "STOPPED"
            state["updated_at"] = datetime.now().isoformat()
            dump(state_path, state)
            print("QUEUE_STOPPED", flush=True)
            return 0

        launched = False
        if not deterministic_failure and not stop_path.exists():
            occupied = {item["gpu_index"] for item in running.values()}
            cards = [card for card in gpu_cards() if card["index"] not in occupied]
            while len(running) < max_concurrent:
                choice = None
                for task in tasks:
                    key = task_id(task)
                    if state["tasks"][key]["status"] not in {"PENDING", "WAITING_RESOURCE"}:
                        continue
                    retry = retries[key]
                    if time.time() < retry["eligible"]:
                        continue
                    candidates = [card for card in cards if card["free_mib"] >= int(task["required_free_mib"])]
                    if retry["count"]:
                        candidates = [card for card in candidates if card["free_mib"] >= retry["last_oom_free"] + 512]
                    if candidates:
                        choice = task, key, candidates[0]
                        break
                    state["tasks"][key].update({
                        "status": "WAITING_RESOURCE", "required_free_mib": task["required_free_mib"],
                        "oom_retry_count": retry["count"],
                    })
                if choice is None:
                    break
                task, key, card = choice
                gpu_lock_path = ROOT / f"runtime/locks/gpu{card['index']}.lock"
                gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
                with gpu_lock_path.open("w") as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    refreshed = next((value for value in gpu_cards() if value["index"] == card["index"]), None)
                    if refreshed is None or refreshed["free_mib"] < int(task["required_free_mib"]):
                        cards = [value for value in cards if value["index"] != card["index"]]
                        continue
                    card = refreshed
                    retry_count = retries[key]["count"]
                    run_id = experiment_id(queue, task, retry_count)
                    run = ROOT / "runs" / run_id
                    task_path = ROOT / "runtime/gate5b_tasks" / f"{queue_path.stem}__{key}.json"
                    dump(task_path, task)
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = card["uuid"]
                    command = [
                        str(PYTHON), "-u", str(runner_for(task)), "--task-file", str(task_path),
                        "--run-dir", str(run),
                    ]
                    process = subprocess.Popen(command, cwd=str(ROOT), env=env)
                    gpu_lock.write(json.dumps({
                        "experiment_id": run_id, "pid": process.pid, "gpu_uuid": card["uuid"],
                        "timestamp": datetime.now().isoformat(),
                    }) + "\n")
                    gpu_lock.flush()
                running[key] = {
                    "process": process, "task": task, "run": run, "gpu_index": card["index"],
                    "free_mib_at_launch": card["free_mib"],
                }
                state["tasks"][key].update({
                    "status": "RUNNING", "experiment_id": run_id, "run_pid": process.pid,
                    "run_dir": str(run), "gpu_index": card["index"], "gpu_uuid": card["uuid"],
                    "gpu_name": card["name"], "free_mib_at_launch": card["free_mib"],
                    "oom_retry_count": retry_count, "reused": False,
                })
                state["status"] = "RUNNING"
                state["updated_at"] = datetime.now().isoformat()
                dump(state_path, state)
                print(f"START {run_id} pid={process.pid} gpu={card['index']} free={card['free_mib']}", flush=True)
                cards = [value for value in cards if value["index"] != card["index"]]
                launched = True

        if running:
            state["status"] = "RUNNING"
            sleep_seconds = 10
        else:
            state["status"] = "WAITING_RESOURCE" if not stop_path.exists() else "STOPPING"
            sleep_seconds = 60
            if time.time() - last_wait >= 55:
                waiting = [key for key, value in state["tasks"].items() if value["status"] == "WAITING_RESOURCE"]
                print(f"WAITING_RESOURCE tasks={len(waiting)}", flush=True)
                last_wait = time.time()
        state["updated_at"] = datetime.now().isoformat()
        dump(state_path, state)
        time.sleep(1 if launched else sleep_seconds)


def status(queue_argument):
    queue = load(resolve(queue_argument))
    state = resolve(queue["state_file"])
    print(state.read_text() if state.is_file() else '{"status":"NOT_STARTED"}')


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--status":
        status(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit("usage: gate5b_scheduler.py QUEUE | --status QUEUE")
    raise SystemExit(main(sys.argv[1]))

