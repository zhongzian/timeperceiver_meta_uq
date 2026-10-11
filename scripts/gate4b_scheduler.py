#!/usr/bin/env python
"""Generic project scheduler with queue-specific state and bounded OOM retry."""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"


def load(path):
    return json.loads(Path(path).read_text())


def load_resolved(path):
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            if not line.strip() or ":" not in line:
                continue
            key, raw = line.split(":", 1)
            try:
                values[key.strip()] = json.loads(raw.strip())
            except Exception:
                values[key.strip()] = raw.strip().strip("\"'")
    except Exception:
        pass
    return values


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def save(state_path, value):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(state_path)


def gpu_free():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True
        )
        cards = []
        for row in output.splitlines():
            index, uuid, name, free = [item.strip() for item in row.split(",", 3)]
            cards.append((int(free), int(index), uuid, name))
        return sorted(cards, reverse=True)
    except Exception:
        return []


def task_id(task):
    return task.get("task_id") or f"{task['dataset']}__{task['missing_type']}_rate{task['missing_rate']}__{task['variant']}"


def valid_run(run, task):
    try:
        status, metrics = load(run / "status.json"), load(run / "metrics.json")
        if str(task.get("stage", "")).startswith("gate4c"):
            return (
                status.get("status") == "SUCCESS"
                and metrics.get("stage") == "gate4c"
                and metrics.get("variant") == task["variant"]
                and metrics.get("dataset") == task["dataset"]
                and int(metrics.get("experiment_seed", -1)) == int(task["experiment_seed"])
                and metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"]
                and metrics.get("selection_split_sha256") == task["selection_split_sha256"]
                and metrics.get("calibration_split_sha256") == task["calibration_split_sha256"]
                and metrics.get("selection_mask_sha256") == task["selection_mask_sha256"]
                and metrics.get("calibration_mask_sha256") == task["calibration_mask_sha256"]
                and metrics.get("test_mask_sha256") == task["test_mask_sha256"]
                and metrics.get("calibration_labels_seen_during_training") is False
                and metrics.get("calibration_labels_seen_during_selection") is False
                and metrics.get("test_labels_seen_during_calibration") is False
                and metrics.get("point_prediction_mutation") is False
                and metrics.get("backbone_non_null_gradient_count") == 0
                and metrics.get("optimizer_backbone_parameter_count") == 0
                and metrics.get("smoke_only") is False
            )
        resolved_config = load_resolved(run / "resolved_config.yaml")
        run_seed = int(metrics.get("experiment_seed", metrics.get("seed", status.get("seed", -1))))
        run_base_seed = int(metrics.get("train_mask_base_seed", resolved_config.get("train_mask_base_seed", -1)))
        expected_metric_hash = task.get("reuse_metric_protocol_hash", task["metric_protocol_hash"])
        expected_training_hash = task.get("reuse_training_protocol_hash", task["training_protocol_hash"])
        schema_audited = task.get("reuse_schema_audit_pass") is True
        return (status.get("status") == "SUCCESS" and not metrics.get("smoke_only")
                and metrics.get("variant") == task["variant"]
                and metrics.get("dataset") == task["dataset"]
                and run_seed == int(task.get("experiment_seed", task["seed"]))
                and run_base_seed == int(task["train_mask_base_seed"])
                and metrics.get("input_mask_applied") is True
                and (metrics.get("mask_application_verified") is True or schema_audited)
                and (metrics.get("legacy_unmasked_invalid") is False or
                     metrics.get("legacy_unmasked_invalid") is None and schema_audited)
                and metrics.get("training_protocol_hash") == expected_training_hash
                and metrics.get("metric_protocol_hash") == expected_metric_hash
                and metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"]
                and metrics.get("validation_mask_sha256") == task["validation_mask_sha256"]
                and metrics.get("test_mask_sha256") == task["test_mask_sha256"]
                and metrics.get("test_used_for_model_selection") is False
                and metrics.get("backbone_non_null_gradient_count") == 0)
    except Exception:
        return False


def reusable(task):
    if task.get("reuse_allowed") is False:
        return None
    explicit = task.get("reuse_run_dir")
    if explicit:
        run = resolve(ROOT, explicit)
        if valid_run(run, task):
            return run
    seed = int(task.get("experiment_seed", task["seed"]))
    pattern = (f"timeperceiver__{task['dataset']}__gate4b*__{task['variant']}__"
               f"{task['missing_type']}_rate{task['missing_rate']}__seed{seed}__*/status.json")
    candidates = [path.parent for path in (ROOT / "runs").glob(pattern) if valid_run(path.parent, task)]
    return sorted(candidates)[-1] if candidates else None


def experiment_id(queue, task, retry):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"__retry{retry}" if retry else ""
    stage_label = queue.get("experiment_stage_label", "gate4b_expanded")
    seed = int(task.get("experiment_seed", task["seed"]))
    variant_label = {
        "student_t_fixed_df8_head": "student_t_fixed_df8",
    }.get(task["variant"], task["variant"])
    return (f"timeperceiver__{task['dataset']}__{stage_label}__{variant_label}__"
            f"{task['missing_type']}_rate{task['missing_rate']}__seed{seed}__{timestamp}{suffix}")


def main(queue_arg):
    queue_path = resolve(ROOT, queue_arg)
    queue = load(queue_path)
    state_path = resolve(ROOT, queue["state_file"])
    lock_path = resolve(ROOT, queue["lock_file"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        print("DUPLICATE_SCHEDULER_BLOCKED", flush=True)
        return 3
    tasks = queue["tasks"]
    max_concurrent = min(2, max(1, int(queue.get("max_concurrent_total", 1))))
    pid_file = resolve(ROOT, queue["pid_file"]) if queue.get("pid_file") else None
    if pid_file:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()) + "\n")
    state = {
        "queue": str(queue_path), "scheduler_pid": os.getpid(), "status": "RUNNING",
        "max_concurrent_total": max_concurrent, "max_tasks_per_gpu": 1,
        "tasks": {task_id(task): {"status": "PENDING", "variant": task["variant"],
                                  "dataset": task["dataset"],
                                  "experiment_seed": int(task.get("experiment_seed", task["seed"])),
                                  "missing_type": task["missing_type"],
                                  "missing_rate": task["missing_rate"]} for task in tasks},
        "updated_at": datetime.now().isoformat(),
    }
    save(state_path, state)
    indexed_tasks = []
    retry_state = {}
    for index, task in enumerate(tasks):
        key = task_id(task)
        old = reusable(task)
        if old:
            state["tasks"][key].update({"status": "SUCCEEDED", "run_dir": str(old), "reused": True})
            save(state_path, state)
            print(f"REUSE {key} run={old}", flush=True)
            continue
        indexed_tasks.append((index, task, key))
        retry_state[key] = {"count": 0, "last_oom_free": -1, "next_eligible": 0.0}

    running = {}
    failed = False
    last_wait_report = 0.0
    while True:
        for key, item in list(running.items()):
            return_code = item["process"].poll()
            if return_code is None:
                continue
            task = item["task"]
            run = item["run"]
            error = ""
            error_type = ""
            try:
                error = load(run / "metrics.json").get("error", "")
                error_type = load(run / "status.json").get("error_type", "")
            except Exception:
                pass
            if return_code == 0 and valid_run(run, task):
                state["tasks"][key].update({
                    "status": "SUCCEEDED", "run_pid": None, "run_dir": str(run), "exit_code": 0,
                })
                print(f"END {run.name} rc=0", flush=True)
            elif (error_type == "OOM" or "out of memory" in str(error).lower()) \
                    and retry_state[key]["count"] < int(queue.get("bounded_oom_retry", 3)):
                retry_state[key]["count"] += 1
                retry_state[key]["last_oom_free"] = item["free_mib_at_launch"]
                retry_state[key]["next_eligible"] = time.time() + 120 * retry_state[key]["count"]
                state["tasks"][key].update({
                    "status": "WAITING_RESOURCE", "run_pid": None,
                    "oom_retry_count": retry_state[key]["count"],
                    "last_oom_time": datetime.now().isoformat(),
                    "last_oom_free_mib": item["free_mib_at_launch"], "last_error": error,
                    "next_eligible_time": retry_state[key]["next_eligible"],
                })
                print(f"OOM_RETRY {key} retry={retry_state[key]['count']}", flush=True)
            else:
                state["tasks"][key].update({
                    "status": "FAILED", "run_pid": None, "exit_code": return_code,
                    "last_error": error, "error_type": error_type or "UNKNOWN_ERROR",
                })
                failed = True
                print(f"FAILED {key} rc={return_code} type={error_type or 'UNKNOWN_ERROR'}", flush=True)
            del running[key]
            state["updated_at"] = datetime.now().isoformat()
            save(state_path, state)

        statuses = [value["status"] for value in state["tasks"].values()]
        if all(value == "SUCCEEDED" for value in statuses):
            state["status"] = "COMPLETE"
            state["updated_at"] = datetime.now().isoformat()
            save(state_path, state)
            print("QUEUE_COMPLETE", flush=True)
            lock_handle.close()
            return 0
        if failed and not running:
            state["status"] = "FAILED"
            state["updated_at"] = datetime.now().isoformat()
            save(state_path, state)
            lock_handle.close()
            return 2

        launched = False
        if not failed:
            occupied_gpus = {item["gpu_index"] for item in running.values()}
            cards = [card for card in gpu_free() if card[1] not in occupied_gpus]
            while len(running) < max_concurrent:
                choice = None
                for index, task, key in indexed_tasks:
                    if state["tasks"][key]["status"] not in {"PENDING", "WAITING_RESOURCE"}:
                        continue
                    retry = retry_state[key]
                    if time.time() < retry["next_eligible"]:
                        continue
                    candidates = [card for card in cards if card[0] >= int(task["required_free_mib"])]
                    if retry["count"] and retry["last_oom_free"] >= 0:
                        candidates = [card for card in candidates
                                      if card[0] >= retry["last_oom_free"] + 512]
                    if candidates:
                        choice = (index, task, key, candidates[0])
                        break
                    state["tasks"][key].update({
                        "status": "WAITING_RESOURCE", "required_free_mib": task["required_free_mib"],
                        "oom_retry_count": retry["count"],
                        "last_oom_free_mib": retry["last_oom_free"] if retry["last_oom_free"] >= 0 else None,
                    })
                if choice is None:
                    break
                index, task, key, card = choice
                free, gpu_index, gpu_uuid, gpu_name = card
                gpu_lock_path = ROOT / f"runtime/locks/gpu{gpu_index}.lock"
                gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
                with gpu_lock_path.open("w") as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    refreshed = {item[1]: item for item in gpu_free()}.get(gpu_index)
                    if refreshed is None or refreshed[0] < int(task["required_free_mib"]):
                        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                        cards = [item for item in cards if item[1] != gpu_index]
                        continue
                    free, gpu_index, gpu_uuid, gpu_name = refreshed
                    retry = retry_state[key]["count"]
                    run_id = experiment_id(queue, task, retry)
                    run = ROOT / "runs" / run_id
                    task_file = ROOT / "runtime" / f"{task.get('stage', 'gate4b')}_{queue_path.stem}_{index}.json"
                    task_file.write_text(json.dumps(task, indent=2) + "\n")
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
                    command_name = (task.get("command") or ["gate4b_formal_runner"])[0]
                    runner_scripts = {
                        "gate4b_formal_runner": "gate4b_formal_runner.py",
                        "gate4c_formal_runner": "gate4c_formal_runner.py",
                    }
                    if command_name not in runner_scripts:
                        raise RuntimeError(f"unsupported scheduler runner: {command_name}")
                    command = [
                        PYTHON, "-u", str(ROOT / "scripts" / runner_scripts[command_name]),
                        "--task-file", str(task_file), "--run-dir", str(run),
                    ]
                    process = subprocess.Popen(command, env=env, cwd=str(ROOT))
                    gpu_lock.write(json.dumps({
                        "experiment_id": run_id, "pid": process.pid, "gpu_uuid": gpu_uuid,
                        "timestamp": datetime.now().isoformat(),
                    }) + "\n")
                    gpu_lock.flush()
                    fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                running[key] = {
                    "process": process, "task": task, "run": run, "gpu_index": gpu_index,
                    "free_mib_at_launch": free,
                }
                state["tasks"][key].update({
                    "status": "RUNNING", "experiment_id": run_id, "run_pid": process.pid,
                    "run_dir": str(run), "gpu_index": gpu_index, "gpu_uuid": gpu_uuid,
                    "gpu_name": gpu_name, "free_mib_at_launch": free,
                    "oom_retry_count": retry, "reused": False,
                })
                state["status"] = "RUNNING"
                state["updated_at"] = datetime.now().isoformat()
                save(state_path, state)
                print(
                    f"START {run_id} pid={process.pid} gpu={gpu_index} uuid={gpu_uuid} free={free}",
                    flush=True,
                )
                launched = True
                cards = [item for item in cards if item[1] != gpu_index]

        if running:
            state["status"] = "RUNNING"
            sleep_seconds = 10
        else:
            state["status"] = "WAITING_RESOURCE"
            sleep_seconds = 60
            if time.time() - last_wait_report >= 55:
                waiting = [key for key, value in state["tasks"].items()
                           if value["status"] == "WAITING_RESOURCE"]
                print(f"WAITING_RESOURCE tasks={len(waiting)}", flush=True)
                last_wait_report = time.time()
        state["updated_at"] = datetime.now().isoformat()
        save(state_path, state)
        time.sleep(1 if launched else sleep_seconds)


def status(queue_arg):
    queue = load(resolve(ROOT, queue_arg))
    state_path = resolve(ROOT, queue["state_file"])
    print(state_path.read_text() if state_path.exists() else '{"status":"NOT_STARTED"}')


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--status":
        status(sys.argv[2])
        raise SystemExit(0)
    if len(sys.argv) != 2:
        raise SystemExit("usage: gate4b_scheduler.py QUEUE | --status QUEUE")
    raise SystemExit(main(sys.argv[1]))
