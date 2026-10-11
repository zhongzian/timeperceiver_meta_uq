#!/usr/bin/env python
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/mnt/data/zza/code/BudgetedActiveChannelTokenFormer")
GPU4090_UUID = "GPU-a3af3617-5bcd-4c65-a68e-17e716b31edb"
GPU4080_UUID = "GPU-c88b779d-77c7-92b9-e031-943bae990121"


def load_json(path, default=None):
    if default is None:
        default = {}
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def query_gpus():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, uuid, name, total, used, free = parts[:6]
        rows.append(
            {
                "index": idx,
                "uuid": uuid,
                "name": name,
                "total_mib": int(total.split()[0]),
                "used_mib": int(used.split()[0]),
                "free_mib": int(free.split()[0]),
                "raw": line.strip(),
            }
        )
    return rows


def pid_alive(pid):
    try:
        pid = int(pid)
        stat_path = Path(f"/proc/{pid}/stat")
        if stat_path.exists():
            proc_stat = stat_path.read_text(encoding="utf-8", errors="replace")
            closing_paren = proc_stat.rfind(")")
            if closing_paren >= 0 and proc_stat[closing_paren + 1 :].split()[0] == "Z":
                return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_status(task):
    return load_json(Path(task["run_dir"]) / "status.json", {})


def set_status(task, status, **extra):
    path = Path(task["run_dir"]) / "status.json"
    data = load_json(path, {})
    data.update({"experiment_id": task["experiment_id"], "status": status, "updated_at": now()})
    data.update(extra)
    write_json(path, data)


def classify_failure(run_dir):
    log = Path(run_dir) / "evaluation.log"
    if not log.exists():
        log = Path(run_dir) / "train.log"
    text = ""
    if log.exists():
        text = log.read_text(errors="ignore")[-20000:]
    low = text.lower()
    if "out of memory" in low or "cuda oom" in low or "cuda error: out of memory" in low:
        return "OOM"
    if "no such file" in low or "file not found" in low:
        return "DATA_ERROR"
    if "valueerror" in low or "assertionerror" in low:
        return "CONFIG_ERROR"
    if "config_error_gpu_mismatch" in low:
        return "CONFIG_ERROR_GPU_MISMATCH"
    if "nan" in low and ("loss" in low or "diverg" in low):
        return "NAN_OR_DIVERGENCE"
    if "traceback" in low:
        return "CODE_ERROR"
    return "UNKNOWN_ERROR"


def choose_gpu(task, cfg, reservations):
    required = int(task.get("required_free_mib") or cfg.get("gpu_safety_margin_mib", 5000))
    try:
        gpus = query_gpus()
    except Exception as exc:
        return None, [{"raw": f"nvidia-smi query failed: {exc}"}]
    allowed = task.get("allowed_gpu_uuids") or cfg.get("allowed_gpu_uuids") or [GPU4090_UUID, GPU4080_UUID]
    forbidden = set(task.get("forbidden_gpu_uuids") or cfg.get("forbidden_gpu_uuids") or [])
    order = [uuid for uuid in allowed if uuid not in forbidden]
    for uuid in order:
        for gpu in gpus:
            reserved = reservations.get(gpu["uuid"], 0)
            effective_free = gpu["free_mib"] - reserved
            if gpu["uuid"] == uuid and effective_free >= required:
                reservations[gpu["uuid"]] = reserved + required
                return gpu, gpus
    return None, gpus


def launch_task(task, gpu):
    run_dir = Path(task["run_dir"])
    log = ROOT / "logs" / f"{task['experiment_id']}_scheduler_launch.log"
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    env["STAGE2A_GPU_UUID"] = gpu["uuid"]
    env["STAGE2A_GPU_NAME"] = gpu["name"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = ["bash", task["command_file"]]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=log.open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
    set_status(
        task,
        "RUNNING",
        pid=proc.pid,
        gpu_index=gpu["index"],
        gpu_uuid=gpu["uuid"],
        gpu_name=gpu["name"],
        gpu_free_mib_at_start=gpu["free_mib"],
        started_at=now(),
    )
    write_json(run_dir / "scheduler_pid.json", {"pid": proc.pid, "launched_at": now(), "gpu_uuid": gpu["uuid"]})


def update_running(tasks):
    for task in tasks:
        st = read_status(task)
        if st.get("status") != "RUNNING":
            continue
        pid = st.get("pid")
        if not pid:
            pid = load_json(Path(task["run_dir"]) / "scheduler_pid.json", {}).get("pid")
            if pid:
                set_status(task, "RUNNING", pid=pid)
        if pid and pid_alive(pid):
            continue
        metrics = load_json(Path(task["run_dir"]) / "metrics.json", {})
        if metrics.get("status") == "SUCCEEDED":
            metrics_payload = dict(metrics)
            metrics_payload.pop("status", None)
            set_status(task, "SUCCEEDED", **metrics_payload, finished_at=metrics.get("finished_at", now()))
        else:
            error_type = classify_failure(task["run_dir"])
            if error_type == "OOM":
                count = int(st.get("oom_retry_count", 0)) + 1
                wait = min(3600, 300 * (2 ** (count - 1)))
                set_status(task, "RETRYING", error_type="OOM", oom_retry_count=count, next_eligible_time=time.time() + wait, retry_wait_seconds=wait)
            else:
                set_status(task, "FAILED", error_type=error_type, finished_at=now())


def scheduler_loop(queue_file, once=False):
    cfg = load_json(queue_file)
    tasks = cfg.get("tasks", [])
    max_concurrent = min(int(cfg.get("max_concurrent_total", cfg.get("max_concurrent", 3))), 3)
    max_per_gpu = int(cfg.get("max_concurrent_per_gpu", 999))
    poll = int(cfg.get("poll_seconds", 60))
    state_file = ROOT / "runtime" / "queue" / "scheduler_state.json"
    write_json(state_file, {"status": "RUNNING", "queue_file": str(queue_file), "pid": os.getpid(), "started_at": now()})
    while True:
        stop_file = ROOT / "runtime" / "queue" / "stop.request"
        if stop_file.exists():
            write_json(state_file, {"status": "STOPPED", "queue_file": str(queue_file), "pid": os.getpid(), "stopped_at": now()})
            return 0
        update_running(tasks)
        running = [t for t in tasks if read_status(t).get("status") == "RUNNING" and pid_alive(read_status(t).get("pid", -1))]
        slots = max(0, max_concurrent - len(running))
        reservations = {}
        running_by_gpu = {}
        for rt in running:
            uuid = read_status(rt).get("gpu_uuid")
            if uuid:
                running_by_gpu[uuid] = running_by_gpu.get(uuid, 0) + 1
        nvidia_snapshot = []
        for task in tasks:
            if slots <= 0:
                break
            st = read_status(task)
            status = st.get("status", "PENDING")
            if status in {"SUCCEEDED", "RUNNING", "FAILED"}:
                continue
            if status == "RETRYING" and time.time() < float(st.get("next_eligible_time", 0)):
                continue
            gpu, nvidia_snapshot = choose_gpu(task, cfg, reservations)
            if not gpu:
                set_status(task, "WAITING_RESOURCE", reason="insufficient_free_gpu_memory", nvidia_smi=";".join(g["raw"] for g in nvidia_snapshot))
                continue
            if running_by_gpu.get(gpu["uuid"], 0) >= max_per_gpu:
                set_status(task, "WAITING_RESOURCE", reason="max_concurrent_per_gpu_reached", gpu_uuid=gpu["uuid"])
                continue
            launch_task(task, gpu)
            running_by_gpu[gpu["uuid"]] = running_by_gpu.get(gpu["uuid"], 0) + 1
            slots -= 1
        summary = {"status": "RUNNING", "queue_file": str(queue_file), "pid": os.getpid(), "updated_at": now(), "tasks": []}
        for task in tasks:
            st = read_status(task)
            summary["tasks"].append({"experiment_id": task["experiment_id"], "status": st.get("status", "UNKNOWN"), "pid": st.get("pid"), "gpu_uuid": st.get("gpu_uuid"), "error_type": st.get("error_type")})
        write_json(state_file, summary)
        if once:
            return 0
        if all(item["status"] in {"SUCCEEDED", "FAILED"} for item in summary["tasks"]):
            summary["status"] = "COMPLETE"
            summary["completed_at"] = now()
            write_json(state_file, summary)
            return 0
        time.sleep(poll)


def status():
    state = load_json(ROOT / "runtime" / "queue" / "scheduler_state.json", {})
    print(json.dumps(state, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("queue_file")
    run.add_argument("--once", action="store_true")
    sub.add_parser("status")
    args = p.parse_args()
    if args.cmd == "run":
        return scheduler_loop(args.queue_file, once=args.once)
    if args.cmd == "status":
        status()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
