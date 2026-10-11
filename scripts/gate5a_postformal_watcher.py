#!/usr/bin/env python3
"""Wait for Gate 5a formal completion, then run the sole final refresh on an idle GPU."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/gate5a_postformal_watcher_state.json"
FORMAL_STATE = ROOT / "runtime/gate5a_distribution_ablation_scheduler_state.json"
LOCK = ROOT / "runtime/locks/gate5a_postformal_watcher.lock"
PID_FILE = ROOT / "runtime/gate5a_postformal_watcher.pid"
REQUIRED_FREE_MIB = 18000


def load(path):
    return json.loads(Path(path).read_text())


def save(value):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATE)


def cards():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, name, free = [part.strip() for part in line.split(",", 3)]
        rows.append((int(free), int(index), uuid, name))
    return sorted(rows, reverse=True)


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE_GATE5A_POSTFORMAL_WATCHER_BLOCKED", flush=True)
        return 3
    PID_FILE.write_text(str(os.getpid()) + "\n")
    state = {
        "status": "WAITING_FORMAL_QUEUE", "watcher_pid": os.getpid(),
        "required_free_mib": REQUIRED_FREE_MIB, "updated_at": datetime.now().isoformat(),
    }
    save(state)
    while True:
        if FORMAL_STATE.is_file():
            formal = load(FORMAL_STATE)
            if formal.get("status") == "FAILED":
                state.update({"status": "BLOCKED_FORMAL_QUEUE_FAILED", "updated_at": datetime.now().isoformat()})
                save(state)
                return 2
            if formal.get("status") == "COMPLETE" and all(
                item.get("status") == "SUCCEEDED" for item in formal.get("tasks", {}).values()
            ) and len(formal.get("tasks", {})) == 12:
                break
        time.sleep(60)
    state.update({"status": "WAITING_RESOURCE", "updated_at": datetime.now().isoformat()})
    save(state)
    process = None
    while process is None:
        selected = None
        while selected is None:
            try:
                selected = next((card for card in cards() if card[0] >= REQUIRED_FREE_MIB), None)
            except Exception as exc:
                state["last_gpu_query_error"] = repr(exc)
            state["updated_at"] = datetime.now().isoformat()
            save(state)
            if selected is None:
                time.sleep(60)
        free, gpu_index, gpu_uuid, gpu_name = selected
        gpu_lock_path = ROOT / f"runtime/locks/gpu{gpu_index}.lock"
        with gpu_lock_path.open("w") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            refreshed = {row[1]: row for row in cards()}.get(gpu_index)
            if refreshed is None or refreshed[0] < REQUIRED_FREE_MIB:
                state.update({"status": "WAITING_RESOURCE", "updated_at": datetime.now().isoformat()})
                save(state)
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
                time.sleep(60)
                continue
            free, gpu_index, gpu_uuid, gpu_name = refreshed
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                ["bash", str(ROOT / "scripts/gate5a_refresh_summary.sh")], cwd=str(ROOT), env=env
            )
            state.update({
                "status": "RUNNING", "refresh_pid": process.pid, "gpu_index": gpu_index,
                "gpu_uuid": gpu_uuid, "gpu_name": gpu_name, "free_mib_at_launch": free,
                "updated_at": datetime.now().isoformat(),
            })
            save(state)
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    return_code = process.wait()
    state.update({
        "status": "COMPLETE" if return_code == 0 else "FAILED",
        "return_code": return_code, "finished_at": datetime.now().isoformat(),
    })
    save(state)
    print(json.dumps(state, indent=2), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
