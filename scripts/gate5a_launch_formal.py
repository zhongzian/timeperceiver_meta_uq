#!/usr/bin/env python3
"""Launch the validated Gate 5a formal scheduler once, detached."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import dump, load, write_json_markdown


QUEUE = "configs/queues/gate5a_distribution_ablation.yaml"


def related_processes():
    output = subprocess.check_output(
        ["ps", "-u", os.environ.get("USER", "zza"), "-o", "pid=,ppid=,stat=,etime=,cmd="], text=True
    )
    duplicate = []
    orphan = []
    for line in output.splitlines():
        if "gate4b_scheduler.py" in line and QUEUE in line:
            duplicate.append(line.strip())
        if ("gate4b_formal_runner.py" in line or "gate4b_train_probabilistic.py" in line) and "gate5a" in line:
            orphan.append(line.strip())
    return duplicate, orphan


def main():
    validation = load(ROOT / "results/gate5a_queue_validation.json")
    if validation.get("status") != "PASS":
        raise RuntimeError("BLOCKED_GATE5A_QUEUE_VALIDATION_FAILURE")
    duplicates, orphans = related_processes()
    if duplicates:
        payload = {"status": "ALREADY_RUNNING", "duplicate_scheduler": duplicates, "orphan_tasks": orphans}
        dump(ROOT / "results/gate5a_launch.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    if orphans:
        payload = {"status": "BLOCKED_ORPHAN_GATE5A_PROCESS", "duplicate_scheduler": [], "orphan_tasks": orphans}
        dump(ROOT / "results/gate5a_launch.json", payload)
        print(json.dumps(payload, indent=2))
        return 2
    gpu_snapshot = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = ROOT / f"logs/gate5a_distribution_ablation_scheduler_{timestamp}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts/gate4b_scheduler.sh"), "start", QUEUE],
        cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    time.sleep(8)
    live = process.poll() is None
    if not live:
        payload = {
            "status": "FAILED", "scheduler_pid": process.pid,
            "scheduler_log": str(log.relative_to(ROOT)), "exit_code": process.returncode,
            "gpu_snapshot": gpu_snapshot,
        }
        dump(ROOT / "results/gate5a_launch.json", payload)
        print(json.dumps(payload, indent=2))
        return 2
    (ROOT / "runtime/gate5a_distribution_ablation_scheduler.pid").write_text(str(process.pid) + "\n")
    state = load(ROOT / "runtime/gate5a_distribution_ablation_scheduler_state.json")
    payload = {
        "status": state.get("status", "RUNNING"), "queue": QUEUE,
        "scheduler_pid": process.pid, "scheduler_log": str(log.relative_to(ROOT)),
        "gpu_snapshot": gpu_snapshot, "duplicate_scheduler": [], "orphan_tasks": [],
        "launched_at": datetime.now().isoformat(),
    }
    dump(ROOT / "results/gate5a_launch.json", payload)
    write_json_markdown(ROOT / "reports/gate5a_launch.md", "Gate 5a Launch", payload)
    gate_status = load(ROOT / "results/gate5a_status.json")
    gate_status.update({
        "status": state.get("status", "RUNNING"), "formal_queue": "VALIDATED",
        "queue_validation": "PASS", "scheduler": state.get("status", "RUNNING"),
        "scheduler_pid": process.pid, "scheduler_log": str(log.relative_to(ROOT)),
    })
    dump(ROOT / "results/gate5a_status.json", gate_status)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
