#!/usr/bin/env python
"""Validated detached launch for the Gate 4b multi-seed scheduler."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_multiseed_common import dump, load, relative, write_json_markdown

QUEUE = "configs/queues/gate4b_multiseed_confirmation.yaml"
STATE = ROOT / "runtime/gate4b_multiseed_confirmation_scheduler_state.json"


def related_processes():
    output = subprocess.check_output(
        ["ps", "-u", os.environ.get("USER", "zza"), "-o", "pid=,cmd="], text=True
    )
    matches = []
    for line in output.splitlines():
        if str(os.getpid()) in line.split(maxsplit=1)[:1]:
            continue
        if (QUEUE in line and "gate4b_scheduler" in line) or (
            "gate4b_multiseed" in line and any(
                token in line for token in ("gate4b_formal_runner", "gate4b_train_probabilistic")
            )
        ):
            matches.append(line.strip())
    return matches


def main():
    validation = load(ROOT / "results/gate4b_multiseed_queue_validation.json")
    smoke = load(ROOT / "results/gate4b_multiseed_seed_smoke.json")
    if validation.get("status") != "PASS" or smoke.get("status") != "PASS":
        raise RuntimeError("BLOCKED_MULTI_SEED_PRELAUNCH_GATE")
    duplicate_or_orphan = related_processes()
    if duplicate_or_orphan:
        raise RuntimeError(f"BLOCKED_DUPLICATE_OR_ORPHAN_MULTI_SEED_PROCESS: {duplicate_or_orphan}")
    upstream_diff = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream/TimePerceiver"), "diff", "--stat"], text=True
    ).strip()
    if upstream_diff:
        raise RuntimeError("BLOCKED_UPSTREAM_TRACKED_DIFF")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = ROOT / "logs" / f"gate4b_multiseed_confirmation_scheduler_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_handle:
        process = subprocess.Popen(
            ["nohup", "bash", "scripts/gate4b_scheduler.sh", "start", QUEUE],
            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(8)
    scheduler_state = load(STATE) if STATE.is_file() else {}
    alive = process.poll() is None
    pid_aligned = int(scheduler_state.get("scheduler_pid", -1)) == process.pid
    launch_ok = alive and pid_aligned and scheduler_state.get("status") in {"RUNNING", "WAITING_RESOURCE"}
    payload = {
        "status": "RUNNING" if launch_ok else "LAUNCH_FAILED",
        "scheduler_pid": process.pid,
        "pid_alive": alive,
        "pid_state_aligned": pid_aligned,
        "queue_status": scheduler_state.get("status", "NOT_CREATED"),
        "queue": QUEUE,
        "scheduler_log": relative(log_path),
        "state_file": relative(STATE),
        "duplicate_or_orphan_prelaunch": duplicate_or_orphan,
        "upstream_tracked_diff_empty": True,
        "launched_at": datetime.now().isoformat(),
    }
    dump(ROOT / "results/gate4b_multiseed_launch.json", payload)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_launch.md",
                        "Gate 4b Multi-Seed Launch", payload)
    status = load(ROOT / "results/gate4b_multiseed_status.json")
    status.update({
        "status": "RUNNING" if launch_ok else "LAUNCH_FAILED",
        "scheduler": "RUNNING" if launch_ok else "LAUNCH_FAILED",
        "scheduler_pid": process.pid,
        "scheduler_log": relative(log_path),
        "scheduler_state_file": relative(STATE),
        "queue_status": scheduler_state.get("status"),
        "summary": "NOT_STARTED",
        "formal_decision": "PENDING",
        "ENTER_GATE4C": False,
    })
    dump(ROOT / "results/gate4b_multiseed_status.json", status)
    write_json_markdown(ROOT / "reports/gate4b_multiseed_status.md",
                        "Gate 4b Multi-Seed Status", status)
    print(json.dumps(payload, indent=2))
    if log_path.is_file():
        print("===== scheduler log tail =====")
        print("\n".join(log_path.read_text().splitlines()[-80:]))
    return 0 if launch_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
