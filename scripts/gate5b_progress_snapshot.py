#!/usr/bin/env python3
"""Write an atomic Gate 5b progress snapshot from authoritative artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import dump, load, write_report  # noqa: E402


def read(path, default):
    path = ROOT / path
    return load(path) if path.is_file() else default


def alive(pid):
    try:
        output = subprocess.check_output(["ps", "-p", str(int(pid)), "-o", "stat="], text=True).strip()
        return bool(output) and not output.startswith("Z")
    except (OSError, subprocess.CalledProcessError, TypeError, ValueError):
        return False


def main():
    smoke = read("results/gate5b_backbone_smoke.json", {"status": "NOT_STARTED", "backbone_smoke_pass": "0/3"})
    probability_validation = read("results/gate5b_distribution_validation.json", {"status": "NOT_STARTED"})
    probability_smoke = read(
        "results/gate5b_probability_smoke.json",
        {"status": "NOT_STARTED", "probability_smoke_pass": "0/6"},
    )
    probability_ready = read("results/gate5b_probability_queue_ready.json", {"status": "NOT_STARTED"})
    formal_decision = read("results/gate5b_formal_decision.json", {"status": "NOT_STARTED", "decision": "NOT_STARTED"})
    closure = read("results/gate5b_closure_verification.json", {"status": "NOT_STARTED"})
    backbone_state = read("runtime/gate5b_patchtst_backbones_scheduler_state.json", {"status": "NOT_STARTED"})
    probability_state = read("runtime/gate5b_patchtst_probability_scheduler_state.json", {"status": "NOT_STARTED"})
    pid_path = ROOT / "runtime/gate5b_continue_after_smokes.pid"
    continuation_pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    probability_scheduler_status = probability_state.get("status", "NOT_STARTED")
    if closure.get("GATE5B_STRICT_CLOSURE_PASS") is True:
        overall = "COMPLETE"
    elif probability_scheduler_status in {"COMPLETE", "RUNNING", "WAITING_RESOURCE", "FAILED"}:
        overall = probability_scheduler_status
    elif backbone_state.get("status") in {"RUNNING", "WAITING_RESOURCE"}:
        overall = backbone_state["status"]
    elif probability_ready.get("status") == "PASS":
        overall = "PROBABILITY_QUEUE_READY"
    elif probability_smoke.get("status") == "PASS":
        overall = "PROBABILITY_QUEUE_VALIDATION_PENDING"
    elif smoke.get("status") == "WAITING_RESOURCE":
        overall = "WAITING_RESOURCE"
    elif smoke.get("status") == "PASS":
        overall = "BACKBONE_QUEUE_READY"
    else:
        overall = smoke.get("status", "PLANNING_PASS")
    payload = {
        "status": overall, "stage": "Gate 5b", "planning": "PASS",
        "source_audit": read("results/gate5b_patchtst_source_audit.json", {}).get("status"),
        "environment_audit": read("results/gate5b_environment_audit.json", {}).get("status"),
        "architecture_audit": read("results/gate5b_patchtst_architecture_audit.json", {}).get("status"),
        "data_protocol_validation": read("results/gate5b_data_protocol_validation.json", {}).get("status"),
        "backbone_smoke": smoke.get("backbone_smoke_pass"),
        "backbone_queue": backbone_state.get("status"),
        "probability_queue": probability_state.get("status"),
        "continuation_pid": continuation_pid,
        "continuation_alive": alive(continuation_pid) if continuation_pid else False,
        "probability_validation": probability_validation.get("status"),
        "probability_smoke": probability_smoke.get("probability_smoke_pass", probability_smoke.get("status")),
        "probability_queue_ready": probability_ready.get("status"),
        "formal_decision": formal_decision.get("decision", formal_decision.get("status")),
        "GATE5B_IMPLEMENTATION_PASS": formal_decision.get("GATE5B_IMPLEMENTATION_PASS"),
        "GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE": formal_decision.get("GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE"),
        "CROSS_BACKBONE_DIRECTION_CONSISTENT": formal_decision.get("CROSS_BACKBONE_DIRECTION_CONSISTENT"),
        "GATE5B_STRICT_CLOSURE_PASS": closure.get("GATE5B_STRICT_CLOSURE_PASS", False),
    }
    dump(ROOT / "results/gate5b_status.json", payload)
    write_report(ROOT / "reports/gate5b_status.md", "Gate 5b Status", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
