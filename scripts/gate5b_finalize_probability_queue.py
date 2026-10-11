#!/usr/bin/env python3
"""Enable Gate 5b1 formal queue only after all immutable prerequisites pass."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5b_common import PYTHON, dump, load, write_report  # noqa: E402


def main():
    queue_path = ROOT / "configs/queues/gate5b_patchtst_probability.yaml"
    prerequisites = {
        "backbone_manifest": load(ROOT / "results/gate5b_patchtst_backbone_manifest.json").get("status") == "PASS",
        "distribution_validation": load(ROOT / "results/gate5b_distribution_validation.json").get("status") == "PASS",
        "pairing_validation": load(ROOT / "results/gate5b_pairing_validation.json").get("status") == "PASS",
        "pairing_3_of_3": load(ROOT / "results/gate5b_pairing_validation.json").get("pairs_pass") == "3/3",
        "probability_smoke": load(ROOT / "results/gate5b_probability_smoke.json").get("status") == "PASS",
        "probability_smoke_6_of_6": (
            load(ROOT / "results/gate5b_probability_smoke.json").get("probability_smoke_pass") == "6/6"
        ),
    }
    if not all(prerequisites.values()):
        raise SystemExit(f"Gate 5b1 prerequisites failed: {prerequisites}")
    precheck = subprocess.run(
        [str(PYTHON), str(ROOT / "scripts/gate5b_validate_queues.py")],
        cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if precheck.returncode != 0:
        raise SystemExit("pre-enable queue validation failed:\n" + precheck.stdout)
    queue = load(queue_path)
    if any(task.get("source_backbone_checkpoint", "").startswith("__FROM_") for task in queue["tasks"]):
        raise SystemExit("unresolved checkpoint placeholder remains")
    queue["launch_enabled"] = True
    queue["blocked_reason"] = ""
    queue["prerequisites"] = {key: "PASS" for key in prerequisites}
    queue["formal_queue_ready"] = True
    dump(queue_path, queue)
    payload = {
        "status": "PASS", "queue": str(queue_path.relative_to(ROOT)),
        "logical_tasks": len(queue["tasks"]), "launch_enabled": True,
        "prerequisites": prerequisites, "dataset_required_free_mib": queue["dataset_required_free_mib"],
        "max_concurrent_total": queue["max_concurrent_total"], "max_tasks_per_gpu": queue["max_tasks_per_gpu"],
    }
    dump(ROOT / "results/gate5b_probability_queue_ready.json", payload)
    write_report(ROOT / "reports/gate5b_probability_queue_ready.md", "Gate 5b Probability Queue Ready", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

