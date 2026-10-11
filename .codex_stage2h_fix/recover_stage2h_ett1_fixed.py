#!/usr/bin/env python
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path("/mnt/data/zza/code/BudgetedActiveChannelTokenFormer")
QUEUE = ROOT / "configs/queues/stage2h_minimal_robustness.yaml"
PLAN = ROOT / "results/stage2h_plan.json"
OLD_ID = "stage2h__ett1_practical__pred192__whole_channel_fixed_utility__b0p625__seed2021__20260710_230401"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    queue = read_json(QUEUE)
    matches = [task for task in queue["tasks"] if task["experiment_id"] == OLD_ID]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one old task, found {len(matches)}")
    old_task = matches[0]
    old_dir = Path(old_task["run_dir"])
    old_status = read_json(old_dir / "status.json")
    if old_status.get("status") not in {"RUNNING", "FAILED"}:
        raise RuntimeError(f"unexpected old task status: {old_status.get('status')}")
    old_status.update({
        "status": "FAILED",
        "error_type": "CODE_ERROR",
        "error": "KeyError: 2021 in stage2g_ett1_side_eval.OFFICIAL_RUNS",
        "recovery_action": "new experiment ID created after local seed2021 mapping fix",
        "finished_at": now(),
    })
    write_json(old_dir / "status.json", old_status)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_id = f"stage2h__ett1_practical__pred192__whole_channel_fixed_utility__b0p625__seed2021__retry_mapping_fix__{timestamp}"
    new_dir = ROOT / "runs" / new_id
    new_dir.mkdir(parents=True, exist_ok=False)
    resolved = read_json(old_dir / "resolved_config.yaml")
    resolved.update({
        "experiment_id": new_id,
        "retry_of": OLD_ID,
        "retry_reason": "Fixed Stage 2H seed2021 official-run mapping; formal configuration unchanged.",
        "created_at": now(),
    })
    write_json(new_dir / "resolved_config.yaml", resolved)
    for name in ("git_state.txt", "code.diff"):
        source = old_dir / name
        if source.exists():
            shutil.copy2(source, new_dir / name)
    command = (old_dir / "command.sh").read_text(encoding="utf-8").replace(str(old_dir), str(new_dir))
    command_file = new_dir / "command.sh"
    command_file.write_text(command, encoding="utf-8")
    os.chmod(command_file, os.stat(command_file).st_mode | 0o110)
    write_json(new_dir / "status.json", {
        "experiment_id": new_id,
        "status": "PENDING",
        "created_at": now(),
        "dataset": "ett1_practical",
        "pred_len": 192,
        "seed": 2021,
        "method": "whole_channel_fixed_utility",
        "budget": 0.625,
        "retry_of": OLD_ID,
    })
    new_task = dict(old_task)
    new_task.update({"experiment_id": new_id, "run_dir": str(new_dir), "command_file": str(command_file)})
    queue["tasks"] = [new_task if task["experiment_id"] == OLD_ID else task for task in queue["tasks"]]
    queue["recovery"] = {"old_experiment_id": OLD_ID, "new_experiment_id": new_id, "reason": "seed2021 official-run mapping fix", "created_at": now()}
    write_json(QUEUE, queue)

    plan = read_json(PLAN)
    plan["tasks"] = [new_task if task["experiment_id"] == OLD_ID else task for task in plan["tasks"]]
    plan["recovery"] = queue["recovery"]
    write_json(PLAN, plan)
    recovery = {
        "status": "READY_TO_RETRY_VIA_SCHEDULER",
        "created_at": now(),
        "old_experiment_id": OLD_ID,
        "old_status": "FAILED",
        "old_error_type": "CODE_ERROR",
        "new_experiment_id": new_id,
        "new_status": "PENDING",
        "queue": str(QUEUE),
        "required_free_mib": new_task["required_free_mib"],
        "configuration_changed": False,
    }
    write_json(ROOT / "results/stage2h_ett1_fixed_recovery.json", recovery)
    report = "\n".join([
        "# Stage 2H ETT1 Fixed Recovery", "", "- status: `READY_TO_RETRY_VIA_SCHEDULER`",
        f"- old experiment: `{OLD_ID}`", "- old error: `CODE_ERROR: KeyError 2021`",
        f"- new experiment: `{new_id}`", "- formal configuration changed: `False`",
        f"- required_free_mib: `{new_task['required_free_mib']}`", "",
    ])
    (ROOT / "reports/stage2h_ett1_fixed_recovery.md").write_text(report, encoding="utf-8")
    print(json.dumps(recovery, indent=2))


if __name__ == "__main__":
    main()
