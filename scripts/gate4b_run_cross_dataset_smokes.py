#!/usr/bin/env python
"""Run the four Gate 4b cross-dataset protocol smokes when GPU memory permits."""
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
STATE = ROOT / "runtime/gate4b_cross_dataset_smoke_state.json"
LOCK = ROOT / "runtime/locks/gate4b_cross_dataset_smoke.lock"
CONFIGS = tuple(
    ROOT / f"configs/gate4b_cross_dataset_smoke/{dataset}_{variant}_random_rate0.3_seed2025.json"
    for dataset in ("weather", "electricity")
    for variant in ("gaussian_head", "student_t_head")
)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cards():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in output.splitlines():
        index, uuid, name, free = [part.strip() for part in line.split(",", 3)]
        result.append((int(free), int(index), uuid, name))
    return sorted(result, reverse=True)


def valid_smoke(run, task):
    try:
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        return (
            status.get("status") == "SUCCESS"
            and metrics.get("smoke_only") is True
            and metrics.get("dataset") == task["dataset"]
            and metrics.get("variant") == task["variant"]
            and metrics.get("input_mask_applied") is True
            and metrics.get("mask_application_verified") is True
            and metrics.get("batch_y_modified") is False
            and metrics.get("backbone_non_null_gradient_count") == 0
            and metrics.get("optimizer_backbone_parameter_count") == 0
            and metrics.get("checkpoint_strict_reload") is True
            and metrics.get("evaluation_after_reload") is True
            and metrics.get("training_protocol_hash") == task["training_protocol_hash"]
            and metrics.get("metric_protocol_hash") == task["metric_protocol_hash"]
            and metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"]
            and metrics.get("validation_mask_sha256") == task["validation_mask_sha256"]
            and metrics.get("test_mask_sha256") == task["test_mask_sha256"]
            and metrics.get("train_batches") == 4
            and metrics.get("val_batches") == 2
            and metrics.get("test_batches") == 2
            and metrics.get("raw_df_nonzero_gradient") is True
            and metrics.get("scale_finite") is True
            and metrics.get("scale_positive") is True
            and (task["variant"] == "gaussian_head"
                 or metrics.get("df_finite") is True and metrics.get("df_above_2_01") is True)
        )
    except Exception:
        return False


def existing(task):
    pattern = (f"timeperceiver__{task['dataset']}__gate4b_cross_dataset__smoke__{task['variant']}__"
               "random_rate0.3__seed2025__*/status.json")
    candidates = [path.parent for path in (ROOT / "runs").glob(pattern) if valid_smoke(path.parent, task)]
    return sorted(candidates)[-1] if candidates else None


def write_final(state):
    rows = []
    for item in state["tasks"].values():
        run = Path(item["run_dir"])
        metrics = load(run / "metrics.json")
        rows.append({
            "dataset": metrics["dataset"], "variant": metrics["variant"], "status": "PASS",
            "experiment_id": run.name, "run_dir": str(run.relative_to(ROOT)),
            "train_batches": metrics["train_batches"], "val_batches": metrics["val_batches"],
            "test_batches": metrics["test_batches"], "input_mask_applied": metrics["input_mask_applied"],
            "mask_application_verified": metrics["mask_application_verified"],
            "actual_train_missing_rate": metrics["actual_train_missing_rate"],
            "finite_nll": metrics["train_nll_finite"] and metrics["validation_nll_finite"]
                          and metrics["test_metrics_finite"],
            "loc_gradient": metrics["loc_nonzero_gradient"], "scale_gradient": metrics["scale_nonzero_gradient"],
            "raw_df_gradient": metrics["raw_df_nonzero_gradient"],
            "backbone_gradients": metrics["backbone_non_null_gradient_count"],
            "checkpoint_reload": metrics["checkpoint_strict_reload"],
            "peak_gpu_memory_mib": metrics["peak_gpu_memory_mib"],
            "target_channels": metrics["target_channels"], "reused_smoke": item.get("reused", False),
        })
    payload = {"status": "PASS", "expected": 4, "passed": len(rows), "rows": rows}
    save(ROOT / "results/gate4b_cross_dataset_protocol_smoke.json", payload)
    lines = ["# Gate 4b Cross-Dataset Protocol Smoke", "", "Status: **PASS**", "",
             "| Dataset | Variant | Status | Input mask | Finite NLL | Backbone gradients | Reload | Peak GPU MiB |",
             "|---|---|---|---|---|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['variant']} | PASS | {row['input_mask_applied']} | "
                     f"{row['finite_nll']} | {row['backbone_gradients']} | {row['checkpoint_reload']} | "
                     f"{row['peak_gpu_memory_mib']:.1f} |")
    (ROOT / "reports/gate4b_cross_dataset_protocol_smoke.md").write_text("\n".join(lines) + "\n")
    status_path = ROOT / "results/gate4b_cross_dataset_status.json"
    status = load(status_path)
    status.update({"status": "SMOKE_PASS", "protocol_smoke": "PASS 4/4"})
    save(status_path, status)


def main():
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("DUPLICATE_CROSS_DATASET_SMOKE_BLOCKED", flush=True)
        return 3
    tasks = [load(path) for path in CONFIGS]
    state = {
        "status": "RUNNING", "pid": os.getpid(), "updated_at": datetime.now().isoformat(),
        "tasks": {f"{task['dataset']}__{task['variant']}": {"status": "PENDING",
                  "dataset": task["dataset"], "variant": task["variant"],
                  "required_free_mib": task["required_free_mib"]} for task in tasks},
    }
    save(STATE, state)
    for task in tasks:
        key = f"{task['dataset']}__{task['variant']}"
        old = existing(task)
        if old:
            state["tasks"][key].update({"status": "SUCCEEDED", "run_dir": str(old), "reused": True})
            save(STATE, state)
            print(f"REUSE_SMOKE {key} run={old}", flush=True)
            continue
        while True:
            eligible = [card for card in cards() if card[0] >= int(task["required_free_mib"])]
            if eligible:
                break
            state["status"] = "WAITING_RESOURCE"
            state["tasks"][key]["status"] = "WAITING_RESOURCE"
            state["updated_at"] = datetime.now().isoformat()
            save(STATE, state)
            print(f"WAITING_RESOURCE smoke={key} required={task['required_free_mib']}", flush=True)
            time.sleep(60)
        free, gpu_index, gpu_uuid, gpu_name = eligible[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = (f"timeperceiver__{task['dataset']}__gate4b_cross_dataset__smoke__{task['variant']}__"
                  f"random_rate0.3__seed2025__{timestamp}")
        run = ROOT / "runs" / run_id
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        command = [PYTHON, "-u", str(ROOT / "scripts/gate4b_formal_runner.py"),
                   "--task-file", str(CONFIGS[tasks.index(task)]), "--run-dir", str(run), "--allow-smoke"]
        process = subprocess.Popen(command, cwd=ROOT, env=env)
        state["status"] = "RUNNING"
        state["tasks"][key].update({"status": "RUNNING", "experiment_id": run_id, "run_pid": process.pid,
                                    "gpu_index": gpu_index, "gpu_uuid": gpu_uuid, "gpu_name": gpu_name,
                                    "free_mib_at_launch": free, "reused": False})
        save(STATE, state)
        print(f"START_SMOKE {run_id} pid={process.pid} gpu={gpu_index} uuid={gpu_uuid} free={free}", flush=True)
        return_code = process.wait()
        if return_code or not valid_smoke(run, task):
            state["status"] = "BLOCKED_CROSS_DATASET_PROTOCOL_SMOKE_FAILURE"
            state["tasks"][key].update({"status": "FAILED", "run_pid": None, "exit_code": return_code})
            save(STATE, state)
            return return_code or 2
        state["tasks"][key].update({"status": "SUCCEEDED", "run_pid": None,
                                    "run_dir": str(run), "exit_code": 0})
        save(STATE, state)
        print(f"END_SMOKE {run_id} rc=0", flush=True)
    state["status"] = "COMPLETE"
    state["updated_at"] = datetime.now().isoformat()
    save(STATE, state)
    write_final(state)
    print("CROSS_DATASET_SMOKE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
