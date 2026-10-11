#!/usr/bin/env python
"""Gate 2 official TimePerceiver baseline runner.

This wrapper keeps official model code untouched while saving the experiment
metadata expected by the project gates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from parse_timeperceiver_metrics import parse_run


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "TimePerceiver"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs" / "gate2"
RUNTIME = ROOT / "runtime"
SEED = 2025


DATASETS: List[Dict[str, Any]] = [
    {
        "dataset": "etth1",
        "display": "ETTh1",
        "model_id": "ETTh1_384_96",
        "script": "scripts/ETT/TimePerceiver_ETTh1.sh",
        "data": "ETTh1",
        "root_path": str(ROOT / "data" / "ETT-small") + "/",
        "data_path": "ETTh1.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seq_len": 384,
        "label_len": 0,
        "pred_len": 96,
        "enc_in": 7,
        "d_model": 256,
        "d_ff": 512,
        "n_heads": 4,
        "num_latents": 32,
        "latent_dim": 64,
        "latent_d_ff": 128,
        "dropout": 0.2,
        "standard": 0,
        "generalized": 1,
        "use_latent": 1,
        "num_latent_blocks": 1,
        "patch_len": 12,
        "separate_ratio": 1,
        "batch_size": 128,
        "train_epochs": 50,
        "learning_rate": 0.0005,
        "warmup_epochs": 5,
        "patience": 10,
        "weight_decay": 0.05,
        "num_workers": 4,
    },
    {
        "dataset": "weather",
        "display": "Weather",
        "model_id": "weather_384_96",
        "script": "scripts/Weather/TimePerceiver.sh",
        "data": "custom",
        "root_path": str(ROOT / "data" / "weather") + "/",
        "data_path": "weather.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seq_len": 384,
        "label_len": 0,
        "pred_len": 96,
        "enc_in": 21,
        "d_model": 512,
        "d_ff": 1024,
        "n_heads": 8,
        "num_latents": 32,
        "latent_dim": 128,
        "latent_d_ff": 256,
        "dropout": 0.2,
        "standard": 0,
        "generalized": 1,
        "use_latent": 1,
        "num_latent_blocks": 1,
        "patch_len": 24,
        "separate_ratio": 1,
        "batch_size": 128,
        "train_epochs": 100,
        "learning_rate": 0.0001,
        "warmup_epochs": 5,
        "patience": 10,
        "weight_decay": 0.05,
        "num_workers": 4,
    },
    {
        "dataset": "electricity",
        "display": "Electricity",
        "model_id": "electricity_384_96",
        "script": "scripts/ECL/TimePerceiver.sh",
        "data": "custom",
        "root_path": str(ROOT / "data" / "electricity") + "/",
        "data_path": "electricity.csv",
        "features": "M",
        "target": "OT",
        "freq": "h",
        "seq_len": 384,
        "label_len": 0,
        "pred_len": 96,
        "enc_in": 321,
        "d_model": 512,
        "d_ff": 1024,
        "n_heads": 8,
        "num_latents": 32,
        "latent_dim": 128,
        "latent_d_ff": 256,
        "dropout": 0.2,
        "standard": 0,
        "generalized": 1,
        "use_latent": 1,
        "num_latent_blocks": 1,
        "patch_len": 24,
        "separate_ratio": 0,
        "batch_size": 64,
        "train_epochs": 100,
        "learning_rate": 0.0005,
        "warmup_epochs": 5,
        "patience": 10,
        "weight_decay": 0.05,
        "num_workers": 4,
    },
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def cmd_output(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> str:
    try:
        return run_cmd(cmd, cwd=cwd, env=env).stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostics
        return f"COMMAND_FAILED {shlex.join(cmd)}: {exc}"


def ensure_dirs() -> None:
    for path in (RUNS, REPORTS, LOGS, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)


def shell_join(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def config_to_command(cfg: Dict[str, Any], run_dir: Path, gpu_arg: int) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-u",
        str(UPSTREAM / "run.py"),
        "--is_training",
        "1",
        "--root_path",
        cfg["root_path"],
        "--data_path",
        cfg["data_path"],
        "--model_id",
        cfg["model_id"],
        "--model",
        "TimePerceiver",
        "--data",
        cfg["data"],
        "--features",
        cfg["features"],
        "--target",
        cfg["target"],
        "--freq",
        cfg["freq"],
        "--seq_len",
        str(cfg["seq_len"]),
        "--label_len",
        str(cfg["label_len"]),
        "--pred_len",
        str(cfg["pred_len"]),
        "--enc_in",
        str(cfg["enc_in"]),
        "--des",
        "Exp",
        "--d_model",
        str(cfg["d_model"]),
        "--d_ff",
        str(cfg["d_ff"]),
        "--n_heads",
        str(cfg["n_heads"]),
        "--itr",
        "1",
        "--num_latents",
        str(cfg["num_latents"]),
        "--latent_dim",
        str(cfg["latent_dim"]),
        "--latent_d_ff",
        str(cfg["latent_d_ff"]),
        "--dropout",
        str(cfg["dropout"]),
        "--standard",
        str(cfg["standard"]),
        "--generalized",
        str(cfg["generalized"]),
        "--use_latent",
        str(cfg["use_latent"]),
        "--num_latent_blocks",
        str(cfg["num_latent_blocks"]),
        "--patch_len",
        str(cfg["patch_len"]),
        "--separate_ratio",
        str(cfg["separate_ratio"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--train_epochs",
        str(cfg["train_epochs"]),
        "--warmup_epochs",
        str(cfg["warmup_epochs"]),
        "--patience",
        str(cfg["patience"]),
        "--learning_rate",
        str(cfg["learning_rate"]),
        "--weight_decay",
        str(cfg["weight_decay"]),
        "--num_workers",
        str(cfg["num_workers"]),
        "--checkpoints",
        str(run_dir / "checkpoints"),
        "--gpu",
        str(gpu_arg),
    ]
    return cmd


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {yaml_scalar(child_value)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n")


def nvidia_smi_snapshot() -> str:
    return cmd_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader"])


def select_gpu() -> int:
    rows = nvidia_smi_snapshot().splitlines()
    best_idx = 0
    best_free = -1
    for row in rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            free = int(parts[5].split()[0])
        except (ValueError, IndexError):
            continue
        if free > best_free:
            best_free = free
            best_idx = idx
    return best_idx


def write_git_state(run_dir: Path) -> None:
    content = []
    for cmd in (
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        ["git", "-C", str(UPSTREAM), "status", "--short"],
        ["git", "-C", str(UPSTREAM), "diff", "--stat"],
        ["git", "-C", str(UPSTREAM), "remote", "-v"],
    ):
        content.append(f"$ {shell_join(cmd)}")
        content.append(cmd_output(cmd))
        content.append("")
    (run_dir / "git_state.txt").write_text("\n".join(content))
    diff = run_cmd(["git", "-C", str(UPSTREAM), "diff", "--binary"]).stdout
    (run_dir / "code.diff").write_text(diff)


def write_env(run_dir: Path, env: Dict[str, str], before: str, after: Optional[str] = None) -> None:
    lines = [
        f"date: {iso_now()}",
        f"hostname: {socket.gethostname()}",
        f"whoami: {cmd_output(['whoami'])}",
        f"cwd: {run_dir}",
        f"conda_default_env: {env.get('CONDA_DEFAULT_ENV', '')}",
        f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}",
        "",
        "python:",
        cmd_output([sys.executable, "--version"], env=env),
        "",
        "torch:",
        cmd_output([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')"], env=env),
        "",
        "nvidia-smi before:",
        before,
    ]
    if after is not None:
        lines.extend(["", "nvidia-smi after:", after])
    (run_dir / "env.txt").write_text("\n".join(lines) + "\n")


def existing_success(dataset: str) -> Optional[Path]:
    for status_path in sorted(RUNS.glob(f"timeperceiver__{dataset}__official_baseline__seed{SEED}__*/status.json")):
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        if status.get("status") == "SUCCESS":
            return status_path.parent
    return None


def monitor_gpu(stop: threading.Event, selected_gpu: int, out: Dict[str, Any]) -> None:
    peak = None
    samples = 0
    while not stop.is_set():
        snapshot = nvidia_smi_snapshot()
        out["last_snapshot"] = snapshot
        for row in snapshot.splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    used = int(parts[4].split()[0])
                except (ValueError, IndexError):
                    continue
                if idx == selected_gpu:
                    peak = used if peak is None else max(peak, used)
                    samples += 1
        time.sleep(10)
    out["peak_memory_used_mib"] = peak
    out["samples"] = samples


def run_one(cfg: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    dataset = cfg["dataset"]
    if not force:
        previous = existing_success(dataset)
        if previous:
            return {
                "dataset": dataset,
                "status": "SKIPPED_SUCCESS_EXISTS",
                "run_dir": str(previous),
                "experiment_id": previous.name,
            }

    selected_gpu = select_gpu()
    # Official Exp_Basic resets CUDA_VISIBLE_DEVICES to args.gpu and then uses
    # cuda:{args.gpu}; after exposing one physical GPU, the visible ordinal is 0.
    official_gpu_arg = 0
    timestamp = now_stamp()
    experiment_id = f"timeperceiver__{dataset}__official_baseline__seed{SEED}__{timestamp}"
    run_dir = RUNS / experiment_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        experiment_id = f"timeperceiver__{dataset}__official_baseline__seed{SEED}__{timestamp}_{suffix}"
        run_dir = RUNS / experiment_id
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(UPSTREAM) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
    env["PYTHONUNBUFFERED"] = "1"

    cmd = config_to_command(cfg, run_dir, official_gpu_arg)
    before = nvidia_smi_snapshot()
    write_env(run_dir, env, before)
    write_git_state(run_dir)

    config_record = dict(cfg)
    config_record.update(
        {
            "experiment_id": experiment_id,
            "mode": "official_baseline",
            "seed": SEED,
            "model": "TimePerceiver",
            "official_command_source": cfg["script"],
            "official_protocol": True,
            "selected_cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "selected_gpu_index_before_visibility": selected_gpu,
            "official_gpu_argument": official_gpu_arg,
            "gpu_mapping_note": "Wrapper selects a physical nvidia-smi index via CUDA_VISIBLE_DEVICES; official run.py receives --gpu 0 because Exp_Basic reassigns CUDA_VISIBLE_DEVICES internally.",
            "run_dir": str(run_dir),
            "smoke": False,
            "gate1_smoke_difference": "official full train_epochs/batch/model dimensions/pred_len from script first block; no smoke-only shrinking",
        }
    )
    write_yaml(run_dir / "resolved_config.yaml", config_record)
    (run_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"cd {shlex.quote(str(run_dir))}\n"
        f"export PYTHONPATH={shlex.quote(str(UPSTREAM))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])}\n"
        f"{shell_join(cmd)}\n"
    )
    os.chmod(run_dir / "command.sh", 0o755)

    status = {
        "experiment_id": experiment_id,
        "status": "RUNNING",
        "dataset": dataset,
        "start_time": iso_now(),
        "end_time": "",
        "exit_code": None,
        "error_summary": "",
        "metrics_path": str(run_dir / "metrics.json"),
        "run_dir": str(run_dir),
    }
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    start = time.time()
    monitor_data: Dict[str, Any] = {}
    stop_event = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop_event, selected_gpu, monitor_data), daemon=True)
    monitor.start()
    train_log = run_dir / "train.log"
    exit_code = None
    try:
        with train_log.open("w") as log:
            log.write(f"# command: {shell_join(cmd)}\n")
            log.write(f"# cwd: {run_dir}\n")
            log.write(f"# CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(run_dir), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
            exit_code = proc.wait()
    finally:
        stop_event.set()
        monitor.join(timeout=5)

    elapsed = time.time() - start
    after = nvidia_smi_snapshot()
    write_env(run_dir, env, before, after)
    parsed = parse_run(run_dir)

    log_text = train_log.read_text(errors="replace") if train_log.exists() else ""
    log_lower = log_text.lower()
    if exit_code == 0 and parsed.get("mse") is not None and parsed.get("mae") is not None:
        run_status = "SUCCESS"
        error_summary = ""
    elif "out of memory" in log_lower or "cuda oom" in log_lower:
        run_status = "OOM"
        error_summary = "CUDA out of memory detected in train.log"
    elif "invalid device ordinal" in log_lower:
        run_status = "CODE_ERROR"
        error_summary = "CUDA invalid device ordinal caused by wrapper/device mapping"
    elif "no such file or directory" in log_lower or "filenotfounderror" in log_lower:
        run_status = "DATA_ERROR"
        error_summary = "Data file or path was not found"
    elif "unrecognized arguments" in log_lower or "argument error" in log_lower:
        run_status = "CONFIG_ERROR"
        error_summary = "Command line configuration error"
    else:
        run_status = "UNKNOWN_ERROR"
        error_summary = "training exited non-zero or metrics were not parsed"

    metrics = {
        "dataset": dataset,
        "mode": "official_baseline",
        "seed": SEED,
        "mse": parsed.get("mse"),
        "mae": parsed.get("mae"),
        "rmse": parsed.get("rmse"),
        "mape": parsed.get("mape"),
        "mspe": parsed.get("mspe"),
        "best_epoch": parsed.get("best_epoch"),
        "final_epoch": parsed.get("final_epoch"),
        "train_exit_code": exit_code,
        "elapsed_seconds": round(elapsed, 3),
        "official_protocol": True,
        "paper_reference": {},
        "checkpoint_path": parsed.get("checkpoint_path"),
        "metrics_source": parsed.get("metrics_source"),
        "parse_warnings": parsed.get("parse_warnings", []),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    status.update(
        {
            "status": run_status,
            "end_time": iso_now(),
            "exit_code": exit_code,
            "error_summary": error_summary,
            "metrics_path": str(run_dir / "metrics.json"),
        }
    )
    (run_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    resource = {
        "start_time": status["start_time"],
        "end_time": status["end_time"],
        "selected_gpu_index": selected_gpu,
        "official_gpu_argument": official_gpu_arg,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "nvidia_smi_before": before,
        "nvidia_smi_after": after,
        "peak_memory_used_mib": monitor_data.get("peak_memory_used_mib"),
        "monitor_samples": monitor_data.get("samples", 0),
        "elapsed_seconds": round(elapsed, 3),
    }
    (run_dir / "resource_usage.json").write_text(json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "dataset": dataset,
        "status": run_status,
        "run_dir": str(run_dir),
        "experiment_id": experiment_id,
        "exit_code": exit_code,
        "metrics": metrics,
    }


def plan_records() -> List[Dict[str, Any]]:
    records = []
    for cfg in DATASETS:
        fake_run = RUNS / f"timeperceiver__{cfg['dataset']}__official_baseline__seed{SEED}__<timestamp>"
        cmd = config_to_command(cfg, fake_run, 0)
        records.append(
            {
                "dataset": cfg["display"],
                "dataset_key": cfg["dataset"],
                "command_source": f"upstream/TimePerceiver/{cfg['script']} first pred_len=96 block; run.py defaults for unspecified args",
                "protocol_match": True,
                "seed": SEED,
                "expected_output_dir": str(fake_run),
                "requires_gpu": True,
                "parallel_policy": "serial by default for Gate 2 stability; max 2 allowed by instructions but not used here",
                "risk": "formal training may be long or OOM; no silent config shrinking",
                "gate1_smoke_difference": f"Gate 1 smoke used tiny smoke-only batches/steps; Gate 2 uses official train_epochs={cfg['train_epochs']}, batch_size={cfg['batch_size']}, seq_len={cfg['seq_len']}, pred_len={cfg['pred_len']}, model dimensions from script.",
                "command": f"CUDA_VISIBLE_DEVICES=0 {shell_join(cmd)}",
            }
        )
    return records


def write_plan() -> None:
    ensure_dirs()
    records = plan_records()
    (REPORTS / "gate2_baseline_plan.json").write_text(json.dumps({"created_at": iso_now(), "records": records}, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Gate 2 Official Baseline Plan",
        "",
        f"Created: {iso_now()}",
        "",
        "Scope: official TimePerceiver single-seed baselines only; no innovation, no multi-seed, no ablation.",
        "",
        "| Dataset | Command source | Protocol match | Seed | Output dir | Risk |",
        "| ------- | -------------- | -------------- | ---: | ---------- | ---- |",
    ]
    for row in records:
        lines.append(
            f"| {row['dataset']} | {row['command_source']} | {row['protocol_match']} | {row['seed']} | `{row['expected_output_dir']}` | {row['risk']} |"
        )
    lines.extend(["", "## Commands", ""])
    for row in records:
        lines.extend([f"### {row['dataset']}", "", "```bash", row["command"], "```", ""])
    (REPORTS / "gate2_baseline_plan.md").write_text("\n".join(lines))


def all_baseline_runs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for status_path in sorted(RUNS.glob("timeperceiver__*__official_baseline__seed2025__*/status.json")):
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        metrics_path = status_path.parent / "metrics.json"
        metrics: Dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                metrics = {}
        row = {**status, **{f"metric_{k}": v for k, v in metrics.items()}}
        row["run_dir"] = str(status_path.parent)
        rows.append(row)
    return rows


def smoke_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for metrics_path in sorted(RUNS.glob("timeperceiver__*__official__smoke__seed2025__*/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        status_path = metrics_path.parent / "status.json"
        status = {}
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
            except json.JSONDecodeError:
                status = {}
        rows.append({"run_dir": str(metrics_path.parent), "status": status.get("status", metrics.get("status")), **metrics})
    return rows


def get_commit() -> str:
    return cmd_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"])


def write_reports(run_results: Optional[List[Dict[str, Any]]] = None) -> None:
    ensure_dirs()
    baseline = all_baseline_runs()
    done_by_dataset = {row.get("dataset"): row for row in baseline if row.get("status") == "SUCCESS"}
    requested = [cfg["dataset"] for cfg in DATASETS]
    success = [ds for ds in requested if ds in done_by_dataset]
    failed = [
        row
        for row in baseline
        if row.get("status") in {"FAILED", "OOM", "EXTERNAL_KILL", "DATA_ERROR", "CONFIG_ERROR", "CODE_ERROR", "UNKNOWN_ERROR"}
    ]
    if len(success) == len(requested):
        gate_status = "PASS"
    elif baseline:
        gate_status = "PARTIAL"
    else:
        gate_status = "PENDING"

    status_doc = {
        "gate": "Gate 2",
        "status": gate_status,
        "timestamp": iso_now(),
        "requested_datasets": requested,
        "datasets_succeeded": success,
        "datasets_pending": [ds for ds in requested if ds not in success],
        "failed_or_oom_runs": failed,
        "official_commit": get_commit(),
        "plan": str(REPORTS / "gate2_baseline_plan.md"),
        "baseline_runs": baseline,
        "notes": [
            "Gate 2 uses official script first pred_len=96 blocks.",
            "Smoke test outputs are not formal reproduction results.",
            "Official model code was not modified by this runner.",
        ],
    }
    (REPORTS / "gate2_status.json").write_text(json.dumps(status_doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    comparison_rows = []
    for cfg in DATASETS:
        row = done_by_dataset.get(cfg["dataset"])
        comparison_rows.append(
            {
                "dataset": cfg["dataset"],
                "pred_len": cfg["pred_len"],
                "seed": SEED,
                "current_mse": row.get("metric_mse") if row else "NA",
                "current_mae": row.get("metric_mae") if row else "NA",
                "paper_mse": "NA",
                "paper_mae": "NA",
                "abs_diff_mse": "NA",
                "abs_diff_mae": "NA",
                "rel_diff_mse_pct": "NA",
                "rel_diff_mae_pct": "NA",
                "protocol_match": True,
                "notes": "paper value not filled yet; needs manual extraction from paper/table",
            }
        )
    write_csv(REPORTS / "gate2_paper_comparison.csv", comparison_rows)
    write_md_table(REPORTS / "gate2_paper_comparison.md", "Gate 2 Paper Comparison", comparison_rows)

    report_lines = [
        "# Gate 2 Official Baseline Report",
        "",
        f"Status: {gate_status}",
        f"Generated: {iso_now()}",
        "",
        "Official model code: not modified by Gate 2 runner.",
        "",
        "| Dataset | Status | Experiment ID | MSE | MAE | RMSE | Best epoch | Elapsed seconds |",
        "| ------- | ------ | ------------- | --: | --: | ---: | ---------: | --------------: |",
    ]
    for cfg in DATASETS:
        row = done_by_dataset.get(cfg["dataset"])
        if row is None:
            candidates = [r for r in baseline if r.get("dataset") == cfg["dataset"]]
            row = candidates[-1] if candidates else {}
        report_lines.append(
            f"| {cfg['display']} | {row.get('status', 'PENDING')} | `{row.get('experiment_id', '')}` | {fmt(row.get('metric_mse'))} | {fmt(row.get('metric_mae'))} | {fmt(row.get('metric_rmse'))} | {fmt(row.get('metric_best_epoch'))} | {fmt(row.get('metric_elapsed_seconds'))} |"
        )
    report_lines.extend(
        [
            "",
            "## Protocol",
            "",
            "All Gate 2 runs use the first `pred_len=96` block from the official dataset script and `run.py` defaults for omitted arguments.",
            "No smoke configuration is counted as a formal baseline.",
            "",
            "## Paper Values",
            "",
            "Paper reference values are left as `NA` until extracted from a reliable paper/table source.",
        ]
    )
    (REPORTS / "gate2_baseline_report.md").write_text("\n".join(report_lines) + "\n")

    latest_rows = []
    commit = get_commit()
    for row in smoke_rows():
        run_dir = row.get("run_dir", "")
        latest_rows.append(
            {
                "experiment_id": Path(run_dir).name,
                "dataset": row.get("dataset"),
                "mode": "smoke",
                "seed": row.get("seed", SEED),
                "status": row.get("status"),
                "mse": row.get("mse"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "mape": row.get("mape"),
                "mspe": row.get("mspe"),
                "best_epoch": row.get("best_epoch"),
                "elapsed_seconds": row.get("elapsed_seconds") or row.get("duration_seconds"),
                "run_dir": run_dir,
                "metrics_path": str(Path(run_dir) / "metrics.json") if run_dir else "",
                "checkpoint_path": row.get("checkpoint_path") or row.get("checkpoint"),
                "commit": commit,
            }
        )
    for row in baseline:
        run_dir = row.get("run_dir", "")
        latest_rows.append(
            {
                "experiment_id": row.get("experiment_id"),
                "dataset": row.get("dataset"),
                "mode": "official_baseline",
                "seed": SEED,
                "status": row.get("status"),
                "mse": row.get("metric_mse"),
                "mae": row.get("metric_mae"),
                "rmse": row.get("metric_rmse"),
                "mape": row.get("metric_mape"),
                "mspe": row.get("metric_mspe"),
                "best_epoch": row.get("metric_best_epoch"),
                "elapsed_seconds": row.get("metric_elapsed_seconds"),
                "run_dir": run_dir,
                "metrics_path": str(Path(run_dir) / "metrics.json") if run_dir else "",
                "checkpoint_path": row.get("metric_checkpoint_path"),
                "commit": commit,
            }
        )
    write_csv(REPORTS / "latest_results.csv", latest_rows)
    write_md_table(REPORTS / "latest_results.md", "Latest Results", latest_rows)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, title: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n")
        return
    keys = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k)) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n")


def run_all(force: bool = False) -> int:
    ensure_dirs()
    write_plan()
    results = []
    for cfg in DATASETS:
        result = run_one(cfg, force=force)
        results.append(result)
        write_reports(results)
    write_reports(results)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["plan", "run", "reports", "status"], default="run")
    parser.add_argument("--force", action="store_true", help="run even when a SUCCESS exists")
    args = parser.parse_args()

    if args.action == "plan":
        write_plan()
        write_reports()
    elif args.action == "run":
        return run_all(force=args.force)
    elif args.action in {"reports", "status"}:
        write_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
