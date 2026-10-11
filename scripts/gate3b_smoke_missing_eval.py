#!/usr/bin/env python
"""Gate 3b-smoke fixed-mask missingness evaluation.

Evaluation-only wrapper for ETTh1 + random rate 0.3 + seed 2025. It reuses the
official TimePerceiver model, dataset loader, checkpoint, and test slicing while
applying the fixed Gate 3a mask only to the test history input `seq_x`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "TimePerceiver"
ENV_PYTHON = Path("/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python")
OFFICIAL_COMMIT = "7e30cc07b51c709f408409fd60a34c81ae8990be"


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_output(cmd: List[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip()
    except Exception as exc:
        return f"COMMAND_FAILED {' '.join(cmd)}: {exc!r}"


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip().strip("\"'")
        if value == "":
            continue
        if value.lower() == "true":
            values[key] = True
        elif value.lower() == "false":
            values[key] = False
        else:
            try:
                values[key] = int(value)
            except ValueError:
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = value
    return values


def latest_success_baseline(dataset: str) -> Path:
    prefix = f"timeperceiver__{dataset}__official_baseline__seed2025__"
    candidates: List[Path] = []
    for run_dir in (ROOT / "runs").glob(prefix + "*"):
        status_path = run_dir / "status.json"
        if not status_path.exists():
            continue
        try:
            status = read_json(status_path)
        except json.JSONDecodeError:
            continue
        if status.get("status") == "SUCCESS":
            candidates.append(run_dir)
    if not candidates:
        raise RuntimeError(f"No SUCCESS Gate 2 baseline run found for {dataset}")
    return sorted(candidates, key=lambda p: p.name)[-1]


def locate_checkpoint(baseline_run: Path) -> Path:
    metrics_path = baseline_run / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        checkpoint = metrics.get("checkpoint_path")
        if checkpoint and Path(checkpoint).exists():
            return Path(checkpoint)
    candidates = list((baseline_run / "checkpoints").rglob("checkpoint.pth"))
    candidates += list((baseline_run / "checkpoints").rglob("*.pt"))
    candidates += list((baseline_run / "checkpoints").rglob("*.pth"))
    candidates = sorted(set(candidates), key=lambda p: (-p.stat().st_size, str(p)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Could not locate checkpoint under {baseline_run}")


def build_args(config: Dict[str, Any], eval_batch_size: int, device_mode: str) -> SimpleNamespace:
    defaults: Dict[str, Any] = {
        "is_training": 0,
        "model_id": "ETTh1_384_96",
        "model": "TimePerceiver",
        "data": "ETTh1",
        "root_path": str(ROOT / "data" / "ETT-small") + "/",
        "data_path": "ETTh1.csv",
        "features": "M",
        "target": "OT",
        "checkpoints": "./checkpoints/",
        "freq": "h",
        "augmentation_ratio": 0,
        "seq_len": 384,
        "label_len": 0,
        "pred_len": 96,
        "inverse": False,
        "seasonal_patterns": "Monthly",
        "enc_in": 7,
        "d_model": 256,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.2,
        "embed": "timeF",
        "num_workers": 0,
        "itr": 1,
        "train_epochs": 50,
        "warmup_epochs": 5,
        "batch_size": eval_batch_size,
        "patience": 10,
        "learning_rate": 0.0005,
        "weight_decay": 0.05,
        "des": "gate3b_missing_eval_smoke",
        "loss": "MSE",
        "lradj": "type3",
        "use_amp": False,
        "use_gpu": False,
        "gpu": 0,
        "gpu_type": "cuda",
        "use_multi_gpu": False,
        "devices": "0",
        "num_latents": 32,
        "latent_dim": 64,
        "latent_d_ff": 128,
        "use_latent": 1,
        "generalized": 1,
        "standard": 0,
        "num_latent_blocks": 1,
        "query_share": 1,
        "patch_len": 12,
        "separate_ratio": 1.0,
        "output_attention": False,
    }
    defaults.update(config)
    defaults["is_training"] = 0
    defaults["num_workers"] = 0
    defaults["batch_size"] = int(eval_batch_size)
    defaults["use_multi_gpu"] = False
    defaults["output_attention"] = False
    defaults["inverse"] = bool(defaults.get("inverse", False))
    if device_mode == "cuda" and torch.cuda.is_available():
        defaults["use_gpu"] = True
        defaults["gpu"] = 0
        defaults["gpu_type"] = "cuda"
    else:
        defaults["use_gpu"] = False
        defaults["gpu"] = 0
        defaults["gpu_type"] = "cpu"
    return SimpleNamespace(**defaults)


def import_official_modules() -> Tuple[Any, Any]:
    sys.path.insert(0, str(UPSTREAM))
    from data_provider.data_factory import data_provider  # type: ignore
    import importlib.util
    path = UPSTREAM / "models" / "TimePerceiver.py"
    spec = importlib.util.spec_from_file_location("official_timeperceiver_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return data_provider, module


def load_checkpoint_strict(model: torch.nn.Module, checkpoint: Path) -> Dict[str, Any]:
    obj = torch.load(checkpoint, map_location="cpu")
    state = obj
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint object type: {type(state)}")
    clean = {}
    for key, value in state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        clean[clean_key] = value
    model.load_state_dict(clean, strict=True)
    return {"strict": True, "num_tensors": len(clean)}


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, data: Dict[str, Any] | List[Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_run_scaffolding(run_dir: Path, command: str) -> None:
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    write_text(run_dir / "command.sh", command)
    os.chmod(run_dir / "command.sh", 0o755)
    write_text(run_dir / "train.log", "Gate 3b-smoke is evaluation-only. No training was started.\n")
    env_lines = [
        f"timestamp: {iso_now()}",
        f"hostname: {socket.gethostname()}",
        f"python: {sys.version.split()[0]}",
        f"torch: {torch.__version__}",
        f"torch_cuda_runtime: {torch.version.cuda}",
        f"cuda_available: {torch.cuda.is_available()}",
        f"device_count: {torch.cuda.device_count()}",
        f"nvidia_smi: {run_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader'])}",
    ]
    write_text(run_dir / "env.txt", "\n".join(env_lines) + "\n")
    git_state = "\n".join(
        [
            run_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
            run_output(["git", "-C", str(UPSTREAM), "status", "--short"]),
            run_output(["git", "-C", str(UPSTREAM), "diff", "--stat"]),
        ]
    )
    write_text(run_dir / "git_state.txt", git_state + "\n")
    write_text(run_dir / "code.diff", run_output(["git", "-C", str(UPSTREAM), "diff", "--binary"]) + "\n")


def append_latest_results(run_dir: Path, metrics: Dict[str, Any], status: str) -> None:
    latest = ROOT / "reports" / "latest_results.csv"
    fieldnames = [
        "experiment_id",
        "dataset",
        "mode",
        "seed",
        "status",
        "mse",
        "mae",
        "rmse",
        "mape",
        "mspe",
        "best_epoch",
        "elapsed_seconds",
        "run_dir",
        "metrics_path",
        "checkpoint_path",
        "commit",
    ]
    rows: List[Dict[str, Any]] = []
    if latest.exists():
        with latest.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
    rows.append(
        {
            "experiment_id": run_dir.name,
            "dataset": metrics.get("dataset", "etth1"),
            "mode": "official_missing_eval_smoke",
            "seed": 2025,
            "status": status,
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "mape": "",
            "mspe": "",
            "best_epoch": "",
            "elapsed_seconds": metrics.get("elapsed_seconds_wall"),
            "run_dir": str(run_dir),
            "metrics_path": str(run_dir / "metrics.json"),
            "checkpoint_path": metrics.get("checkpoint"),
            "commit": OFFICIAL_COMMIT,
        }
    )
    with latest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(run_dir: Path, status_payload: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    gate_status = {
        "gate": "Gate 3b-smoke",
        "status": "PASS" if status_payload["status"] == "SUCCESS" else "BLOCKED",
        "timestamp": iso_now(),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": "etth1",
        "mask_type": "random",
        "missing_rate": 0.3,
        "mask_seed": 2025,
        "smoke_only": True,
        "not_full_gate3b_result": True,
        "metrics_path": str(run_dir / "metrics.json"),
        "next_gate": "Gate 3b-full" if status_payload["status"] == "SUCCESS" else "Fix Gate 3b-smoke blocker",
        "error_summary": status_payload.get("error_summary", ""),
    }
    write_json(reports / "gate3b_smoke_status.json", gate_status)

    md = [
        "# Gate 3b-smoke Missingness Evaluation Report",
        "",
        f"- Status: **{gate_status['status']}**",
        f"- Run ID: `{run_dir.name}`",
        f"- Run dir: `{run_dir}`",
        "- Dataset: `ETTh1`",
        "- Mask: `random`, rate `0.3`, seed `2025`",
        "- Smoke only: `true`",
        "- Full Gate 3b result: `false`",
        "",
        "## Metrics",
        "",
        "| MSE | MAE | RMSE | Evaluated batches | Evaluated samples |",
        "|---:|---:|---:|---:|---:|",
        f"| {metrics.get('mse')} | {metrics.get('mae')} | {metrics.get('rmse')} | {metrics.get('evaluated_batches')} | {metrics.get('evaluated_samples')} |",
        "",
        "## Mask",
        "",
        f"- mask path: `{metrics.get('mask_path')}`",
        f"- metadata actual missing rate: `{metrics.get('actual_missing_rate_metadata')}`",
        f"- evaluated missing rate mean: `{metrics.get('actual_missing_rate_evaluated_mean')}`",
        "",
        "## Baseline Checkpoint",
        "",
        f"- baseline run: `{metrics.get('baseline_run')}`",
        f"- checkpoint: `{metrics.get('checkpoint')}`",
        "",
        "## Safety Checks",
        "",
        "| Check | Status |",
        "|---|---|",
        "| Training started | NO |",
        "| Official model structure modified | NO |",
        "| Mask applied only to seq_x/history | YES |",
        "| Future prediction target masked | NO |",
        "| Evaluation-time resampling | NO |",
        "| Entered Gate 4 | NO |",
    ]
    if gate_status["status"] != "PASS":
        md += ["", "## Blocker", "", f"`{status_payload.get('error_summary', '')}`"]
    write_text(reports / "gate3b_smoke_report.md", "\n".join(md) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="etth1")
    parser.add_argument("--mask_path", default="artifacts/masks/etth1/random/rate_0.3/seed_2025.npz")
    parser.add_argument("--mask_meta", default="artifacts/masks/etth1/random/rate_0.3/seed_2025.json")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--max_eval_batches", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args_cli = parser.parse_args()

    start_time = iso_now()
    start_wall = datetime.now()
    run_dir = Path(args_cli.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    command = (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        f"cd {ROOT}\n"
        f"{ENV_PYTHON} scripts/gate3b_smoke_missing_eval.py "
        f"--dataset {args_cli.dataset} --mask_path {args_cli.mask_path} "
        f"--mask_meta {args_cli.mask_meta} --run_dir {run_dir} "
        f"--max_eval_batches {args_cli.max_eval_batches} --eval_batch_size {args_cli.eval_batch_size} "
        f"--device {args_cli.device}\n"
    )
    save_run_scaffolding(run_dir, command)
    eval_log = run_dir / "eval.log"
    metrics: Dict[str, Any] = {}
    status = "BLOCKED"
    error_summary = ""

    try:
        print(f"Gate 3b-smoke eval-only run: {run_dir.name}", flush=True)
        print("No training will be started.", flush=True)
        baseline_run = latest_success_baseline(args_cli.dataset)
        checkpoint = locate_checkpoint(baseline_run)
        config = parse_simple_yaml(baseline_run / "resolved_config.yaml")
        eval_args = build_args(config, args_cli.eval_batch_size, args_cli.device)

        mask_path = ROOT / args_cli.mask_path
        mask_meta_path = ROOT / args_cli.mask_meta
        mask_meta = read_json(mask_meta_path)
        loaded = np.load(mask_path, allow_pickle=False)
        mask = loaded["mask"].astype(np.float32)
        if mask.ndim != 3:
            raise RuntimeError(f"Mask must be 3D [N,T,C], got {mask.shape}")
        if set(np.unique(mask).tolist()) - {0.0, 1.0}:
            raise RuntimeError("Mask contains values outside {0,1}")

        shutil.copy2(mask_meta_path, artifacts_dir / "used_mask_metadata.json")
        write_text(artifacts_dir / "used_checkpoint.txt", str(checkpoint) + "\n")

        data_provider, model_module = import_official_modules()
        test_data, test_loader = data_provider(eval_args, "test")
        if getattr(test_loader, "sampler", None).__class__.__name__ not in {"SequentialSampler"}:
            raise RuntimeError(f"Expected deterministic SequentialSampler, got {test_loader.sampler}")
        if len(test_data) != mask.shape[0]:
            raise RuntimeError(f"Mask windows {mask.shape[0]} != test dataset len {len(test_data)}")
        if int(eval_args.seq_len) != mask.shape[1]:
            raise RuntimeError(f"Mask seq_len {mask.shape[1]} != seq_len {eval_args.seq_len}")
        if int(eval_args.enc_in) != mask.shape[2]:
            raise RuntimeError(f"Mask features {mask.shape[2]} != enc_in {eval_args.enc_in}")

        device = torch.device("cuda:0" if args_cli.device == "cuda" and torch.cuda.is_available() else "cpu")
        model = model_module.Model(eval_args).float().to(device)
        load_info = load_checkpoint_strict(model, checkpoint)
        model.eval()

        sse = 0.0
        sae = 0.0
        count = 0
        evaluated_batches = 0
        evaluated_samples = 0
        mask_rates: List[float] = []
        batch_debug: List[Dict[str, Any]] = []
        global_start = 0
        pred_len = int(eval_args.pred_len)
        label_len = int(eval_args.label_len)
        patch_len = int(eval_args.patch_len)
        indices = (
            list(range(0, int(eval_args.seq_len) // patch_len)),
            list(range(int(eval_args.seq_len) // patch_len, (int(eval_args.seq_len) + pred_len) // patch_len)),
        )

        with torch.no_grad():
            for batch_id, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if batch_id >= int(args_cli.max_eval_batches):
                    break
                batch_size = int(batch_x.shape[0])
                mask_batch_np = mask[global_start : global_start + batch_size]
                if mask_batch_np.shape[0] != batch_size:
                    raise RuntimeError(f"Mask batch truncated at batch {batch_id}")
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                mask_batch = torch.from_numpy(mask_batch_np).float().to(device)
                if tuple(mask_batch.shape) != tuple(batch_x.shape):
                    raise RuntimeError(f"Mask batch shape {tuple(mask_batch.shape)} != batch_x {tuple(batch_x.shape)}")

                batch_x_masked = batch_x * mask_batch
                dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1).float().to(device)
                outputs = model(batch_x_masked, batch_x_mark, dec_inp, batch_y_mark, indices)

                outputs = outputs[:, -pred_len:, :]
                target = batch_y[:, -pred_len:, :]
                f_dim = -1 if getattr(eval_args, "features", "M") == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                target = target[:, :, f_dim:]

                diff = outputs - target
                sse += float(torch.sum(diff * diff).item())
                sae += float(torch.sum(torch.abs(diff)).item())
                count += int(diff.numel())
                evaluated_batches += 1
                evaluated_samples += batch_size
                mask_rate = float(1.0 - mask_batch.mean().item())
                mask_rates.append(mask_rate)
                if batch_id < 3:
                    batch_debug.append(
                        {
                            "batch_id": batch_id,
                            "global_start": global_start,
                            "batch_size": batch_size,
                            "batch_x_shape": list(batch_x.shape),
                            "batch_y_shape": list(batch_y.shape),
                            "mask_shape": list(mask_batch.shape),
                            "mask_actual_missing_rate": mask_rate,
                            "target_unchanged": True,
                        }
                    )
                global_start += batch_size

        if evaluated_batches == 0 or count == 0:
            raise RuntimeError("No smoke evaluation batches were processed")

        mse = sse / count
        mae = sae / count
        rmse = math.sqrt(mse)
        elapsed = (datetime.now() - start_wall).total_seconds()
        metrics = {
            "dataset": args_cli.dataset,
            "mode": "official_missing_eval_smoke",
            "smoke_only": True,
            "not_full_gate3b_result": True,
            "max_eval_batches": int(args_cli.max_eval_batches),
            "evaluated_batches": evaluated_batches,
            "evaluated_samples": evaluated_samples,
            "mask_type": mask_meta.get("mask_type", "random"),
            "target_missing_rate": mask_meta.get("target_missing_rate", 0.3),
            "actual_missing_rate_metadata": mask_meta.get("actual_missing_rate"),
            "actual_missing_rate_evaluated_mean": float(np.mean(mask_rates)),
            "mask_seed": mask_meta.get("seed", 2025),
            "mask_path": str(mask_path),
            "mask_meta": str(mask_meta_path),
            "mask_sha256": sha256_file(mask_path),
            "baseline_run": baseline_run.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_load_info": load_info,
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "seq_len": int(eval_args.seq_len),
            "label_len": int(eval_args.label_len),
            "pred_len": pred_len,
            "features": getattr(eval_args, "features", None),
            "batch_size_eval": int(eval_args.batch_size),
            "device": str(device),
            "train_started": False,
            "model_code_modified": False,
            "future_target_masked": False,
            "mask_resampled": False,
            "start_time": start_time,
            "end_time": iso_now(),
            "elapsed_seconds_wall": elapsed,
        }
        write_json(artifacts_dir / "batch_debug.json", batch_debug)
        write_text(
            run_dir / "resolved_config.yaml",
            "\n".join(f"{k}: {v}" for k, v in sorted(vars(eval_args).items())) + "\n",
        )
        status = "SUCCESS"
    except Exception as exc:
        error_summary = repr(exc)
        metrics = {
            "dataset": args_cli.dataset,
            "mode": "official_missing_eval_smoke",
            "smoke_only": True,
            "not_full_gate3b_result": True,
            "mse": None,
            "mae": None,
            "rmse": None,
            "train_started": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "start_time": start_time,
            "end_time": iso_now(),
            "elapsed_seconds_wall": (datetime.now() - start_wall).total_seconds(),
        }

    write_json(run_dir / "metrics.json", metrics)
    status_payload = {
        "experiment_id": run_dir.name,
        "status": status,
        "dataset": args_cli.dataset,
        "mode": "official_missing_eval_smoke",
        "start_time": start_time,
        "end_time": metrics.get("end_time", iso_now()),
        "exit_code": 0 if status == "SUCCESS" else 2,
        "error_summary": error_summary,
        "metrics_path": str(run_dir / "metrics.json"),
    }
    write_json(run_dir / "status.json", status_payload)
    write_json(
        run_dir / "resource_usage.json",
        {
            "experiment_id": run_dir.name,
            "dataset": args_cli.dataset,
            "mode": "official_missing_eval_smoke",
            "train_started": False,
            "device": metrics.get("device", args_cli.device),
            "cuda_available": torch.cuda.is_available(),
            "elapsed_seconds_wall": metrics.get("elapsed_seconds_wall"),
        },
    )
    write_reports(run_dir, status_payload, metrics)
    append_latest_results(run_dir, metrics, status)
    print(json.dumps({"status": status, "run_dir": str(run_dir), "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0 if status == "SUCCESS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
