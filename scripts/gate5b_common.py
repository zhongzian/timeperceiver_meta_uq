#!/usr/bin/env python3
"""Immutable protocol and utility definitions for Gate 5b."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python")
PATCHTST = ROOT / "upstream" / "PatchTST"
TIMEPERCEIVER = ROOT / "upstream" / "TimePerceiver"
PATCHTST_COMMIT = "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
TIMEPERCEIVER_COMMIT = "7e30cc07b51c709f408409fd60a34c81ae8990be"
DATASETS = ("etth1", "weather", "electricity")
DISPLAY = {"etth1": "ETTh1", "weather": "Weather", "electricity": "Electricity"}
BATCH_SIZES = {"etth1": 32, "weather": 16, "electricity": 8}
RESOURCE_FLOORS = {"etth1": 8192, "weather": 10000, "electricity": 20000}
TARGET_CHANNELS = {"etth1": 7, "weather": 21, "electricity": 321}
VAL_MASKS = {
    "etth1": "artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz",
    "weather": "artifacts/gate4b_val_masks/weather/random/rate_0.3/seed_43025.npz",
    "electricity": "artifacts/gate4b_val_masks/electricity/random/rate_0.3/seed_43025.npz",
}
TEST_MASKS = {
    dataset: f"artifacts/masks/{dataset}/random/rate_0.3/seed_2025.npz" for dataset in DATASETS
}
FORMAL_MASK_INVENTORY = "configs/queues/gate5a_distribution_ablation.yaml"
TRAIN_MASK_RULES = {
    "etth1": "sha256(base_seed|epoch|sample_index|missing_type|target_rate)",
    "weather": "sha256(base_seed|epoch|sample_index|dataset|missing_type|target_rate)",
    "electricity": "sha256(base_seed|epoch|sample_index|dataset|missing_type|target_rate)",
}
BASELINE_CONFIGS = {
    "etth1": "runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/resolved_config.yaml",
    "weather": "runs/timeperceiver__weather__official_baseline__seed2025__20260701_203558/resolved_config.yaml",
    "electricity": "runs/timeperceiver__electricity__official_baseline__seed2025__20260705_090044/resolved_config.yaml",
}
OFFICIAL_SCRIPTS = {
    dataset: f"upstream/PatchTST/PatchTST_supervised/scripts/PatchTST/{name}.sh"
    for dataset, name in {"etth1": "etth1", "weather": "weather", "electricity": "electricity"}.items()
}

# Values are resolved from each dataset's official supervised pred_len=96 script.
OFFICIAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "etth1": {
        "enc_in": 7, "seq_len": 336, "pred_len": 96, "e_layers": 3,
        "n_heads": 4, "d_model": 16, "d_ff": 128, "dropout": 0.3,
        "fc_dropout": 0.3, "head_dropout": 0.0, "patch_len": 16,
        "stride": 8, "train_epochs": 100, "patience": 100,
        "batch_size": 128, "learning_rate": 1e-4, "lradj": "type3",
        "pct_start": 0.3,
    },
    "weather": {
        "enc_in": 21, "seq_len": 336, "pred_len": 96, "e_layers": 3,
        "n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
        "fc_dropout": 0.2, "head_dropout": 0.0, "patch_len": 16,
        "stride": 8, "train_epochs": 100, "patience": 20,
        "batch_size": 128, "learning_rate": 1e-4, "lradj": "type3",
        "pct_start": 0.3,
    },
    "electricity": {
        "enc_in": 321, "seq_len": 336, "pred_len": 96, "e_layers": 3,
        "n_heads": 16, "d_model": 128, "d_ff": 256, "dropout": 0.2,
        "fc_dropout": 0.2, "head_dropout": 0.0, "patch_len": 16,
        "stride": 8, "train_epochs": 100, "patience": 10,
        "batch_size": 32, "learning_rate": 1e-4, "lradj": "TST",
        "pct_start": 0.2,
    },
}

ARCHITECTURE_DEFAULTS = {
    "padding_patch": "end", "revin": True, "affine": False,
    "subtract_last": False, "individual": False,
}
BACKBONE_TRAINING_PROTOCOL = {
    "backbone": "patchtst", "seq_len": 384, "pred_len": 96,
    "missingness": "none", "loss": "MSE", "optimizer": "Adam",
    "gradient_clip": 1.0, "early_stopping_metric": "validation_mse",
    "test_used_for_model_selection": False, "experiment_seed": 2025,
}
PROBABILITY_TRAINING_PROTOCOL = {
    "backbone": "patchtst", "seq_len": 384, "pred_len": 96,
    "missing_type": "random", "missing_rate": 0.3, "experiment_seed": 2025,
    "train_mask_base_seed": 42025, "freeze_backbone": True,
    "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
    "gradient_clip": 1.0, "max_epochs": 20, "patience": 3,
    "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
    "initial_scale": 0.5, "scale_floor": 1e-4, "fixed_t_df": 8.0,
}
METRIC_PROTOCOL = {
    "implementation": "scripts/gate5a_distribution_metrics.py",
    "point_prediction": "loc", "nll_reduction": "elementwise_mean",
    "crps_method": "empirical_monte_carlo", "crps_samples": 64,
    "crps_seed": 52025, "crps_chunk_elements": 65536,
    "interval_levels": [0.5, 0.8, 0.9, 0.95], "conformal": False,
    "posthoc_calibration": False,
}


def run_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"COMMAND_FAILED: {exc!r}"


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, title, payload, intro=""):
    text = f"# {title}\n\nStatus: **{payload.get('status', 'UNKNOWN')}**\n\n"
    if intro:
        text += intro.rstrip() + "\n\n"
    text += "```json\n" + json.dumps(payload, indent=2) + "\n```\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_resolved(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = json.loads(raw.strip())
        except Exception:
            values[key.strip()] = raw.strip().strip("\"'")
    return values


def write_resolved(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(f"{key}: {json.dumps(value)}" for key, value in sorted(values.items())) + "\n")


def dataset_spec(dataset):
    if dataset not in DATASETS:
        raise ValueError(dataset)
    base = load_resolved(ROOT / BASELINE_CONFIGS[dataset])
    inventory = load(ROOT / FORMAL_MASK_INVENTORY)
    inventory_task = next(
        task for task in inventory["tasks"]
        if task["dataset"] == dataset and task["variant"] == "gaussian_head"
    )
    for key in ("validation_mask", "test_mask"):
        expected_hash = inventory_task[f"{key}_sha256"]
        actual_hash = sha256_file(ROOT / inventory_task[key])
        if actual_hash != expected_hash:
            raise RuntimeError(f"formal mask inventory hash mismatch for {dataset} {key}")
    architecture = {**OFFICIAL_CONFIGS[dataset], **ARCHITECTURE_DEFAULTS}
    architecture.update({"seq_len": 384, "pred_len": 96, "batch_size": BATCH_SIZES[dataset]})
    return {
        "dataset": dataset,
        "display_name": DISPLAY[dataset],
        "baseline_config": BASELINE_CONFIGS[dataset],
        "data": base["data"], "root_path": base["root_path"], "data_path": base["data_path"],
        "features": base.get("features", "M"), "target": base.get("target", "OT"),
        "freq": base.get("freq", "h"), "enc_in": int(base["enc_in"]),
        "seq_len": 384, "pred_len": 96, "batch_size": BATCH_SIZES[dataset],
        "validation_mask": inventory_task["validation_mask"],
        "validation_mask_sha256": inventory_task["validation_mask_sha256"],
        "test_mask": inventory_task["test_mask"],
        "test_mask_sha256": inventory_task["test_mask_sha256"],
        "formal_mask_inventory": FORMAL_MASK_INVENTORY,
        "train_mask_seed_rule": inventory_task["train_mask_seed_rule"],
        "official_source_script": OFFICIAL_SCRIPTS[dataset],
        "architecture": architecture,
    }


def backbone_config(dataset, smoke=False):
    spec = dataset_spec(dataset)
    official = spec["architecture"]
    config = {
        "stage": "gate5b0_backbone_smoke" if smoke else "gate5b0_backbone",
        "substage": "deterministic_clean_backbone", "backbone": "patchtst",
        "patchtst_upstream_commit": PATCHTST_COMMIT, "dataset": dataset,
        "config": spec["baseline_config"], "seq_len": 384, "pred_len": 96,
        "enc_in": spec["enc_in"], "batch_size": spec["batch_size"],
        "missingness": "none", "experiment_seed": 2025, "seed": 2025,
        "loss": "MSE", "optimizer": "Adam", "learning_rate": official["learning_rate"],
        "gradient_clip": 1.0, "max_epochs": 1 if smoke else official["train_epochs"],
        "patience": official["patience"], "lradj": official["lradj"],
        "pct_start": official["pct_start"], "early_stopping_metric": "validation_mse",
        "test_used_for_model_selection": False,
        "max_train_batches": 4 if smoke else None,
        "max_val_batches": 2 if smoke else None,
        "max_test_batches": 2 if smoke else None,
        "smoke_only": smoke, "validation_mask": spec["validation_mask"],
        "validation_mask_sha256": spec["validation_mask_sha256"],
        "test_mask": spec["test_mask"], "test_mask_sha256": spec["test_mask_sha256"],
        "training_protocol_hash": stable_hash({**BACKBONE_TRAINING_PROTOCOL, "dataset": dataset, **official}),
        "metric_protocol_hash": stable_hash(METRIC_PROTOCOL),
        "required_free_mib": RESOURCE_FLOORS[dataset],
        "command": ["gate5b_train_backbone", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
        **{key: value for key, value in official.items() if key not in {"batch_size", "seq_len", "pred_len"}},
    }
    return config


def probability_config(dataset, distribution, smoke=False, checkpoint=None, checkpoint_sha256=None):
    if distribution not in {"gaussian", "fixed_t8"}:
        raise ValueError(distribution)
    spec = dataset_spec(dataset)
    architecture = spec["architecture"]
    config = {
        "stage": "gate5b1_probability_smoke" if smoke else "gate5b1_probability",
        "substage": "probability_replication", "backbone": "patchtst",
        "patchtst_upstream_commit": PATCHTST_COMMIT, "dataset": dataset,
        "variant": "gaussian_head" if distribution == "gaussian" else "student_t_fixed_df8_head",
        "distribution": distribution, "config": spec["baseline_config"],
        "seq_len": 384, "pred_len": 96, "enc_in": spec["enc_in"],
        "batch_size": spec["batch_size"], "missing_type": "random", "missing_rate": 0.3,
        "experiment_seed": 2025, "seed": 2025, "train_mask_base_seed": 42025,
        "train_mask_seed_rule": spec["train_mask_seed_rule"],
        "validation_mask": spec["validation_mask"],
        "validation_mask_sha256": spec["validation_mask_sha256"],
        "test_mask": spec["test_mask"], "test_mask_sha256": spec["test_mask_sha256"],
        "source_backbone_checkpoint": checkpoint or "__FROM_GATE5B0_MANIFEST__",
        "source_backbone_checkpoint_sha256": checkpoint_sha256 or "__FROM_GATE5B0_MANIFEST__",
        "freeze_backbone": True, "optimizer": "AdamW", "learning_rate": 1e-3,
        "weight_decay": 1e-4, "gradient_clip": 1.0,
        "max_epochs": 1 if smoke else 20, "patience": 3,
        "early_stopping_metric": "validation_nll", "test_used_for_model_selection": False,
        "max_train_batches": 4 if smoke else None, "max_val_batches": 2 if smoke else None,
        "max_test_batches": 2 if smoke else None, "smoke_only": smoke,
        "initial_scale": 0.5, "scale_floor": 1e-4,
        "fixed_t_df": 8.0 if distribution == "fixed_t8" else None,
        "learnable_df": False, "crps_method": "empirical_monte_carlo",
        "crps_samples": 64, "crps_seed": 52025, "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95], "conformal": False,
        "posthoc_calibration": False, "use_adapter": False,
        "training_protocol_hash": stable_hash({**PROBABILITY_TRAINING_PROTOCOL, "dataset": dataset}),
        "metric_protocol_hash": stable_hash(METRIC_PROTOCOL),
        "required_free_mib": RESOURCE_FLOORS[dataset],
        "command": ["gate5b_train_probability", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
        **{key: architecture[key] for key in (
            "e_layers", "n_heads", "d_model", "d_ff", "dropout", "fc_dropout",
            "head_dropout", "patch_len", "stride", "padding_patch", "revin",
            "affine", "subtract_last", "individual",
        )},
    }
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_text():
    return (
        f"hostname: {socket.gethostname()}\npython: {sys.version.split()[0]}\n"
        f"torch: {torch.__version__}\ncuda_runtime: {torch.version.cuda}\n"
        f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n"
        f"gpu: {run_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,memory.free', '--format=csv,noheader'])}\n"
    )


def upstream_state_text():
    chunks = []
    for name, path in (("TimePerceiver", TIMEPERCEIVER), ("PatchTST", PATCHTST)):
        chunks.append(f"[{name}]\ncommit={run_output(['git', '-C', str(path), 'rev-parse', 'HEAD'])}")
        chunks.append("status:\n" + run_output(["git", "-C", str(path), "status", "--short"]))
        chunks.append("diff_stat:\n" + run_output(["git", "-C", str(path), "diff", "--stat"]))
    return "\n".join(chunks) + "\n"


def upstream_diff_text():
    return (
        "# TimePerceiver tracked diff\n" + run_output(["git", "-C", str(TIMEPERCEIVER), "diff", "--binary"])
        + "\n# PatchTST tracked diff\n" + run_output(["git", "-C", str(PATCHTST), "diff", "--binary"]) + "\n"
    )
