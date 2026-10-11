#!/usr/bin/env python3
"""Shared immutable protocol definitions for Gate 5a."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("etth1", "weather", "electricity")
HEADS = ("gaussian_head", "laplace_head", "student_t_fixed_df8_head", "student_t_head")
NEW_HEADS = ("laplace_head", "student_t_fixed_df8_head")
REUSED_HEADS = ("gaussian_head", "student_t_head")
DISPLAY_NAMES = {
    "gaussian_head": "Gaussian",
    "laplace_head": "Laplace",
    "student_t_fixed_df8_head": "Fixed-t8",
    "student_t_head": "Learnable-t",
}
RESOURCE_FLOORS = {"etth1": 4096, "weather": 7000, "electricity": 18000}
VAL_MASKS = {
    "etth1": "artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz",
    "weather": "artifacts/gate4b_val_masks/weather/random/rate_0.3/seed_43025.npz",
    "electricity": "artifacts/gate4b_val_masks/electricity/random/rate_0.3/seed_43025.npz",
}
TEST_MASKS = {
    dataset: f"artifacts/masks/{dataset}/random/rate_0.3/seed_2025.npz" for dataset in DATASETS
}
REUSE_RUNS = {
    ("etth1", "gaussian_head"): (
        "runs/timeperceiver__etth1__gate4b_corrected_minimal__gaussian_head__"
        "random_rate0.3__seed2025__20260713_130557"
    ),
    ("etth1", "student_t_head"): (
        "runs/timeperceiver__etth1__gate4b_corrected_minimal__student_t_head__"
        "random_rate0.3__seed2025__20260713_130713"
    ),
    ("weather", "gaussian_head"): (
        "runs/timeperceiver__weather__gate4b_cross_dataset__gaussian_head__"
        "random_rate0.3__seed2025__20260715_233846"
    ),
    ("weather", "student_t_head"): (
        "runs/timeperceiver__weather__gate4b_cross_dataset__student_t_head__"
        "random_rate0.3__seed2025__20260715_234156"
    ),
    ("electricity", "gaussian_head"): (
        "runs/timeperceiver__electricity__gate4b_cross_dataset__gaussian_head__"
        "random_rate0.3__seed2025__20260716_062020"
    ),
    ("electricity", "student_t_head"): (
        "runs/timeperceiver__electricity__gate4b_cross_dataset__student_t_head__"
        "random_rate0.3__seed2025__20260716_184923"
    ),
}
LEGACY_INVALID_RUNS = (
    "runs/timeperceiver__etth1__gate4b__gaussian_head__random_rate0.3__seed2025__20260712_232425",
    "runs/timeperceiver__etth1__gate4b__student_t_head__random_rate0.3__seed2025__20260712_232450",
)
PROTECTED_FORMAL_SOURCES = (
    "results/gate4a_formal_decision.json",
    "results/gate4b_multiseed_formal_decision.json",
    "results/gate4c_formal_decision.json",
    "reports/gate4b_multiseed_results.csv",
    "reports/gate4b_multiseed_paired_deltas.csv",
    "results/final_project_status.json",
    "reports/final_project_synthesis.md",
    "results/final_evidence_registry.json",
    "reports/final_evidence_registry.csv",
)


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


def write_json_markdown(path, title, payload, note=""):
    text = f"# {title}\n\nStatus: **{payload.get('status', 'UNKNOWN')}**\n\n"
    if note:
        text += note.rstrip() + "\n\n"
    text += f"```json\n{json.dumps(payload, indent=2)}\n```\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_resolved(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            values[key.strip()] = json.loads(raw.strip())
        except Exception:
            values[key.strip()] = raw.strip().strip("\"'")
    return values


def checkpoint_inventory():
    return {row["dataset"]: row for row in load(ROOT / "results/gate3b_full_checkpoint_inventory.json")}


def dataset_spec(dataset):
    inventory = checkpoint_inventory()[dataset]
    config = load_resolved(inventory["baseline_config_path"])
    gaussian_run = ROOT / REUSE_RUNS[(dataset, "gaussian_head")]
    gaussian_metrics = load(gaussian_run / "metrics.json")
    gaussian_resolved = load_resolved(gaussian_run / "resolved_config.yaml")
    return {
        "dataset": dataset,
        "config": relative(inventory["baseline_config_path"]),
        "source_checkpoint": relative(inventory["checkpoint_path"]),
        "source_checkpoint_sha256": inventory["checkpoint_sha256"],
        "validation_mask": VAL_MASKS[dataset],
        "validation_mask_sha256": sha256_file(ROOT / VAL_MASKS[dataset]),
        "test_mask": TEST_MASKS[dataset],
        "test_mask_sha256": sha256_file(ROOT / TEST_MASKS[dataset]),
        "target_channels": int(config["enc_in"]),
        "batch_size": int(config["batch_size"]),
        "seq_len": int(config["seq_len"]),
        "pred_len": int(config["pred_len"]),
        "normalization_reference": gaussian_metrics["normalization_reference"],
        "train_mask_seed_rule": gaussian_metrics.get(
            "train_mask_seed_rule", gaussian_resolved["train_mask_seed_rule"]
        ),
        "training_protocol_hash": gaussian_metrics["training_protocol_hash"],
        "metric_protocol_hash": gaussian_metrics["metric_protocol_hash"],
    }


def task_config(dataset, variant, smoke=False):
    spec = dataset_spec(dataset)
    if variant not in HEADS:
        raise ValueError(variant)
    task = {
        "stage": "gate5a_protocol_smoke" if smoke else "gate5a",
        "dataset": dataset,
        "variant": variant,
        "distribution": {
            "gaussian_head": "normal",
            "laplace_head": "laplace",
            "student_t_fixed_df8_head": "student_t_fixed_df8",
            "student_t_head": "student_t_per_channel_learnable_df",
        }[variant],
        "missing_type": "random",
        "missing_rate": 0.3,
        "seed": 2025,
        "experiment_seed": 2025,
        "probability_head_training_seed": 2025,
        "backbone_seed_scope": "fixed_official_checkpoint",
        "seq_len": 384,
        "pred_len": 96,
        "target_channels": spec["target_channels"],
        "batch_size": spec["batch_size"],
        "config": spec["config"],
        "source_checkpoint": spec["source_checkpoint"],
        "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
        "validation_mask": spec["validation_mask"],
        "validation_mask_seed": 43025,
        "validation_mask_sha256": spec["validation_mask_sha256"],
        "test_mask": spec["test_mask"],
        "test_mask_seed": 2025,
        "test_mask_sha256": spec["test_mask_sha256"],
        "train_mask_base_seed": 42025,
        "train_mask_seed_rule": spec["train_mask_seed_rule"],
        "variant_in_mask_seed": False,
        "training_protocol_hash": spec["training_protocol_hash"],
        "metric_protocol_hash": spec["metric_protocol_hash"],
        "freeze_backbone": True,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "max_epochs": 1 if smoke else 20,
        "patience": 3,
        "early_stopping_metric": "validation_nll",
        "test_used_for_model_selection": False,
        "max_train_batches": 4 if smoke else None,
        "max_val_batches": 2 if smoke else None,
        "max_test_batches": 2 if smoke else None,
        "smoke_only": smoke,
        "required_free_mib": RESOURCE_FLOORS[dataset],
        "initial_scale": 0.5,
        "scale_floor": 1e-4,
        "initial_df": 8.0 if variant in {"student_t_fixed_df8_head", "student_t_head"} else None,
        "fixed_t_df": 8.0 if variant == "student_t_fixed_df8_head" else None,
        "learnable_df": variant == "student_t_head",
        "df_parameterization": (
            "2.01 + softplus(raw_df)" if variant == "student_t_head"
            else "constant 8.0" if variant == "student_t_fixed_df8_head" else None
        ),
        "crps_method": "empirical_monte_carlo",
        "crps_samples": 64,
        "crps_seed": 52025,
        "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95],
        "metric_space": "official_normalized",
        "normalization_reference": spec["normalization_reference"],
        "input_mask_application": "batch_x * mask before model forward",
        "mask_application_verified": True,
        "legacy_unmasked_invalid": False,
        "conformal_calibrated": False,
        "posthoc_calibration": False,
        "use_gate4a_adapter": False,
        "use_second_backbone": False,
        "official_backbone_retrained": False,
        "command": ["gate4b_formal_runner", "--task-file", "{task_file}", "--run-dir", "{run_dir}"],
    }
    return task


def corrected_peak_mib(dataset):
    return max(
        float(load(ROOT / REUSE_RUNS[(dataset, variant)] / "metrics.json")["peak_gpu_memory_mib"])
        for variant in REUSED_HEADS
    )

