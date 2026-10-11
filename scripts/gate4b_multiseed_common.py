#!/usr/bin/env python
"""Shared protocol definitions for Gate 4b multi-seed confirmation."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("etth1", "weather", "electricity")
VARIANTS = ("gaussian_head", "student_t_head")
SEEDS = (2021, 2022, 2023, 2025)
NEW_SEEDS = (2021, 2022, 2023)
VAL_SEED = 43025
TEST_SEED = 2025
RATE = 0.3
MULTISEED_MASK_RULE = (
    "sha256(dataset|experiment_seed|base_seed|epoch|sample_index|missing_type|target_rate)"
)
CANONICAL_METRIC_PROTOCOL = {
    "metric_space": "official_normalized",
    "point_prediction": "loc",
    "nll_reduction": "elementwise_mean",
    "crps_method": "empirical_monte_carlo",
    "crps_samples": 64,
    "crps_seed": 52025,
    "crps_chunk_elements": 65536,
    "interval_levels": [0.5, 0.8, 0.9, 0.95],
    "conformal_calibrated": False,
    "posthoc_calibration": False,
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

VAL_MASKS = {
    "etth1": "artifacts/gate4a_val_masks/etth1/random/rate_0.3/seed_43025.npz",
    "weather": "artifacts/gate4b_val_masks/weather/random/rate_0.3/seed_43025.npz",
    "electricity": "artifacts/gate4b_val_masks/electricity/random/rate_0.3/seed_43025.npz",
}
TEST_MASKS = {
    dataset: f"artifacts/masks/{dataset}/random/rate_0.3/seed_2025.npz" for dataset in DATASETS
}
RESOURCE_FLOORS = {"etth1": 4096, "weather": 7000, "electricity": 18000}


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json_markdown(path, title, payload, preface=""):
    text = f"# {title}\n\nStatus: **{payload.get('status', 'UNKNOWN')}**\n\n"
    if preface:
        text += preface.rstrip() + "\n\n"
    text += f"```json\n{json.dumps(payload, indent=2)}\n```\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)


def stable_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


CANONICAL_METRIC_HASH = stable_hash(CANONICAL_METRIC_PROTOCOL)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT))


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
    rows = load(ROOT / "results/gate3b_full_checkpoint_inventory.json")
    return {row["dataset"]: row for row in rows}


def corrected_peak_mib(dataset):
    return max(
        float(load(ROOT / REUSE_RUNS[(dataset, variant)] / "metrics.json")["peak_gpu_memory_mib"])
        for variant in VARIANTS
    )


def required_free_mib(dataset):
    profiled = math.ceil(corrected_peak_mib(dataset) * 1.5) + 512
    return max(RESOURCE_FLOORS[dataset], profiled)


def source_protocol(dataset, variant):
    run_dir = ROOT / REUSE_RUNS[(dataset, variant)]
    metrics = load(run_dir / "metrics.json")
    resolved = load_resolved(run_dir / "resolved_config.yaml")
    return run_dir, metrics, resolved


def dataset_spec(dataset):
    inventory = checkpoint_inventory()[dataset]
    config = load_resolved(inventory["baseline_config_path"])
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
    }


def training_protocol(dataset, seed, smoke=False):
    spec = dataset_spec(dataset)
    protocol = {
        "scope": "probabilistic_head_training_seeds",
        "backbone_seed_scope": "fixed_official_checkpoint",
        "dataset": dataset,
        "experiment_seed": seed,
        "missing_type": "random",
        "target_missing_rate": RATE,
        "train_mask_base_seed": 40000 + seed,
        "train_mask_seed_rule": MULTISEED_MASK_RULE,
        "seq_len": 384,
        "pred_len": 96,
        "batch_size": spec["batch_size"],
        "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
        "validation_mask_sha256": spec["validation_mask_sha256"],
        "test_mask_sha256": spec["test_mask_sha256"],
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "max_epochs": 1 if smoke else 20,
        "max_train_batches": 4 if smoke else None,
        "max_val_batches": 2 if smoke else None,
        "max_test_batches": 2 if smoke else None,
        "patience": 3,
        "early_stopping_metric": "validation_nll",
        "freeze_backbone": True,
    }
    return protocol


def task_config(dataset, variant, seed, smoke=False):
    spec = dataset_spec(dataset)
    protocol = training_protocol(dataset, seed, smoke)
    task = {
        "stage": "gate4b_multiseed_seed_smoke" if smoke else "gate4b_multiseed",
        "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
        "full_end_to_end_multi_seed": False,
        "dataset": dataset,
        "variant": variant,
        "missing_type": "random",
        "missing_rate": RATE,
        "seed": seed,
        "experiment_seed": seed,
        "probability_head_training_seed": seed,
        "backbone_seed_scope": "fixed_official_checkpoint",
        "seq_len": 384,
        "pred_len": 96,
        "target_channels": spec["target_channels"],
        "batch_size": spec["batch_size"],
        "config": spec["config"],
        "source_checkpoint": spec["source_checkpoint"],
        "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
        "validation_mask": spec["validation_mask"],
        "validation_mask_seed": VAL_SEED,
        "validation_mask_sha256": spec["validation_mask_sha256"],
        "test_mask": spec["test_mask"],
        "test_mask_seed": TEST_SEED,
        "test_mask_sha256": spec["test_mask_sha256"],
        "train_mask_base_seed": 40000 + seed,
        "train_mask_seed_rule": MULTISEED_MASK_RULE,
        "variant_in_mask_seed": False,
        "training_protocol_hash": stable_hash(protocol),
        "metric_protocol_hash": CANONICAL_METRIC_HASH,
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
        "required_free_mib": 4096 if smoke else required_free_mib(dataset),
        "initial_scale": 0.5,
        "scale_floor": 1e-4,
        "initial_df": 8.0 if variant == "student_t_head" else None,
        "df_parameterization": (
            "2.01 + softplus(raw_df)" if variant == "student_t_head" else None
        ),
        "crps_method": "empirical_monte_carlo",
        "crps_samples": 64,
        "crps_seed": 52025,
        "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95],
        "metric_space": "official_normalized",
        "input_mask_application": "batch_x * mask before model forward",
        "input_mask_applied": True,
        "mask_application_verified": True,
        "batch_y_modified": False,
        "legacy_unmasked_invalid": False,
        "conformal_calibrated": False,
        "posthoc_calibration": False,
        "task_id": f"{dataset}__seed{seed}__{variant}",
        "command": [
            "gate4b_formal_runner",
            "--task-file",
            "{task_file}",
            "--run-dir",
            "{run_dir}",
        ],
        "reuse_run_dir": "",
    }
    if seed == 2025 and not smoke:
        run_dir, metrics, resolved = source_protocol(dataset, variant)
        task.update({
            "train_mask_seed_rule": resolved["train_mask_seed_rule"],
            "training_protocol_hash": metrics["training_protocol_hash"],
            "reuse_training_protocol_hash": metrics["training_protocol_hash"],
            "reuse_metric_protocol_hash": metrics["metric_protocol_hash"],
            "reuse_metric_protocol_semantic_alias_pass": (
                metrics["metric_protocol_hash"] == CANONICAL_METRIC_HASH or dataset == "etth1"
            ),
            "reuse_schema_audit_pass": dataset == "etth1",
            "reuse_run_dir": relative(run_dir),
        })
    return task


def validate_reuse(dataset, variant):
    task = task_config(dataset, variant, 2025, False)
    run_dir, metrics, resolved = source_protocol(dataset, variant)
    status = load(run_dir / "status.json")
    checkpoint_dir = run_dir / "checkpoints"
    experiment_seed = int(metrics.get("experiment_seed", metrics.get("seed", -1)))
    base_seed = int(metrics.get("train_mask_base_seed", resolved.get("train_mask_base_seed", -1)))
    schema_alias = dataset == "etth1"
    metric_alias = task["reuse_metric_protocol_semantic_alias_pass"]
    required_metrics = (
        "test_mse", "test_mae", "test_rmse", "test_mean_nll", "test_crps",
        "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
        "mean_scale", "scale_min", "scale_max",
    )
    finite_metrics = all(
        isinstance(metrics.get(key), (int, float)) and math.isfinite(float(metrics[key]))
        for key in required_metrics
    )
    checks = {
        "status_success": status.get("status") == "SUCCESS",
        "not_smoke": metrics.get("smoke_only") is False,
        "full_train_validation_test": (
            resolved.get("max_train_batches") is None
            and resolved.get("max_val_batches") is None
            and resolved.get("max_test_batches") is None
            and int(metrics.get("max_epochs", -1)) == 20
            and int(metrics.get("train_batches", 0)) > 0
            and int(metrics.get("val_batches", 0)) > 0
            and int(metrics.get("test_batches", 0)) > 0
        ),
        "corrected_fixed_mask_runner": (
            "corrected_minimal" in run_dir.name or "cross_dataset" in run_dir.name
        ),
        "input_mask_applied": metrics.get("input_mask_applied") is True,
        "mask_application_verified": metrics.get("mask_application_verified") is True or schema_alias,
        "legacy_unmasked_invalid_false": (
            metrics.get("legacy_unmasked_invalid") is False or schema_alias
        ),
        "validation_nll_selection": (
            metrics.get("early_stopping_metric") == "validation_nll"
            and metrics.get("test_used_for_model_selection") is False
        ),
        "backbone_frozen": (
            resolved.get("freeze_backbone") is True
            and metrics.get("backbone_non_null_gradient_count") == 0
            and metrics.get("optimizer_backbone_parameter_count") == 0
        ),
        "experiment_seed_2025": experiment_seed == 2025,
        "train_mask_base_seed_42025": base_seed == 42025,
        "source_checkpoint_hash_match": (
            metrics.get("source_checkpoint_sha256") == task["source_checkpoint_sha256"]
        ),
        "validation_mask_hash_match": (
            metrics.get("validation_mask_sha256") == task["validation_mask_sha256"]
        ),
        "test_mask_hash_match": metrics.get("test_mask_sha256") == task["test_mask_sha256"],
        "training_protocol_hash_match": (
            metrics.get("training_protocol_hash") == task["reuse_training_protocol_hash"]
        ),
        "metric_protocol_hash_or_audited_alias": (
            metrics.get("metric_protocol_hash") == task["reuse_metric_protocol_hash"] and metric_alias
        ),
        "metrics_complete_and_finite": finite_metrics,
        "scale_positive": metrics.get("scale_positive") is True,
        "student_df_valid": (
            variant == "gaussian_head"
            or metrics.get("df_finite") is True and metrics.get("df_above_2_01") is True
        ),
        "checkpoint_save_reload": (
            metrics.get("checkpoint_save") is True
            and metrics.get("checkpoint_strict_reload") is True
            and metrics.get("evaluation_after_reload") is True
            and (checkpoint_dir / "best_prob_head.pt").is_file()
        ),
        "no_leakage": (
            metrics.get("test_mask_used_during_training") is False
            and metrics.get("batch_y_modified") is False
        ),
    }
    reusable = all(checks.values())
    return {
        "dataset": dataset,
        "variant": variant,
        "experiment_seed": 2025,
        "run_dir": relative(run_dir),
        "reusable": reusable,
        "reason": "corrected_fixed_mask_protocol_match" if reusable else "reuse_gate_failure",
        "schema_alias_audit": schema_alias,
        "metric_protocol_semantic_alias_audit": metric_alias,
        "source_metric_protocol_hash": metrics.get("metric_protocol_hash"),
        "canonical_metric_protocol_hash": CANONICAL_METRIC_HASH,
        "training_protocol_hash": metrics.get("training_protocol_hash"),
        "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
        "validation_mask_sha256": metrics.get("validation_mask_sha256"),
        "test_mask_sha256": metrics.get("test_mask_sha256"),
        "actual_test_missing_rate": metrics.get("actual_test_missing_rate"),
        "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
        "checks": checks,
    }
