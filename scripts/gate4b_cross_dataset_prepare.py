#!/usr/bin/env python
"""Prepare and audit Gate 4b Weather/Electricity fixed-mask assets."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3a_generate_missingness_assets import make_mask
from gate3b_smoke_missing_eval import (
    build_args,
    import_official_modules,
    load_checkpoint_strict,
    parse_simple_yaml,
    sha256_file,
)
from models.probabilistic_timeperceiver import Model

DATASETS = ("weather", "electricity")
VARIANTS = ("gaussian_head", "student_t_head")
VAL_SEED = 43025
TEST_SEED = 2025
RATE = 0.3
ENV_PYTHON = "/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT))


def npz_mask(path):
    with np.load(path) as archive:
        return archive["mask"]


def validate_binary(mask, expected_shape):
    return (tuple(mask.shape) == tuple(expected_shape) and mask.dtype == np.uint8
            and bool(np.logical_or(mask == 0, mask == 1).all()))


def save_validation_mask(dataset, shape, baseline_id, config, path, metadata_path):
    created = False
    if path.exists() or metadata_path.exists():
        if not path.exists() or not metadata_path.exists():
            raise RuntimeError(f"incomplete validation mask asset: {path}")
        metadata = load(metadata_path)
        mask = npz_mask(path)
        if metadata.get("sha256") != sha256_file(path):
            raise RuntimeError(f"validation metadata hash mismatch: {dataset}")
    else:
        mask = make_mask("random", VAL_SEED, tuple(shape), RATE)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mask=mask,
            mask_semantics=np.array(["1=observed/keep, 0=missing/mask"]),
            dataset=np.array([dataset]),
            mask_type=np.array(["random"]),
            target_missing_rate=np.array([RATE], dtype=np.float32),
            seed=np.array([VAL_SEED], dtype=np.int64),
            seq_len=np.array([shape[1]], dtype=np.int64),
            pred_len=np.array([96], dtype=np.int64),
        )
        metadata = {
            "shape": list(shape),
            "dtype": str(mask.dtype),
            "target_missing_rate": RATE,
            "actual_missing_rate": float(1.0 - mask.mean()),
            "sha256": sha256_file(path),
            "generator_version": "gate3a_v2_official_window_history_mask",
            "dataset": dataset,
            "baseline_experiment_id": baseline_id,
            "data_file": str(Path(config["root_path"]) / config["data_path"]),
            "seq_len": int(config["seq_len"]),
            "pred_len": int(config["pred_len"]),
            "num_features": int(config["enc_in"]),
            "num_validation_windows": int(shape[0]),
            "mask_type": "random",
            "seed": VAL_SEED,
            "mask_semantics": "1=observed/keep, 0=missing/mask",
            "applies_to": "official validation input/history window seq_x only",
            "does_not_mask": "decoder label or future prediction target",
            "split": "validation",
            "normalization_rule": "StandardScaler fit on official train split only",
            "future_target_used": False,
        }
        dump(metadata_path, metadata)
        created = True
    verified = (validate_binary(mask, shape) and metadata.get("dataset") == dataset
                and metadata.get("mask_type") == "random" and int(metadata.get("seed")) == VAL_SEED
                and abs(float(metadata.get("target_missing_rate")) - RATE) < 1e-12
                and metadata.get("split") == "validation" and metadata.get("future_target_used") is False)
    row = {
        "dataset": dataset,
        "status": "PASS" if verified else "FAIL",
        "created": created,
        "reused": not created,
        "verified": verified,
        "path": relative(path),
        "metadata_path": relative(metadata_path),
        "shape": "x".join(map(str, mask.shape)),
        "dtype": str(mask.dtype),
        "target_missing_rate": RATE,
        "actual_missing_rate": float(1.0 - mask.mean()),
        "sha256": sha256_file(path),
        "split": "validation",
        "future_target_used": False,
    }
    del mask
    return row


def task_protocol(dataset, baseline, config, validation, test, variant, smoke):
    training_protocol = {
        "dataset": dataset,
        "missing_type": "random",
        "target_missing_rate": RATE,
        "base_seed": 42025,
        "seed_rule": "sha256(base_seed|epoch|sample_index|dataset|missing_type|target_rate)",
        "seq_len": 384,
        "pred_len": 96,
        "batch_size": int(config["batch_size"]),
        "source_checkpoint_sha256": baseline["checkpoint_sha256"],
        "validation_mask_sha256": validation["sha256"],
        "test_mask_sha256": test["sha256"],
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip": 1.0,
        "max_epochs": 20,
        "patience": 3,
        "early_stopping_metric": "validation_nll",
        "freeze_backbone": True,
    }
    metric_protocol = {
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
    required = 7000 if dataset == "weather" else 10000
    task = {
        "stage": "gate4b_cross_dataset_smoke" if smoke else "gate4b_cross_dataset",
        "dataset": dataset,
        "variant": variant,
        "missing_type": "random",
        "missing_rate": RATE,
        "seed": TEST_SEED,
        "seq_len": 384,
        "pred_len": 96,
        "target_channels": int(config["enc_in"]),
        "batch_size": int(config["batch_size"]),
        "config": relative(baseline["baseline_config_path"]),
        "source_checkpoint": relative(baseline["checkpoint_path"]),
        "source_checkpoint_sha256": baseline["checkpoint_sha256"],
        "validation_mask": validation["path"],
        "validation_mask_sha256": validation["sha256"],
        "test_mask": test["path"],
        "test_mask_sha256": test["sha256"],
        "train_mask_base_seed": 42025,
        "train_mask_seed_rule": training_protocol["seed_rule"],
        "training_protocol_hash": stable_hash(training_protocol),
        "metric_protocol_hash": stable_hash(metric_protocol),
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
        "required_free_mib": required,
        "initial_scale": 0.5,
        "scale_floor": 1e-4,
        "initial_df": 8.0 if variant == "student_t_head" else None,
        "df_parameterization": "2.01 + softplus(raw_df)" if variant == "student_t_head" else None,
        "crps_method": "empirical_monte_carlo",
        "crps_samples": 64,
        "crps_seed": 52025,
        "crps_chunk_elements": 65536,
        "interval_levels": [0.5, 0.8, 0.9, 0.95],
        "metric_space": "official_normalized",
        "input_mask_application": "batch_x * mask before model forward",
        "mask_application_verified": True,
        "legacy_unmasked_invalid": False,
        "conformal_calibrated": False,
        "posthoc_calibration": False,
    }
    return task


def main():
    expanded_status = load(ROOT / "results/gate4b_expanded_status.json")
    expanded_decision = load(ROOT / "results/gate4b_expanded_formal_decision.json")
    scheduler_state = load(ROOT / "runtime/gate4b_expanded_etth1_scheduler_state.json")
    expanded_rows = list(csv.DictReader((ROOT / "reports/gate4b_expanded_results.csv").open()))
    probability_rows = [row for row in expanded_rows if row["variant"] != "official"]
    contamination = []
    for row in probability_rows:
        run = Path(row["run_dir"])
        metrics = load(run / "metrics.json")
        if metrics.get("input_mask_applied") is not True:
            contamination.append(str(run))
    prerequisite = {
        "status": "PASS",
        "expanded_scheduler_complete": scheduler_state.get("status") == "COMPLETE",
        "expanded_tasks_complete": sum(task.get("status") == "SUCCEEDED" for task in scheduler_state["tasks"].values()) == 12,
        "queue_complete_marker": "QUEUE_COMPLETE" in (ROOT / expanded_status["scheduler_log"]).read_text(),
        "GATE4B_EXPANDED_IMPLEMENTATION_PASS": expanded_decision.get("GATE4B_EXPANDED_IMPLEMENTATION_PASS"),
        "GATE4B_EXPANDED_SIGNAL_POSITIVE": expanded_decision.get("GATE4B_EXPANDED_SIGNAL_POSITIVE"),
        "ENTER_GATE4B_CROSS_DATASET_MATRIX": expanded_decision.get("ENTER_GATE4B_CROSS_DATASET_MATRIX"),
        "probability_rows": len(probability_rows),
        "invalid_unmasked_contamination": contamination,
        "historical_unmasked": {
            "INVALID_FOR_REUSE": True,
            "INVALID_FOR_FORMAL_EVIDENCE": True,
            "AUDIT_RETAINED": True,
        },
        "ENTER_GATE4C": False,
    }
    if not all((prerequisite["expanded_scheduler_complete"], prerequisite["expanded_tasks_complete"],
                prerequisite["queue_complete_marker"], prerequisite["GATE4B_EXPANDED_IMPLEMENTATION_PASS"],
                prerequisite["GATE4B_EXPANDED_SIGNAL_POSITIVE"],
                prerequisite["ENTER_GATE4B_CROSS_DATASET_MATRIX"])) or contamination:
        prerequisite["status"] = "BLOCKED_CROSS_DATASET_PREREQUISITE_INCOMPLETE"
        dump(ROOT / "results/gate4b_cross_dataset_prerequisite.json", prerequisite)
        raise RuntimeError(prerequisite["status"])
    dump(ROOT / "results/gate4b_cross_dataset_prerequisite.json", prerequisite)

    checkpoint_inventory = {row["dataset"]: row for row in load(ROOT / "results/gate3b_full_checkpoint_inventory.json")}
    gate3b_rows = list(csv.DictReader((ROOT / "reports/gate3b_full_results.csv").open()))
    provider, official_module = import_official_modules()
    architecture_rows = []
    checkpoint_rows = []
    validation_rows = []
    test_rows = []
    official_rows = []
    configs = {}
    baselines = {}

    for dataset in DATASETS:
        baseline = dict(checkpoint_inventory[dataset])
        baseline["checkpoint_path"] = Path(baseline["checkpoint_path"])
        baseline["baseline_config_path"] = Path(baseline["baseline_config_path"])
        baselines[dataset] = baseline
        config = parse_simple_yaml(baseline["baseline_config_path"])
        configs[dataset] = config
        args = build_args(config, int(config["batch_size"]), "cpu")
        train_data, _ = provider(args, "train")
        val_data, _ = provider(args, "val")
        test_data, _ = provider(args, "test")
        checkpoint_hash = sha256_file(baseline["checkpoint_path"])
        official = official_module.Model(args).float()
        strict = load_checkpoint_strict(official, baseline["checkpoint_path"])
        raw = torch.load(baseline["checkpoint_path"], map_location="cpu")
        if any("probabilistic_head" in key for key in raw):
            raise RuntimeError(f"probability weights found in official checkpoint: {dataset}")
        gaussian = Model(args, "gaussian_head", True).float()
        student = Model(args, "student_t_head", True).float()
        raw_df_shape = list(student.probabilistic_head.raw_df.shape)
        architecture_ok = (
            str(config["dataset"]).lower() == dataset
            and config["features"] == "M"
            and int(config["seq_len"]) == 384
            and int(config["pred_len"]) == 96
            and gaussian.probabilistic_head.loc_projection.out_features == int(config["patch_len"])
            and gaussian.probabilistic_head.scale_projection.out_features == int(config["patch_len"])
            and raw_df_shape == [1, 1, int(config["enc_in"])]
            and student.probabilistic_head.raw_df.numel() == int(config["enc_in"])
        )
        architecture_rows.append({
            "dataset": dataset, "status": "PASS" if architecture_ok else "FAIL",
            "dataset_identifier": config["dataset"], "data_path": str(Path(config["root_path"]) / config["data_path"]),
            "features": config["features"], "seq_len": int(config["seq_len"]),
            "pred_len": int(config["pred_len"]), "input_channels": int(config["enc_in"]),
            "target_channels": int(config["enc_in"]), "target_slicing": "batch_y[:, -pred_len:, :]",
            "hidden_dimension": int(gaussian.embed_dim), "patch_len": int(config["patch_len"]),
            "output_projection_shape": list(official.output_projection.weight.shape),
            "gaussian_loc_projection_shape": list(gaussian.probabilistic_head.loc_projection.weight.shape),
            "gaussian_scale_projection_shape": list(gaussian.probabilistic_head.scale_projection.weight.shape),
            "student_t_raw_df_shape": raw_df_shape, "student_t_extra_parameters": int(config["enc_in"]),
            "normalization_reference": "official train split StandardScaler",
            "metric_space": "official_normalized", "official_batch_size": int(config["batch_size"]),
            "split_lengths": {"train": len(train_data), "validation": len(val_data), "test": len(test_data)},
        })
        checkpoint_ok = (baseline["status"] == "SUCCESS" and checkpoint_hash == baseline["checkpoint_sha256"]
                         and bool(strict.get("strict_load", True)))
        checkpoint_rows.append({
            "dataset": dataset, "status": "PASS" if checkpoint_ok else "FAIL",
            "official_experiment_id": baseline["experiment_id"], "run_dir": baseline["run_dir"],
            "checkpoint_path": str(baseline["checkpoint_path"]), "checkpoint_sha256": checkpoint_hash,
            "strict_load": True, "tensor_count": len(raw), "clean_mse": baseline["baseline_mse"],
            "clean_mae": baseline["baseline_mae"], "clean_rmse": baseline["baseline_rmse"],
            "gate4a_adapter": False, "probability_head_weights": False,
        })
        del official, gaussian, student, raw, train_data

        val_path = ROOT / f"artifacts/gate4b_val_masks/{dataset}/random/rate_0.3/seed_43025.npz"
        val_meta = val_path.with_name("seed_43025.metadata.json")
        validation_rows.append(save_validation_mask(
            dataset, (len(val_data), int(config["seq_len"]), int(config["enc_in"])),
            baseline["experiment_id"], config, val_path, val_meta))

        test_path = ROOT / f"artifacts/masks/{dataset}/random/rate_0.3/seed_2025.npz"
        test_meta_path = test_path.with_suffix(".json")
        test_meta = load(test_meta_path)
        test_mask = npz_mask(test_path)
        expected_test_shape = (len(test_data), int(config["seq_len"]), int(config["enc_in"]))
        test_hash = sha256_file(test_path)
        verified = (validate_binary(test_mask, expected_test_shape) and test_meta["sha256"] == test_hash
                    and test_meta["dataset"] == dataset and test_meta["mask_type"] == "random"
                    and int(test_meta["seed"]) == TEST_SEED and test_meta["split"]["num_test_windows"] == len(test_data)
                    and test_meta["applies_to"].startswith("test input/history")
                    and "future" in test_meta["does_not_mask"])
        test_rows.append({
            "dataset": dataset, "status": "PASS" if verified else "FAIL", "verified": verified,
            "path": relative(test_path), "metadata_path": relative(test_meta_path),
            "shape": "x".join(map(str, test_mask.shape)), "dtype": str(test_mask.dtype),
            "target_missing_rate": RATE, "actual_missing_rate": float(1.0 - test_mask.mean()),
            "sha256": test_hash, "split": "test", "future_target_used": False,
            "hash_mismatch": test_meta["sha256"] != test_hash, "metadata_mismatch": not verified,
        })
        del test_mask, val_data, test_data

        official = next(row for row in gate3b_rows if row["dataset"] == dataset
                        and row["missing_type"] == "random" and float(row["target_missing_rate"]) == RATE
                        and int(row["seed"]) == TEST_SEED)
        official_rows.append({
            "dataset": dataset, "variant": "official", "status": official["status"],
            "experiment_id": official["experiment_id"], "mse": float(official["mse"]),
            "mae": float(official["mae"]), "rmse": float(official["rmse"]),
            "actual_missing_rate": float(official["actual_missing_rate"]),
            "sample_count": int(official["sample_count"]), "batch_count": int(official["batch_count"]),
            "checkpoint_sha256": official["checkpoint_sha256"], "test_mask_sha256": official["mask_sha256"],
            "input_mask_applied": True, "NLL": "NA", "CRPS": "NA", "PICP": "NA",
            "MPIW": "NA", "coverage_error": "NA", "interval_score": "NA", "scale": "NA", "df": "NA",
            "run_dir": official["run_dir"],
        })

    if any(row["status"] != "PASS" for row in architecture_rows):
        raise RuntimeError("BLOCKED_CROSS_DATASET_ARCHITECTURE_UNRESOLVED")
    if any(row["status"] != "PASS" for row in checkpoint_rows):
        raise RuntimeError("BLOCKED_CROSS_DATASET_CHECKPOINT_FAILURE")
    if any(row["status"] != "PASS" for row in validation_rows + test_rows):
        raise RuntimeError("BLOCKED_CROSS_DATASET_MASK_FAILURE")

    reuse_rows = []
    formal_tasks = []
    smoke_tasks = []
    for dataset in DATASETS:
        validation = next(row for row in validation_rows if row["dataset"] == dataset)
        test = next(row for row in test_rows if row["dataset"] == dataset)
        for variant in VARIANTS:
            formal = task_protocol(dataset, baselines[dataset], configs[dataset], validation, test, variant, False)
            smoke = task_protocol(dataset, baselines[dataset], configs[dataset], validation, test, variant, True)
            formal_tasks.append(formal)
            smoke_tasks.append(smoke)
            formal_path = ROOT / f"configs/gate4b_cross_dataset/{dataset}_{variant}_random_rate0.3_seed2025.json"
            smoke_path = ROOT / f"configs/gate4b_cross_dataset_smoke/{dataset}_{variant}_random_rate0.3_seed2025.json"
            dump(formal_path, formal)
            dump(smoke_path, smoke)
            candidates = []
            pattern = f"timeperceiver__{dataset}__gate4b*__{variant}__random_rate0.3__seed2025__*/metrics.json"
            for metrics_path in (ROOT / "runs").glob(pattern):
                metrics = load(metrics_path)
                if (metrics.get("smoke_only") is False and metrics.get("input_mask_applied") is True
                        and metrics.get("mask_application_verified") is True
                        and metrics.get("training_protocol_hash") == formal["training_protocol_hash"]
                        and metrics.get("metric_protocol_hash") == formal["metric_protocol_hash"]
                        and metrics.get("source_checkpoint_sha256") == formal["source_checkpoint_sha256"]
                        and metrics.get("validation_mask_sha256") == formal["validation_mask_sha256"]
                        and metrics.get("test_mask_sha256") == formal["test_mask_sha256"]):
                    candidates.append(metrics_path.parent)
            reusable = str(sorted(candidates)[-1].relative_to(ROOT)) if candidates else ""
            reuse_rows.append({
                "dataset": dataset, "variant": variant, "reusable": bool(reusable),
                "run_dir": reusable, "reason": "corrected_fixed_mask_protocol_match" if reusable else "no_matching_corrected_formal_run",
                "input_mask_applied": bool(reusable), "mask_application_verified": bool(reusable),
                "test_mask_sha256": formal["test_mask_sha256"],
                "actual_test_missing_rate": test["actual_missing_rate"],
                "legacy_unmasked_invalid": False,
            })

    outputs = [
        ("architecture_audit", {"status": "PASS", "rows": architecture_rows}),
        ("checkpoint_audit", {"status": "PASS", "expected": 2, "verified": 2, "rows": checkpoint_rows}),
        ("val_mask_inventory", {"status": "PASS", "expected": 2,
                                "created_or_reused": len(validation_rows), "verified": 2, "rows": validation_rows}),
        ("test_mask_inventory", {"status": "PASS", "expected": 2, "verified": 2,
                                 "hash_mismatch": 0, "metadata_mismatch": 0, "rows": test_rows}),
        ("official_inventory", {"status": "PASS", "expected": 2, "verified": 2, "rows": official_rows}),
        ("reuse_inventory", {"status": "PASS", "expected_probability_tasks": 4,
                             "reused_probability_tasks": sum(row["reusable"] for row in reuse_rows), "rows": reuse_rows}),
    ]
    for name, payload in outputs:
        dump(ROOT / f"results/gate4b_cross_dataset_{name}.json", payload)
        rows = payload.get("rows", [])
        if rows:
            write_csv(ROOT / f"reports/gate4b_cross_dataset_{name}.csv", rows)
        (ROOT / f"reports/gate4b_cross_dataset_{name}.md").write_text(
            f"# Gate 4b Cross-Dataset {name.replace('_', ' ').title()}\n\n"
            f"Status: **{payload['status']}**\n\n```json\n{json.dumps(payload, indent=2)}\n```\n")

    status = {
        "stage": "Gate 4b cross-dataset", "status": "PLANNING_PASS",
        "prerequisite": "PASS", "architecture_audit": "PASS", "checkpoint_audit": "PASS 2/2",
        "validation_masks": "PASS 2/2", "test_masks": "PASS 2/2", "official_inventory": "PASS 2/2",
        "protocol_smoke": "NOT_STARTED", "reuse_audit": "PASS",
        "logical_probability_tasks": 4, "reused_tasks": sum(row["reusable"] for row in reuse_rows),
        "active_tasks": 4 - sum(row["reusable"] for row in reuse_rows), "queue": "NOT_CREATED",
        "scheduler": "NOT_STARTED", "historical_unmasked_invalid": True,
        "ENTER_GATE4C": False,
    }
    dump(ROOT / "results/gate4b_cross_dataset_status.json", status)
    (ROOT / "reports/gate4b_cross_dataset_status.md").write_text(
        "# Gate 4b Cross-Dataset Status\n\nStatus: **PLANNING_PASS**\n\n"
        "Preconditions, architecture, checkpoints, masks, official baselines, and reuse audit passed. "
        "Protocol smoke is the next gate. Historical unmasked Gate 4b evidence remains invalid.\n")
    print(json.dumps({"status": "PASS", "datasets": 2, "validation_masks": 2,
                      "test_masks": 2, "smoke_tasks": len(smoke_tasks),
                      "reused": status["reused_tasks"]}, indent=2))


if __name__ == "__main__":
    main()
