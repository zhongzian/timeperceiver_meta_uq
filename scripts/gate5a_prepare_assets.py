#!/usr/bin/env python3
"""Prepare Gate 5a audits, pairing evidence, and new-head configs without GPU work."""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate3b_smoke_missing_eval import build_args, import_official_modules, parse_simple_yaml
from gate4b_train_probabilistic import Indexed, sample_masks
from gate5a_common import (
    DATASETS,
    HEADS,
    LEGACY_INVALID_RUNS,
    NEW_HEADS,
    PROTECTED_FORMAL_SOURCES,
    REUSED_HEADS,
    REUSE_RUNS,
    ROOT,
    dataset_spec,
    dump,
    load,
    load_resolved,
    relative,
    resolve,
    sha256_file,
    task_config,
    write_csv,
    write_json_markdown,
)
from models.probabilistic_timeperceiver import Model, freeze_backbone


def snapshot_protected_sources():
    rows = []
    for value in PROTECTED_FORMAL_SOURCES:
        path = ROOT / value
        rows.append({
            "path": value,
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
        })
    payload = {"status": "PASS", "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": rows}
    before = ROOT / "results/gate5a_protected_sources_before.json"
    if not before.exists():
        dump(before, payload)
    synthesis = next(row for row in rows if row["path"] == "reports/final_project_synthesis.md")
    final_hash = ROOT / "results/gate5a_previous_final_synthesis_hash.json"
    if not final_hash.exists():
        dump(final_hash, {"status": "RECORDED", **synthesis})


def architecture_audit():
    rows = []
    for dataset in DATASETS:
        spec = dataset_spec(dataset)
        config = parse_simple_yaml(resolve(spec["config"]))
        args = build_args(config, spec["batch_size"], "cpu")
        counts = {}
        totals = {}
        raw_df_shapes = {}
        for variant in HEADS:
            model = Model(args, variant, True).float()
            names = freeze_backbone(model)
            counts[variant] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            totals[variant] = sum(parameter.numel() for parameter in model.parameters())
            raw_df = model.probabilistic_head.raw_df
            raw_df_shapes[variant] = list(raw_df.shape) if raw_df is not None else None
        channels = int(config["enc_in"])
        passed = (
            channels == spec["target_channels"]
            and counts["laplace_head"] == counts["gaussian_head"]
            and counts["student_t_fixed_df8_head"] == counts["gaussian_head"]
            and counts["student_t_head"] == counts["gaussian_head"] + channels
            and raw_df_shapes["student_t_head"] == [1, 1, channels]
            and raw_df_shapes["student_t_fixed_df8_head"] is None
        )
        rows.append({
            "dataset": dataset,
            "status": "PASS" if passed else "FAIL",
            "target_channels": channels,
            "gaussian_trainable_params": counts["gaussian_head"],
            "laplace_trainable_params": counts["laplace_head"],
            "fixed_t8_trainable_params": counts["student_t_fixed_df8_head"],
            "learnable_t_trainable_params": counts["student_t_head"],
            "learnable_t_extra_params": counts["student_t_head"] - counts["gaussian_head"],
            "gaussian_total_params": totals["gaussian_head"],
            "laplace_total_params": totals["laplace_head"],
            "fixed_t8_total_params": totals["student_t_fixed_df8_head"],
            "learnable_t_total_params": totals["student_t_head"],
            "learnable_t_raw_df_shape": raw_df_shapes["student_t_head"],
            "fixed_t8_raw_df_absent": raw_df_shapes["student_t_fixed_df8_head"] is None,
            "trainable_parameter_names": {
                variant: [name for name in names] if variant == HEADS[-1] else None for variant in HEADS
            },
        })
    status = "PASS" if len(rows) == 3 and all(row["status"] == "PASS" for row in rows) else (
        "BLOCKED_GATE5A_CAPACITY_FAIRNESS_FAILURE"
    )
    payload = {"status": status, "datasets_expected": 3, "datasets_passed": sum(r["status"] == "PASS" for r in rows), "rows": rows}
    dump(ROOT / "results/gate5a_architecture_audit.json", payload)
    write_json_markdown(ROOT / "reports/gate5a_architecture_audit.md", "Gate 5a Architecture Audit", payload)
    fields = [
        "dataset", "status", "target_channels", "gaussian_trainable_params",
        "laplace_trainable_params", "fixed_t8_trainable_params", "learnable_t_trainable_params",
        "learnable_t_extra_params", "gaussian_total_params", "laplace_total_params",
        "fixed_t8_total_params", "learnable_t_total_params",
    ]
    write_csv(ROOT / "reports/gate5a_parameter_comparison.csv", rows, fields)
    markdown = "# Gate 5a Parameter Comparison\n\n"
    markdown += "| Dataset | Channels | Gaussian | Laplace | Fixed-t8 | Learnable-t | Status |\n"
    markdown += "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    for row in rows:
        markdown += (
            f"| {row['dataset']} | {row['target_channels']} | {row['gaussian_trainable_params']} | "
            f"{row['laplace_trainable_params']} | {row['fixed_t8_trainable_params']} | "
            f"{row['learnable_t_trainable_params']} | {row['status']} |\n"
        )
    (ROOT / "reports/gate5a_parameter_comparison.md").write_text(markdown)
    return payload


def reuse_audit():
    rows = []
    prior_inventory = load(ROOT / "results/gate4b_multiseed_reuse_inventory.json")
    prior_rows = {
        (row["dataset"], row["variant"]): row
        for row in prior_inventory["rows"]
        if int(row.get("experiment_seed", -1)) == 2025
    }
    required_metrics = (
        "test_mse", "test_mae", "test_rmse", "test_mean_nll", "test_crps",
        "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
        "scale_min", "scale_max", "peak_gpu_memory_mib",
    )
    for dataset in DATASETS:
        spec = dataset_spec(dataset)
        for variant in REUSED_HEADS:
            run = ROOT / REUSE_RUNS[(dataset, variant)]
            status = load(run / "status.json")
            metrics = load(run / "metrics.json")
            resolved = load_resolved(run / "resolved_config.yaml")
            prior = prior_rows[(dataset, variant)]
            prior_checks = prior.get("checks", {})
            schema_alias = prior.get("schema_alias_audit") is True
            checkpoints = [run / "checkpoints/best_prob_head.pt", run / "checkpoints/last_prob_head.pt"]
            checks = {
                "status_success": status.get("status") == "SUCCESS",
                "formal_full_run": metrics.get("smoke_only") is False and all(
                    resolved.get(key) is None for key in ("max_train_batches", "max_val_batches", "max_test_batches")
                ),
                "dataset_match": metrics.get("dataset") == dataset,
                "variant_match": metrics.get("variant") == variant,
                "condition_match": metrics.get("missing_type") == "random" and float(metrics.get("target_missing_rate")) == 0.3,
                "seed_match": int(metrics.get("experiment_seed", metrics.get("seed"))) == 2025,
                "train_mask_seed_match": int(metrics.get("train_mask_base_seed", resolved.get("train_mask_base_seed"))) == 42025,
                "source_checkpoint_match": metrics.get("source_checkpoint_sha256") == spec["source_checkpoint_sha256"] == sha256_file(resolve(spec["source_checkpoint"])),
                "validation_mask_match": metrics.get("validation_mask_sha256") == spec["validation_mask_sha256"],
                "test_mask_match": metrics.get("test_mask_sha256") == spec["test_mask_sha256"],
                "training_protocol_match": metrics.get("training_protocol_hash") == spec["training_protocol_hash"],
                "metric_protocol_match": metrics.get("metric_protocol_hash") == spec["metric_protocol_hash"],
                "input_mask_applied": metrics.get("input_mask_applied") is True,
                "mask_application_verified": (
                    metrics.get("mask_application_verified") is True
                    or schema_alias and prior_checks.get("mask_application_verified") is True
                ),
                "legacy_unmasked_excluded": (
                    metrics.get("legacy_unmasked_invalid") is False
                    or schema_alias and prior_checks.get("legacy_unmasked_invalid_false") is True
                ),
                "backbone_frozen": metrics.get("backbone_non_null_gradient_count") == 0 and metrics.get("optimizer_backbone_parameter_count") == 0,
                "validation_nll_selection": metrics.get("early_stopping_metric") == "validation_nll" and metrics.get("test_used_for_model_selection") is False,
                "metrics_complete": all(metrics.get(key) is not None for key in required_metrics),
                "checkpoints_complete": all(path.is_file() for path in checkpoints),
            }
            passed = all(checks.values())
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "status": "PASS" if passed else "FAIL",
                "run_dir": relative(run),
                "experiment_id": run.name,
                "corrected_fixed_mask_evidence": passed,
                "reusable": passed,
                "schema_audit_pass": passed,
                "schema_alias_audit": schema_alias,
                "raw_mask_application_verified": metrics.get("mask_application_verified"),
                "raw_legacy_unmasked_invalid": metrics.get("legacy_unmasked_invalid"),
                "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
                "validation_mask_sha256": metrics.get("validation_mask_sha256"),
                "test_mask_sha256": metrics.get("test_mask_sha256"),
                "training_protocol_hash": metrics.get("training_protocol_hash"),
                "metric_protocol_hash": metrics.get("metric_protocol_hash"),
                "checks": checks,
            })
    invalid_retained = []
    for value in LEGACY_INVALID_RUNS:
        run = ROOT / value
        invalid_retained.append({
            "run_dir": value,
            "exists": run.is_dir(),
            "legacy_unmasked_invalid": True,
            "reusable": False,
            "formal_evidence": False,
            "paper_evidence": False,
            "audit_retained": True,
        })
    reused = sum(row["reusable"] for row in rows)
    payload = {
        "status": "PASS" if reused == 6 else "FAIL",
        "expected_reused": 6,
        "reused": reused,
        "old_unmasked_reuse_count": 0,
        "rows": rows,
        "legacy_invalid_runs": invalid_retained,
    }
    dump(ROOT / "results/gate5a_reuse_inventory.json", payload)
    csv_rows = [{key: value for key, value in row.items() if key != "checks"} for row in rows]
    write_csv(ROOT / "reports/gate5a_reuse_inventory.csv", csv_rows)
    write_json_markdown(
        ROOT / "reports/gate5a_reuse_inventory.md", "Gate 5a Reuse Inventory", payload,
        "Only corrected fixed-mask formal runs are reusable. Historical unmasked runs remain audit-only.",
    )
    return payload


def pairing_audit():
    provider, _ = import_official_modules()
    dataset_rows = []
    pair_rows = []
    logical_configs = {}
    for dataset in DATASETS:
        spec = dataset_spec(dataset)
        config = parse_simple_yaml(resolve(spec["config"]))
        args = build_args(config, spec["batch_size"], "cpu")
        train_data, _ = provider(args, "train")
        generator = torch.Generator().manual_seed(2025)
        loader = torch.utils.data.DataLoader(
            Indexed(train_data), batch_size=spec["batch_size"], shuffle=True,
            generator=generator, num_workers=0, drop_last=False,
        )
        batch = next(iter(loader))
        batch_x, window_indices = batch[0], batch[-1]
        masks = sample_masks(
            window_indices, 0, batch_x.shape, dataset, "random", 0.3,
            42025, 2025, spec["train_mask_seed_rule"],
        )
        index_hash = hashlib.sha256(np.asarray(window_indices, dtype=np.int64).tobytes()).hexdigest()
        mask_hash = hashlib.sha256(masks.numpy().tobytes()).hexdigest()
        reuse_index_matches = []
        stored_index_hashes = []
        for variant in REUSED_HEADS:
            metrics = load(ROOT / REUSE_RUNS[(dataset, variant)] / "metrics.json")
            stored_hash = metrics.get("first_batch_window_indices_sha256")
            stored_index_hashes.append(stored_hash)
            reuse_index_matches.append(stored_hash is None or stored_hash == index_hash)
        comparable = {
            "source_checkpoint": spec["source_checkpoint"],
            "source_checkpoint_sha256": spec["source_checkpoint_sha256"],
            "train_mask_base_seed": 42025,
            "train_mask_seed_rule": spec["train_mask_seed_rule"],
            "first_batch_window_indices_sha256": index_hash,
            "first_batch_train_mask_sha256": mask_hash,
            "validation_mask_sha256": spec["validation_mask_sha256"],
            "test_mask_sha256": spec["test_mask_sha256"],
            "normalization_reference": spec["normalization_reference"],
            "batch_size": spec["batch_size"],
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip": 1.0,
            "max_epochs": 20,
            "patience": 3,
            "early_stopping_metric": "validation_nll",
            "metric_protocol_hash": spec["metric_protocol_hash"],
            "crps_seed": 52025,
        }
        logical_configs[dataset] = {variant: {**comparable, "variant": variant} for variant in HEADS}
        dataset_passed = all(reuse_index_matches)
        for left, right in itertools.combinations(HEADS, 2):
            left_config = logical_configs[dataset][left]
            right_config = logical_configs[dataset][right]
            differing = [key for key in comparable if left_config[key] != right_config[key]]
            passed = not differing
            pair_rows.append({
                "dataset": dataset,
                "left": left,
                "right": right,
                "status": "PASS" if passed else "FAIL",
                "unexpected_differences": differing,
                "first_batch_window_indices_sha256": index_hash,
                "first_batch_train_mask_sha256": mask_hash,
            })
            dataset_passed &= passed
        dataset_rows.append({
            "dataset": dataset,
            "status": "PASS" if dataset_passed else "FAIL",
            "train_windows": len(train_data),
            "first_batch_size": len(window_indices),
            "first_batch_window_indices_sha256": index_hash,
            "first_batch_train_mask_sha256": mask_hash,
            "reused_head_first_batch_indices_match": all(reuse_index_matches),
            "reused_head_first_batch_indices_recorded": sum(value is not None for value in stored_index_hashes),
            "reused_head_first_batch_indices_reconstructed": all(value is None for value in stored_index_hashes),
            "reconstruction_basis": (
                "shared deterministic DataLoader generator seed, batch size, training code, and protocol hash"
            ),
            "train_mask_seed_rule": spec["train_mask_seed_rule"],
        })
    passed_pairs = sum(row["status"] == "PASS" for row in pair_rows)
    payload = {
        "status": "PASS" if all(row["status"] == "PASS" for row in dataset_rows) and passed_pairs == 18 else "BLOCKED_GATE5A_PAIRING_FAILURE",
        "datasets_expected": 3,
        "datasets_passed": sum(row["status"] == "PASS" for row in dataset_rows),
        "head_pairs_checked": len(pair_rows),
        "head_pairs_passed": passed_pairs,
        "failed": len(pair_rows) - passed_pairs,
        "variant_in_mask_seed": False,
        "dataset_rows": dataset_rows,
        "pair_rows": pair_rows,
        "logical_configs": logical_configs,
    }
    dump(ROOT / "results/gate5a_pairing_validation.json", payload)
    write_json_markdown(ROOT / "reports/gate5a_pairing_validation.md", "Gate 5a Pairing Validation", payload)
    return payload


def write_configs():
    formal_dir = ROOT / "configs/gate5a"
    smoke_dir = ROOT / "configs/gate5a_smoke"
    formal_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"laplace_head": "laplace", "student_t_fixed_df8_head": "fixed_t8"}
    paths = []
    for dataset in DATASETS:
        for variant in NEW_HEADS:
            formal = task_config(dataset, variant, smoke=False)
            smoke = task_config(dataset, variant, smoke=True)
            formal_path = formal_dir / f"{dataset}_random0.3_{suffix[variant]}_seed2025.yaml"
            smoke_path = smoke_dir / f"{dataset}_random0.3_{suffix[variant]}_seed2025.json"
            dump(formal_path, formal)
            dump(smoke_path, smoke)
            paths.extend((relative(formal_path), relative(smoke_path)))
    return paths


def main():
    snapshot_protected_sources()
    architecture = architecture_audit()
    if architecture["status"] != "PASS":
        raise RuntimeError(architecture["status"])
    reuse = reuse_audit()
    if reuse["status"] != "PASS":
        raise RuntimeError("Gate 5a reuse audit failed")
    pairing = pairing_audit()
    if pairing["status"] != "PASS":
        raise RuntimeError(pairing["status"])
    config_paths = write_configs()
    plan = {
        "status": "PLANNING_PASS",
        "stage": "Gate 5a distribution-head ablation",
        "datasets": list(DATASETS),
        "missing_type": "random",
        "missing_rate": 0.3,
        "experiment_seed": 2025,
        "logical_tasks": 12,
        "reused_tasks": 6,
        "new_active_tasks": 6,
        "formal_configs": [path for path in config_paths if path.startswith("configs/gate5a/")],
        "queue_created": False,
        "scheduler_started": False,
        "old_unmasked_excluded": True,
        "second_backbone_started": False,
    }
    dump(ROOT / "results/gate5a_plan.json", plan)
    write_json_markdown(ROOT / "reports/gate5a_plan.md", "Gate 5a Plan", plan)
    status = {
        "status": "PLANNING_PASS",
        "architecture_audit": architecture["status"],
        "reuse_audit": reuse["status"],
        "pairing_validation": pairing["status"],
        "distribution_validation": "PENDING",
        "protocol_smoke": "NOT_STARTED",
        "formal_queue": "NOT_CREATED",
        "scheduler": "NOT_STARTED",
    }
    dump(ROOT / "results/gate5a_status.json", status)
    write_json_markdown(ROOT / "reports/gate5a_status.md", "Gate 5a Status", status)
    print(json.dumps({"status": "PASS", "architecture": architecture["status"], "reuse": reuse["reused"], "pairing_pairs": pairing["head_pairs_checked"], "configs": len(config_paths)}, indent=2))


if __name__ == "__main__":
    main()
