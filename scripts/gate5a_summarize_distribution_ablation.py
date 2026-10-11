#!/usr/bin/env python3
"""Summarize a complete Gate 5a queue into formal result tables."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import DISPLAY_NAMES, HEADS, ROOT, dump, load, write_csv, write_json_markdown


QUEUE = ROOT / "configs/queues/gate5a_distribution_ablation.yaml"
STATE = ROOT / "runtime/gate5a_distribution_ablation_scheduler_state.json"
COMPARISONS = (
    ("laplace_head", "gaussian_head", "Laplace vs Gaussian"),
    ("student_t_fixed_df8_head", "gaussian_head", "Fixed-t vs Gaussian"),
    ("student_t_head", "student_t_fixed_df8_head", "Learnable-t vs Fixed-t"),
    ("student_t_head", "laplace_head", "Learnable-t vs Laplace"),
)


def task_key(task):
    return task.get("task_id") or f"{task['dataset']}__random_rate0.3__{task['variant']}"


def learned_df_from_checkpoint(run):
    checkpoint = torch.load(run / "checkpoints/best_prob_head.pt", map_location="cpu")
    raw_df = checkpoint["prob_head_state_dict"]["probabilistic_head.raw_df"]
    return (2.01 + torch.nn.functional.softplus(raw_df)).reshape(-1).tolist()


def main():
    queue = load(QUEUE)
    state = load(STATE)
    if state.get("status") != "COMPLETE":
        raise RuntimeError("Gate 5a queue is not COMPLETE")
    if len(state.get("tasks", {})) != 12 or any(
        item.get("status") != "SUCCEEDED" for item in state["tasks"].values()
    ):
        raise RuntimeError("Gate 5a logical tasks are incomplete")
    launch = load(ROOT / "results/gate5a_launch.json")
    scheduler_log = ROOT / launch["scheduler_log"]
    if "QUEUE_COMPLETE" not in scheduler_log.read_text():
        raise RuntimeError("QUEUE_COMPLETE marker missing")

    rows = []
    for task in queue["tasks"]:
        state_task = state["tasks"][task_key(task)]
        run = Path(state_task["run_dir"])
        if not run.is_absolute():
            run = ROOT / run
        status = load(run / "status.json")
        metrics = load(run / "metrics.json")
        variant = task["variant"]
        if status.get("status") != "SUCCESS" or metrics.get("variant") != variant:
            raise RuntimeError(f"invalid formal run: {run}")
        per_channel_df = metrics.get("per_channel_df")
        if variant == "student_t_head" and not per_channel_df:
            per_channel_df = learned_df_from_checkpoint(run)
        fixed_df = 8.0 if variant == "student_t_fixed_df8_head" else metrics.get("fixed_df")
        row = {
            "dataset": task["dataset"],
            "head": DISPLAY_NAMES[variant],
            "variant": variant,
            "status": "SUCCESS",
            "reused": bool(state_task.get("reused")),
            "experiment_id": run.name,
            "run_dir": str(run),
            "experiment_seed": 2025,
            "missing_type": "random",
            "missing_rate": 0.3,
            "test_mse": metrics["test_mse"],
            "test_mae": metrics["test_mae"],
            "test_rmse": metrics["test_rmse"],
            "test_mean_nll": metrics["test_mean_nll"],
            "test_median_element_nll": metrics["test_median_element_nll"],
            "test_p90_element_nll": metrics["test_p90_element_nll"],
            "test_p99_element_nll": metrics["test_p99_element_nll"],
            "test_crps": metrics["test_crps"],
            "mean_absolute_coverage_error": metrics["mean_absolute_coverage_error"],
            "mean_interval_width": metrics["mean_interval_width"],
            "mean_interval_score": metrics["mean_interval_score"],
            "trainable_parameter_count": metrics["trainable_parameter_count"],
            "total_parameter_count": metrics["total_parameter_count"],
            "parameter_overhead_ratio": metrics["parameter_overhead_ratio"],
            "peak_gpu_memory_mib": metrics["peak_gpu_memory_mib"],
            "elapsed_seconds": metrics["elapsed_seconds"],
            "epochs_completed": metrics["epochs_completed"],
            "best_epoch": metrics["best_epoch"],
            "best_validation_nll": metrics["best_validation_nll"],
            "mean_scale": metrics["mean_scale"],
            "median_scale": metrics["median_scale"],
            "p90_scale": metrics["p90_scale"],
            "scale_min": metrics["scale_min"],
            "scale_max": metrics["scale_max"],
            "fixed_df": fixed_df,
            "mean_df": metrics.get("mean_df"),
            "median_df": metrics.get("median_df"),
            "min_df": metrics.get("min_df"),
            "max_df": metrics.get("max_df"),
            "per_channel_df": json.dumps(per_channel_df) if per_channel_df is not None else "",
            "source_checkpoint_sha256": metrics["source_checkpoint_sha256"],
            "validation_mask_sha256": metrics["validation_mask_sha256"],
            "test_mask_sha256": metrics["test_mask_sha256"],
            "training_protocol_hash": metrics["training_protocol_hash"],
            "metric_protocol_hash": metrics["metric_protocol_hash"],
            "input_mask_applied": metrics["input_mask_applied"],
            "mask_application_verified": metrics.get("mask_application_verified", task.get("reuse_schema_audit_pass")),
            "legacy_unmasked_invalid": metrics.get("legacy_unmasked_invalid", False),
            "backbone_non_null_gradient_count": metrics["backbone_non_null_gradient_count"],
            "optimizer_backbone_parameter_count": metrics["optimizer_backbone_parameter_count"],
            "test_used_for_model_selection": metrics["test_used_for_model_selection"],
        }
        for level in (50, 80, 90, 95):
            for prefix in ("picp", "mpiw", "coverage_gap", "absolute_coverage_error", "interval_score"):
                row[f"{prefix}_{level}"] = metrics[f"{prefix}_{level}"]
        rows.append(row)
    rows.sort(key=lambda row: (row["dataset"], HEADS.index(row["variant"])))
    write_csv(ROOT / "reports/gate5a_distribution_results.csv", rows)
    result_payload = {"status": "PASS", "expected_rows": 12, "rows": rows}
    write_json_markdown(ROOT / "reports/gate5a_distribution_results.md", "Gate 5a Distribution Results", result_payload)

    by_key = {(row["dataset"], row["variant"]): row for row in rows}
    deltas = []
    for dataset in ("etth1", "weather", "electricity"):
        for candidate, reference, label in COMPARISONS:
            left = by_key[(dataset, candidate)]
            right = by_key[(dataset, reference)]
            row = {
                "dataset": dataset,
                "comparison": label,
                "candidate": candidate,
                "reference": reference,
                "delta_nll": left["test_mean_nll"] - right["test_mean_nll"],
                "delta_crps": left["test_crps"] - right["test_crps"],
                "delta_mse": left["test_mse"] - right["test_mse"],
                "delta_coverage_error": left["mean_absolute_coverage_error"] - right["mean_absolute_coverage_error"],
                "delta_interval_score": left["mean_interval_score"] - right["mean_interval_score"],
                "nll_win": left["test_mean_nll"] < right["test_mean_nll"],
                "crps_noninferior": left["test_crps"] <= right["test_crps"],
                "point_noninferior": left["test_mse"] <= right["test_mse"] * 1.01,
            }
            deltas.append(row)
    write_csv(ROOT / "reports/gate5a_pairwise_deltas.csv", deltas)
    write_json_markdown(
        ROOT / "reports/gate5a_pairwise_deltas.md", "Gate 5a Pairwise Deltas",
        {"status": "PASS", "rows": deltas}, "NLL deltas are absolute differences, never percentage changes.",
    )

    efficiency_rows = [{
        "dataset": row["dataset"], "head": row["head"], "variant": row["variant"],
        "trainable_parameter_count": row["trainable_parameter_count"],
        "total_parameter_count": row["total_parameter_count"],
        "parameter_overhead_ratio": row["parameter_overhead_ratio"],
        "training_wall_seconds": row["elapsed_seconds"],
        "epochs_completed": row["epochs_completed"],
        "mean_epoch_seconds": row["elapsed_seconds"] / row["epochs_completed"],
        "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
        "run_dir": row["run_dir"],
    } for row in rows]
    write_csv(ROOT / "reports/gate5a_parameter_efficiency.csv", efficiency_rows)
    write_json_markdown(
        ROOT / "reports/gate5a_parameter_efficiency.md", "Gate 5a Parameter and Training Efficiency",
        {"status": "PASS", "rows": efficiency_rows},
        "Formal wall times are hardware-tagged in each run and are not treated as controlled speed comparisons.",
    )

    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream/TimePerceiver"), "status", "--short"], text=True
    ).splitlines()
    tracked_diff = [line for line in upstream_status if not line.startswith("??")]
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in ("test_mse", "test_mean_nll", "test_crps", "scale_min", "scale_max")
    )
    implementation_checks = {
        "datasets_3": len({row["dataset"] for row in rows}) == 3,
        "logical_rows_12": len(rows) == 12,
        "active_success_6": sum(not row["reused"] for row in rows) == 6,
        "corrected_fixed_mask": all(row["input_mask_applied"] is True and row["mask_application_verified"] is True for row in rows),
        "old_unmasked_excluded": all(row["legacy_unmasked_invalid"] is False for row in rows),
        "hashes_paired": all(
            len({by_key[(dataset, variant)][field] for variant in HEADS}) == 1
            for dataset in ("etth1", "weather", "electricity")
            for field in ("source_checkpoint_sha256", "validation_mask_sha256", "test_mask_sha256", "training_protocol_hash", "metric_protocol_hash")
        ),
        "metrics_finite": finite,
        "scales_positive": all(float(row["scale_min"]) > 0 for row in rows),
        "fixed_df_exactly_8": all(row["variant"] != "student_t_fixed_df8_head" or row["fixed_df"] == 8.0 for row in rows),
        "backbone_frozen": all(row["backbone_non_null_gradient_count"] == 0 and row["optimizer_backbone_parameter_count"] == 0 for row in rows),
        "no_test_selection": all(row["test_used_for_model_selection"] is False for row in rows),
        "upstream_tracked_diff_empty": not tracked_diff,
        "queue_complete": state.get("status") == "COMPLETE",
    }
    status_payload = {
        "status": "FORMAL_RESULTS_COMPLETE_PENDING_DECISION" if all(implementation_checks.values()) else "FAILED",
        "GATE5A_IMPLEMENTATION_PASS": all(implementation_checks.values()),
        "logical_tasks": 12,
        "reused_tasks": sum(row["reused"] for row in rows),
        "active_tasks_succeeded": sum(not row["reused"] for row in rows),
        "scheduler_status": state["status"],
        "scheduler_pid": state.get("scheduler_pid"),
        "scheduler_log": str(scheduler_log.relative_to(ROOT)),
        "implementation_checks": implementation_checks,
        "upstream_tracked_status": tracked_diff,
    }
    dump(ROOT / "results/gate5a_status.json", status_payload)
    write_json_markdown(ROOT / "reports/gate5a_status.md", "Gate 5a Status", status_payload)
    print(json.dumps(status_payload, indent=2))
    return 0 if status_payload["GATE5A_IMPLEMENTATION_PASS"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
