#!/usr/bin/env python
"""Summarize Gate 4b multi-seed results after the formal queue completes."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate4b_multiseed_common import (
    DATASETS,
    SEEDS,
    VARIANTS,
    dump,
    load,
    resolve,
    write_csv,
)

QUEUE_PATH = ROOT / "configs/queues/gate4b_multiseed_confirmation.yaml"
STATE_PATH = ROOT / "runtime/gate4b_multiseed_confirmation_scheduler_state.json"


def number(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


def mean(values):
    return statistics.fmean(values)


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.9f}"
            values.append(str(value if value is not None else "NA"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def official_rows():
    with (ROOT / "reports/gate3b_full_results.csv").open() as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for dataset in DATASETS:
        row = next(item for item in source if item["dataset"] == dataset
                   and item["missing_type"] == "random"
                   and math.isclose(float(item["target_missing_rate"]), 0.3)
                   and int(item["seed"]) == 2025)
        rows.append({
            "dataset": dataset,
            "variant": "official",
            "experiment_seed": "fixed_official_checkpoint",
            "status": row["status"],
            "experiment_id": row["experiment_id"],
            "reused": True,
            "run_dir": row["run_dir"],
            "actual_test_missing_rate": number(row["actual_missing_rate"]),
            "test_mse": number(row["mse"]),
            "test_mae": number(row["mae"]),
            "test_rmse": number(row["rmse"]),
            "test_mean_nll": None,
            "test_crps": None,
            "mean_absolute_coverage_error": None,
            "mean_interval_width": None,
            "mean_interval_score": None,
            "source_checkpoint_sha256": row["checkpoint_sha256"],
            "test_mask_sha256": row["mask_sha256"],
            "corrected_fixed_mask_evidence": True,
            "legacy_unmasked_invalid": False,
        })
    return rows


def main():
    queue = load(QUEUE_PATH)
    state = load(STATE_PATH)
    status_artifact = load(ROOT / "results/gate4b_multiseed_status.json")
    scheduler_log_value = status_artifact.get("scheduler_log")
    scheduler_log = resolve(scheduler_log_value) if scheduler_log_value else None
    state_tasks = state.get("tasks", {})
    succeeded = sum(item.get("status") == "SUCCEEDED" for item in state_tasks.values())
    reused = sum(item.get("reused") is True for item in state_tasks.values())
    new_succeeded = sum(item.get("status") == "SUCCEEDED" and item.get("reused") is not True
                        for item in state_tasks.values())
    queue_complete_marker = bool(
        scheduler_log and scheduler_log.is_file() and "QUEUE_COMPLETE" in scheduler_log.read_text()
    )
    scheduler_pid = int(state.get("scheduler_pid", -1))
    scheduler_exited = scheduler_pid <= 0 or not Path(f"/proc/{scheduler_pid}").exists()
    gates = {
        "scheduler_state_complete": state.get("status") == "COMPLETE",
        "probability_tasks_24_of_24": len(state_tasks) == 24 and succeeded == 24,
        "reused_tasks_6_of_6": reused == 6,
        "new_tasks_18_of_18": new_succeeded == 18,
        "queue_complete_marker": queue_complete_marker,
        "scheduler_exited_normally": scheduler_exited,
    }
    if not all(gates.values()):
        raise RuntimeError(f"MULTI_SEED_IMPLEMENTATION_INCOMPLETE: {gates}")

    task_by_id = {task["task_id"]: task for task in queue["tasks"]}
    probability_rows = []
    for task_id, task in task_by_id.items():
        state_row = state_tasks[task_id]
        run_dir = resolve(state_row["run_dir"])
        metrics = load(run_dir / "metrics.json")
        run_status = load(run_dir / "status.json")
        row = {
            "dataset": task["dataset"],
            "variant": task["variant"],
            "experiment_seed": task["experiment_seed"],
            "train_mask_base_seed": task["train_mask_base_seed"],
            "status": run_status.get("status"),
            "experiment_id": run_dir.name,
            "reused": state_row.get("reused") is True,
            "run_dir": str(run_dir),
            "actual_train_missing_rate": metrics.get("actual_train_missing_rate"),
            "actual_validation_missing_rate": metrics.get("actual_validation_missing_rate"),
            "actual_test_missing_rate": metrics.get("actual_test_missing_rate"),
            "best_epoch": metrics.get("best_epoch"),
            "best_validation_nll": metrics.get("best_validation_nll"),
            "epochs_completed": metrics.get("epochs_completed"),
            "test_mse": metrics.get("test_mse"),
            "test_mae": metrics.get("test_mae"),
            "test_rmse": metrics.get("test_rmse"),
            "test_mean_nll": metrics.get("test_mean_nll"),
            "test_median_element_nll": metrics.get("test_median_element_nll"),
            "test_p90_element_nll": metrics.get("test_p90_element_nll"),
            "test_p99_element_nll": metrics.get("test_p99_element_nll"),
            "test_crps": metrics.get("test_crps"),
            "mean_absolute_coverage_error": metrics.get("mean_absolute_coverage_error"),
            "mean_interval_width": metrics.get("mean_interval_width"),
            "mean_interval_score": metrics.get("mean_interval_score"),
            "mean_scale": metrics.get("mean_scale"),
            "median_scale": metrics.get("median_scale"),
            "p90_scale": metrics.get("p90_scale"),
            "scale_min": metrics.get("scale_min"),
            "scale_max": metrics.get("scale_max"),
            "mean_df": metrics.get("mean_df"),
            "median_df": metrics.get("median_df"),
            "min_df": metrics.get("min_df"),
            "max_df": metrics.get("max_df"),
            "mean_predictive_std": metrics.get("mean_predictive_std"),
            "peak_gpu_memory_mib": metrics.get("peak_gpu_memory_mib"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
            "source_checkpoint_sha256": metrics.get("source_checkpoint_sha256"),
            "validation_mask_sha256": metrics.get("validation_mask_sha256"),
            "test_mask_sha256": metrics.get("test_mask_sha256"),
            "training_protocol_hash": metrics.get("training_protocol_hash"),
            "metric_protocol_hash": metrics.get("metric_protocol_hash"),
            "input_mask_applied": metrics.get("input_mask_applied"),
            "mask_application_verified": metrics.get("mask_application_verified", task.get("reuse_schema_audit_pass")),
            "legacy_unmasked_invalid": metrics.get("legacy_unmasked_invalid", False),
            "test_used_for_model_selection": metrics.get("test_used_for_model_selection"),
            "test_mask_used_during_training": metrics.get("test_mask_used_during_training"),
            "batch_y_modified": metrics.get("batch_y_modified"),
            "backbone_non_null_gradient_count": metrics.get("backbone_non_null_gradient_count"),
            "optimizer_backbone_parameter_count": metrics.get("optimizer_backbone_parameter_count"),
            "checkpoint_strict_reload": metrics.get("checkpoint_strict_reload"),
            "corrected_fixed_mask_evidence": True,
        }
        for level in (50, 80, 90, 95):
            for prefix in ("picp", "mpiw", "coverage_gap", "absolute_coverage_error", "interval_score"):
                row[f"{prefix}_{level}"] = metrics.get(f"{prefix}_{level}")
        probability_rows.append(row)

    raw_rows = official_rows() + probability_rows
    raw_rows.sort(key=lambda row: (DATASETS.index(row["dataset"]),
                                   0 if row["variant"] == "official" else 1,
                                   str(row["experiment_seed"]), row["variant"]))
    write_csv(ROOT / "reports/gate4b_multiseed_results.csv", raw_rows)
    (ROOT / "reports/gate4b_multiseed_results.md").write_text(
        "# Gate 4b Multi-Seed Results\n\n"
        "The three official rows are fixed-checkpoint references; the 24 probability rows are the formal "
        "multi-seed evidence. Historical unmasked runs are excluded.\n\n" +
        markdown_table(raw_rows, [
            "dataset", "variant", "experiment_seed", "reused", "status", "test_mean_nll",
            "test_crps", "test_mse", "test_mae", "mean_absolute_coverage_error", "mean_df",
        ]) + "\n"
    )

    by_pair = defaultdict(dict)
    for row in probability_rows:
        by_pair[(row["dataset"], int(row["experiment_seed"]))][row["variant"]] = row
    paired_rows = []
    delta_fields = {
        "nll": "test_mean_nll", "crps": "test_crps", "mse": "test_mse",
        "mae": "test_mae", "rmse": "test_rmse",
        "coverage_error": "mean_absolute_coverage_error",
        "interval_score": "mean_interval_score", "interval_width": "mean_interval_width",
    }
    for dataset in DATASETS:
        for seed in SEEDS:
            gaussian = by_pair[(dataset, seed)]["gaussian_head"]
            student = by_pair[(dataset, seed)]["student_t_head"]
            row = {"dataset": dataset, "experiment_seed": seed}
            for short, field in delta_fields.items():
                row[f"gaussian_{short}"] = gaussian[field]
                row[f"student_t_{short}"] = student[field]
                row[f"delta_{short}"] = float(student[field]) - float(gaussian[field])
            row["nll_win"] = row["delta_nll"] < 0
            row["crps_noninferior"] = row["delta_crps"] <= 0
            row["point_noninferior"] = student["test_mse"] <= gaussian["test_mse"] * 1.01
            row["full_pair_pass"] = (
                row["nll_win"] and row["crps_noninferior"] and row["point_noninferior"]
            )
            paired_rows.append(row)
    write_csv(ROOT / "reports/gate4b_multiseed_paired_deltas.csv", paired_rows)
    (ROOT / "reports/gate4b_multiseed_paired_deltas.md").write_text(
        "# Gate 4b Multi-Seed Paired Deltas\n\nDelta is Student-t minus Gaussian.\n\n" +
        markdown_table(paired_rows, [
            "dataset", "experiment_seed", "delta_nll", "delta_crps", "delta_mse",
            "nll_win", "crps_noninferior", "point_noninferior", "full_pair_pass",
        ]) + "\n"
    )

    metric_fields = {
        "nll": "test_mean_nll", "crps": "test_crps", "mse": "test_mse",
        "mae": "test_mae", "rmse": "test_rmse",
        "coverage_error": "mean_absolute_coverage_error",
        "interval_score": "mean_interval_score", "interval_width": "mean_interval_width",
    }
    dataset_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = sorted((row for row in probability_rows
                            if row["dataset"] == dataset and row["variant"] == variant),
                           key=lambda row: int(row["experiment_seed"]))
            summary = {"summary_type": "variant", "dataset": dataset, "variant": variant,
                       "seed_count": len(group), "seeds": json.dumps(list(SEEDS))}
            for short, field in metric_fields.items():
                values = [float(row[field]) for row in group]
                summary[f"{short}_mean"] = mean(values)
                summary[f"{short}_std_ddof1"] = sample_std(values)
                summary[f"{short}_median"] = statistics.median(values)
                summary[f"{short}_min"] = min(values)
                summary[f"{short}_max"] = max(values)
                summary[f"{short}_values"] = json.dumps(values)
            dataset_rows.append(summary)
        paired = [row for row in paired_rows if row["dataset"] == dataset]
        summary = {
            "summary_type": "paired_delta", "dataset": dataset,
            "variant": "student_t_minus_gaussian", "seed_count": len(paired),
            "nll_win_count": sum(row["nll_win"] for row in paired),
            "crps_noninferior_count": sum(row["crps_noninferior"] for row in paired),
            "point_noninferior_count": sum(row["point_noninferior"] for row in paired),
            "full_pair_pass_count": sum(row["full_pair_pass"] for row in paired),
        }
        for short in metric_fields:
            values = [float(row[f"delta_{short}"]) for row in paired]
            summary[f"delta_{short}_mean"] = mean(values)
            summary[f"delta_{short}_std_ddof1"] = sample_std(values)
            summary[f"delta_{short}_median"] = statistics.median(values)
        dataset_rows.append(summary)
    write_csv(ROOT / "reports/gate4b_multiseed_dataset_summary.csv", dataset_rows)
    (ROOT / "reports/gate4b_multiseed_dataset_summary.md").write_text(
        "# Gate 4b Multi-Seed Dataset Summary\n\nAll standard deviations use ddof=1. "
        "No statistical-significance claim is made.\n\n" +
        markdown_table(dataset_rows, [
            "summary_type", "dataset", "variant", "seed_count", "nll_mean", "nll_std_ddof1",
            "crps_mean", "mse_mean", "nll_win_count", "full_pair_pass_count",
        ]) + "\n"
    )

    interval_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            group = [row for row in probability_rows if row["dataset"] == dataset and row["variant"] == variant]
            for level in (50, 80, 90, 95):
                row = {"dataset": dataset, "variant": variant, "interval_level": level / 100,
                       "seed_count": len(group), "interval_type": "native_parametric",
                       "conformal_calibrated": False}
                for short, field in (
                    ("picp", f"picp_{level}"), ("mpiw", f"mpiw_{level}"),
                    ("absolute_coverage_error", f"absolute_coverage_error_{level}"),
                    ("interval_score", f"interval_score_{level}"),
                ):
                    values = [float(item[field]) for item in group]
                    row[f"{short}_mean"] = mean(values)
                    row[f"{short}_std_ddof1"] = sample_std(values)
                interval_rows.append(row)
    write_csv(ROOT / "reports/gate4b_multiseed_interval_summary.csv", interval_rows)
    (ROOT / "reports/gate4b_multiseed_interval_summary.md").write_text(
        "# Gate 4b Multi-Seed Native Interval Summary\n\nThese are native parametric intervals, not calibrated intervals.\n\n" +
        markdown_table(interval_rows, [
            "dataset", "variant", "interval_level", "picp_mean", "mpiw_mean",
            "absolute_coverage_error_mean", "interval_score_mean",
        ]) + "\n"
    )

    df_rows = []
    for dataset in DATASETS:
        group = sorted((row for row in probability_rows
                        if row["dataset"] == dataset and row["variant"] == "student_t_head"),
                       key=lambda row: int(row["experiment_seed"]))
        for item in group:
            df_rows.append({
                "dataset": dataset, "experiment_seed": item["experiment_seed"],
                "mean_df": item["mean_df"], "median_df": item["median_df"],
                "min_df": item["min_df"], "max_df": item["max_df"],
                "mean_predictive_std": item["mean_predictive_std"],
            })
        mean_dfs = [float(item["mean_df"]) for item in group]
        df_rows.append({
            "dataset": dataset, "experiment_seed": "cross_seed_summary",
            "mean_df": mean(mean_dfs), "mean_df_std_ddof1": sample_std(mean_dfs),
            "median_df": statistics.median(mean_dfs), "min_df": min(mean_dfs), "max_df": max(mean_dfs),
        })
    write_csv(ROOT / "reports/gate4b_multiseed_df_diagnostics.csv", df_rows)
    (ROOT / "reports/gate4b_multiseed_df_diagnostics.md").write_text(
        "# Gate 4b Multi-Seed Student-t DF Diagnostics\n\n" +
        markdown_table(df_rows, [
            "dataset", "experiment_seed", "mean_df", "mean_df_std_ddof1", "median_df",
            "min_df", "max_df", "mean_predictive_std",
        ]) + "\n"
    )

    status = {
        "stage": "Gate 4b multi-seed confirmation",
        "status": "SUMMARY_COMPLETE_DECISION_PENDING",
        "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
        "FULL_END_TO_END_MULTI_SEED": False,
        "logical_probability_tasks": 24,
        "probability_tasks_succeeded_or_reused": succeeded,
        "reused_tasks": reused,
        "new_tasks_succeeded": new_succeeded,
        "raw_result_rows": len(raw_rows),
        "paired_delta_rows": len(paired_rows),
        "scheduler_state": state.get("status"),
        "scheduler_pid": scheduler_pid,
        "scheduler_log": scheduler_log_value,
        "completion_gates": gates,
        "legacy_unmasked_results_excluded": True,
        "statistical_significance_claim": False,
        "conformal_calibrated": False,
        "formal_decision": "PENDING",
        "ENTER_GATE4C": False,
    }
    dump(ROOT / "results/gate4b_multiseed_status.json", status)
    (ROOT / "reports/gate4b_multiseed_status.md").write_text(
        "# Gate 4b Multi-Seed Status\n\nStatus: **SUMMARY_COMPLETE_DECISION_PENDING**\n\n"
        f"Probability tasks: {succeeded}/24; reused: {reused}/6; new: {new_succeeded}/18.\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
