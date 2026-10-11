#!/usr/bin/env python
"""Summarize the completed six-row Gate 4b cross-dataset matrix."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_cross_dataset.yaml"
STATE = ROOT / "runtime/gate4b_cross_dataset_scheduler_state.json"
DATASETS = ("weather", "electricity")


def load(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    queue, state = load(QUEUE), load(STATE)
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"cross-dataset scheduler is not COMPLETE: {state.get('status')}")
    official = {row["dataset"]: row for row in load(
        ROOT / "results/gate4b_cross_dataset_official_inventory.json")["rows"]}
    rows, interval_rows, df_rows = [], [], []
    for dataset in DATASETS:
        baseline = official[dataset]
        rows.append({
            "dataset": dataset, "variant": "official", "status": "SUCCESS",
            "experiment_id": baseline["experiment_id"], "actual_test_missing_rate": baseline["actual_missing_rate"],
            "best_epoch": "NA", "best_validation_nll": "NA", "test_mean_nll": "NA", "test_crps": "NA",
            "test_mse": baseline["mse"], "test_mae": baseline["mae"], "test_rmse": baseline["rmse"],
            "mean_absolute_coverage_error": "NA", "mean_interval_width": "NA", "mean_interval_score": "NA",
            "mean_df": "NA", "median_df": "NA", "min_df": "NA", "max_df": "NA",
            "reused": True, "run_dir": baseline["run_dir"],
        })
        for variant in ("gaussian_head", "student_t_head"):
            key = f"{dataset}__random_rate0.3__{variant}"
            task_state = state["tasks"][key]
            if task_state.get("status") != "SUCCEEDED":
                raise RuntimeError(f"task not successful: {key}")
            run = Path(task_state["run_dir"])
            metrics = load(run / "metrics.json")
            selected = {key2: metrics.get(key2) for key2 in (
                "actual_train_missing_rate", "actual_validation_missing_rate", "actual_test_missing_rate",
                "best_epoch", "best_validation_nll", "epochs_completed", "early_stopped",
                "test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
                "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
                "mean_scale", "median_scale", "p90_scale", "scale_min", "scale_max",
                "mean_df", "median_df", "min_df", "max_df", "mean_predictive_std",
                "target_channels", "trainable_parameter_count", "total_parameter_count",
                "parameter_overhead_ratio", "peak_gpu_memory_mib", "training_protocol_hash",
                "metric_protocol_hash", "source_checkpoint_sha256", "validation_mask_sha256", "test_mask_sha256",
                "input_mask_applied", "mask_application_verified", "legacy_unmasked_invalid",
                "freeze_backbone", "optimizer_backbone_parameter_count", "backbone_non_null_gradient_count",
                "test_used_for_model_selection", "test_mask_used_during_training", "batch_y_modified")}
            selected.update({"dataset": dataset, "variant": variant, "status": "SUCCESS",
                             "experiment_id": run.name, "reused": bool(task_state.get("reused")),
                             "run_dir": str(run)})
            rows.append(selected)
            for level in (50, 80, 90, 95):
                interval_rows.append({
                    "dataset": dataset, "variant": variant, "level": level,
                    "picp": metrics[f"picp_{level}"], "mpiw": metrics[f"mpiw_{level}"],
                    "coverage_gap": metrics[f"coverage_gap_{level}"],
                    "absolute_coverage_error": metrics[f"absolute_coverage_error_{level}"],
                    "interval_score": metrics[f"interval_score_{level}"],
                })
            if variant == "student_t_head":
                df_rows.append({"dataset": dataset, "target_channels": metrics["target_channels"],
                                "mean_df": metrics["mean_df"], "median_df": metrics["median_df"],
                                "min_df": metrics["min_df"], "max_df": metrics["max_df"],
                                "mean_predictive_std": metrics["mean_predictive_std"]})
    if len(rows) != 6 or len(interval_rows) != 16 or len(df_rows) != 2:
        raise RuntimeError("cross-dataset row-count invariant failed")
    deltas = []
    for dataset in DATASETS:
        gaussian = next(row for row in rows if row["dataset"] == dataset and row["variant"] == "gaussian_head")
        student = next(row for row in rows if row["dataset"] == dataset and row["variant"] == "student_t_head")
        delta = {"dataset": dataset}
        for key in ("test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
                    "mean_absolute_coverage_error", "mean_interval_score", "mean_interval_width"):
            delta[f"delta_{key}"] = float(student[key]) - float(gaussian[key])
        delta["relative_crps_improvement"] = ((float(gaussian["test_crps"]) - float(student["test_crps"]))
                                               / float(gaussian["test_crps"]))
        delta["relative_mse_improvement"] = ((float(gaussian["test_mse"]) - float(student["test_mse"]))
                                              / float(gaussian["test_mse"]))
        delta["student_t_nll_win"] = float(student["test_mean_nll"]) < float(gaussian["test_mean_nll"])
        delta["student_t_crps_noninferior"] = float(student["test_crps"]) <= float(gaussian["test_crps"])
        delta["student_t_point_noninferior"] = float(student["test_mse"]) <= float(gaussian["test_mse"]) * 1.01
        deltas.append(delta)
    write_csv(ROOT / "reports/gate4b_cross_dataset_results.csv", rows)
    write_csv(ROOT / "reports/gate4b_cross_dataset_interval_metrics.csv", interval_rows)
    write_csv(ROOT / "reports/gate4b_cross_dataset_deltas.csv", deltas)
    write_csv(ROOT / "reports/gate4b_cross_dataset_df_diagnostics.csv", df_rows)
    result_lines = ["# Gate 4b Cross-Dataset Results", "", "Single seed (`2025`); no significance claim.", "",
                    "| Dataset | Variant | Best epoch | Val NLL | Test NLL | CRPS | MSE | Coverage error | Interval score |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        result_lines.append("| " + " | ".join(str(row[key]) for key in (
            "dataset", "variant", "best_epoch", "best_validation_nll", "test_mean_nll", "test_crps",
            "test_mse", "mean_absolute_coverage_error", "mean_interval_score")) + " |")
    (ROOT / "reports/gate4b_cross_dataset_results.md").write_text("\n".join(result_lines) + "\n")
    for name, title, data in (("interval_metrics", "Interval Metrics", interval_rows),
                              ("deltas", "Dataset Deltas", deltas),
                              ("df_diagnostics", "Student-t df Diagnostics", df_rows)):
        (ROOT / f"reports/gate4b_cross_dataset_{name}.md").write_text(
            f"# Gate 4b Cross-Dataset {title}\n\n```json\n{json.dumps(data, indent=2)}\n```\n")
    print(json.dumps({"status": "PASS", "result_rows": 6, "interval_rows": 16,
                      "delta_rows": 2, "df_rows": 2}, indent=2))


if __name__ == "__main__":
    main()
