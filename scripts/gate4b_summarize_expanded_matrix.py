#!/usr/bin/env python
"""Summarize the completed 18-row Gate 4b expanded matrix."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_expanded_etth1.yaml"
STATE = ROOT / "runtime/gate4b_expanded_etth1_scheduler_state.json"
CONDITIONS = (("random", 0.1), ("random", 0.3), ("random", 0.5),
              ("block", 0.3), ("channel", 0.3), ("mixed", 0.3))


def load(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def task_key(mask_type, rate, variant):
    return f"{mask_type}_rate{rate}__{variant}"


def main():
    queue, state = load(QUEUE), load(STATE)
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"expanded scheduler is not COMPLETE: {state.get('status')}")
    official_inventory = load(ROOT / "results/gate4b_expanded_official_baseline_inventory.json")["rows"]
    rows, interval_rows, df_rows = [], [], []
    for mask_type, rate in CONDITIONS:
        condition = f"{mask_type}_rate{rate}"
        official = next(row for row in official_inventory if row["condition"] == condition)
        rows.append({
            "condition": condition, "missing_type": mask_type, "target_missing_rate": rate,
            "variant": "official", "experiment_id": official["experiment_id"], "status": "SUCCESS",
            "actual_test_missing_rate": official["actual_missing_rate"], "best_epoch": "NA",
            "best_validation_nll": "NA", "test_mean_nll": "NA", "test_crps": "NA",
            "test_mse": official["mse"], "test_mae": official["mae"], "test_rmse": official["rmse"],
            "mean_absolute_coverage_error": "NA", "mean_interval_width": "NA",
            "mean_interval_score": "NA", "mean_df": "NA", "median_df": "NA",
            "min_df": "NA", "max_df": "NA", "reused": True, "run_dir": official["run_dir"],
        })
        for variant in ("gaussian_head", "student_t_head"):
            key = task_key(mask_type, rate, variant)
            task_state = state["tasks"][key]
            if task_state.get("status") != "SUCCEEDED":
                raise RuntimeError(f"task not successful: {key}")
            run = Path(task_state["run_dir"])
            metrics = load(run / "metrics.json")
            row = {key2: metrics.get(key2) for key2 in (
                "actual_train_missing_rate", "actual_validation_missing_rate", "actual_test_missing_rate",
                "best_epoch", "best_validation_nll", "epochs_completed", "early_stopped",
                "test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
                "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
                "mean_scale", "median_scale", "p90_scale", "scale_min", "scale_max",
                "mean_df", "median_df", "min_df", "max_df", "trainable_parameter_count",
                "total_parameter_count", "parameter_overhead_ratio", "peak_gpu_memory_mib",
                "training_protocol_hash", "metric_protocol_hash", "source_checkpoint_sha256",
                "validation_mask_sha256", "test_mask_sha256")}
            row.update({"condition": condition, "missing_type": mask_type, "target_missing_rate": rate,
                        "variant": variant, "experiment_id": run.name, "status": "SUCCESS",
                        "reused": bool(task_state.get("reused")), "run_dir": str(run)})
            rows.append(row)
            for level in (50, 80, 90, 95):
                interval_rows.append({"condition": condition, "variant": variant, "level": level,
                    "picp": metrics[f"picp_{level}"], "mpiw": metrics[f"mpiw_{level}"],
                    "coverage_gap": metrics[f"coverage_gap_{level}"],
                    "absolute_coverage_error": metrics[f"absolute_coverage_error_{level}"],
                    "interval_score": metrics[f"interval_score_{level}"]})
            if variant == "student_t_head":
                df_rows.append({"condition": condition, "mean_df": metrics["mean_df"],
                                "median_df": metrics["median_df"], "min_df": metrics["min_df"],
                                "max_df": metrics["max_df"], "mean_predictive_std": metrics["mean_predictive_std"]})
    if len(rows) != 18 or len(interval_rows) != 48 or len(df_rows) != 6:
        raise RuntimeError("expanded row-count invariant failed")
    deltas = []
    for mask_type, rate in CONDITIONS:
        condition = f"{mask_type}_rate{rate}"
        gaussian = next(row for row in rows if row["condition"] == condition and row["variant"] == "gaussian_head")
        student = next(row for row in rows if row["condition"] == condition and row["variant"] == "student_t_head")
        delta = {"condition": condition}
        for key in ("test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
                    "mean_absolute_coverage_error", "mean_interval_score", "mean_interval_width"):
            delta[f"delta_{key}"] = float(student[key]) - float(gaussian[key])
        delta["relative_crps_improvement"] = (float(gaussian["test_crps"]) - float(student["test_crps"])) / float(gaussian["test_crps"])
        delta["relative_mse_improvement"] = (float(gaussian["test_mse"]) - float(student["test_mse"])) / float(gaussian["test_mse"])
        delta["student_t_nll_win"] = float(student["test_mean_nll"]) < float(gaussian["test_mean_nll"])
        delta["student_t_crps_noninferior"] = float(student["test_crps"]) <= float(gaussian["test_crps"])
        delta["student_t_point_noninferior"] = float(student["test_mse"]) <= float(gaussian["test_mse"]) * 1.01
        deltas.append(delta)
    write_csv(ROOT / "reports/gate4b_expanded_results.csv", rows)
    write_csv(ROOT / "reports/gate4b_expanded_interval_metrics.csv", interval_rows)
    write_csv(ROOT / "reports/gate4b_expanded_condition_deltas.csv", deltas)
    write_csv(ROOT / "reports/gate4b_expanded_df_diagnostics.csv", df_rows)
    result_lines = ["# Gate 4b Expanded Results", "", "Single seed (`2025`); no significance claim.", "",
                    "| Condition | Variant | Best epoch | Val NLL | Test NLL | CRPS | MSE | Coverage error | Interval score |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        result_lines.append("| " + " | ".join(str(row[key]) for key in (
            "condition", "variant", "best_epoch", "best_validation_nll", "test_mean_nll", "test_crps",
            "test_mse", "mean_absolute_coverage_error", "mean_interval_score")) + " |")
    (ROOT / "reports/gate4b_expanded_results.md").write_text("\n".join(result_lines) + "\n")
    for name, title, data in (("interval_metrics", "Interval Metrics", interval_rows),
                              ("condition_deltas", "Condition Deltas", deltas),
                              ("df_diagnostics", "Student-t df Diagnostics", df_rows)):
        lines = [f"# Gate 4b Expanded {title}", "", "```json", json.dumps(data, indent=2), "```"]
        (ROOT / f"reports/gate4b_expanded_{name}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "result_rows": len(rows), "interval_rows": len(interval_rows),
                      "delta_rows": len(deltas), "df_rows": len(df_rows)}, indent=2))


if __name__ == "__main__":
    main()
