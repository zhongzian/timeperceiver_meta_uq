#!/usr/bin/env python
"""Summarize completed Gate 4c formal runs into reproducible CSV/Markdown tables."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runtime/gate4c_minimal_conformal_scheduler_state.json"


def load(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def formal_runs():
    state = load(STATE)
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"Gate 4c scheduler is not COMPLETE: {state.get('status')}")
    runs = {}
    for task in state["tasks"].values():
        if task.get("status") != "SUCCEEDED":
            raise RuntimeError(f"incomplete task in scheduler state: {task}")
        run = Path(task["run_dir"])
        if not run.is_absolute():
            run = ROOT / run
        status, metrics = load(run / "status.json"), load(run / "metrics.json")
        if status.get("status") != "SUCCESS" or metrics.get("smoke_only") is not False:
            raise RuntimeError(f"invalid formal run: {run}")
        runs[metrics["variant"]] = (run, metrics)
    if set(runs) != {"gaussian_head", "student_t_head"}:
        raise RuntimeError(f"formal variant set mismatch: {set(runs)}")
    return state, runs


def main():
    state, runs = formal_runs()
    with (ROOT / "reports/gate3b_full_results.csv").open(newline="") as handle:
        baseline = next(
            row for row in csv.DictReader(handle)
            if row["dataset"] == "etth1" and row["missing_type"] == "random"
            and float(row["target_missing_rate"]) == 0.3 and int(row["seed"]) == 2025
        )
    result_rows = [{
        "head": "official_baseline", "method": "point_reference", "experiment_id": baseline["experiment_id"],
        "test_mse": baseline["mse"], "test_mae": baseline["mae"], "test_rmse": baseline["rmse"],
        "mean_absolute_coverage_error": "NA", "mean_interval_width": "NA", "mean_interval_score": "NA",
        "best_epoch": "NA", "best_selection_nll": "NA", "trainable_parameter_count": 0,
        "peak_gpu_memory_mib": "NA", "run_dir": baseline["run_dir"],
    }]
    interval_rows = []
    bin_rows = []
    quantile_rows = []
    methods = (
        ("native", "native"),
        ("marginal_conformal", "marginal"),
        ("mask_severity_conditioned_conformal", "conditioned"),
    )
    for variant in ("gaussian_head", "student_t_head"):
        run, metrics = runs[variant]
        for metric_method, label in methods:
            summary = metrics["interval_metrics"][metric_method]
            result_rows.append({
                "head": variant, "method": label, "experiment_id": run.name,
                "test_mse": metrics["test_mse"], "test_mae": metrics["test_mae"],
                "test_rmse": metrics["test_rmse"],
                "mean_absolute_coverage_error": summary["mean_absolute_coverage_error"],
                "mean_interval_width": summary["mean_interval_width"],
                "mean_interval_score": summary["mean_interval_score"],
                "best_epoch": metrics["best_epoch"], "best_selection_nll": metrics["best_selection_nll"],
                "trainable_parameter_count": metrics["trainable_parameter_count"],
                "peak_gpu_memory_mib": metrics["peak_gpu_memory_mib"], "run_dir": str(run),
            })
            for tag, row in summary["by_level"].items():
                interval_rows.append({
                    "head": variant, "method": label, "coverage": float(tag) / 100.0,
                    **row, "experiment_id": run.name,
                })
            if metric_method in {"marginal_conformal", "mask_severity_conditioned_conformal"}:
                severity = metrics["severity_bin_metrics"][metric_method]
                boundaries = metrics["condition_bin_boundaries"]
                ranges = [f"(-inf,{boundaries[0]}]", f"({boundaries[0]},{boundaries[1]}]", f"({boundaries[1]},inf)"]
                for tag, level_data in severity.items():
                    for bin_index, row in level_data["bins"].items():
                        bin_rows.append({
                            "head": variant, "method": label, "coverage": float(tag) / 100.0,
                            "bin": int(bin_index), "missing_rate_range": ranges[int(bin_index)],
                            **row, "experiment_id": run.name,
                        })
        for tag, row in metrics["marginal_qhat"].items():
            quantile_rows.append({
                "head": variant, "method": "marginal", "coverage": float(tag) / 100.0,
                "bin": "all", "q_hat": row["q_hat"], "n": row["n"], "k": row["k"],
            })
        for tag, by_bin in metrics["conditioned_qhat"].items():
            for bin_index, row in by_bin.items():
                quantile_rows.append({
                    "head": variant, "method": "conditioned", "coverage": float(tag) / 100.0,
                    "bin": int(bin_index), "q_hat": row["q_hat"], "n": row["n"], "k": row["k"],
                })
    write_csv(ROOT / "reports/gate4c_minimal_results.csv", result_rows)
    write_csv(ROOT / "reports/gate4c_interval_metrics.csv", interval_rows)
    write_csv(ROOT / "reports/gate4c_conditioned_bin_metrics.csv", bin_rows)
    write_csv(ROOT / "reports/gate4c_calibration_quantiles.csv", quantile_rows)

    result_md = [
        "# Gate 4c Minimal Results", "",
        "The probability heads were retrained with selection-only early stopping. Old Gate 4b head checkpoints were not reused.", "",
        "| Head | Method | MSE | MAE | RMSE | Mean ACE | Mean width | Mean score | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    result_md += [
        f"| {row['head']} | {row['method']} | {row['test_mse']} | {row['test_mae']} | {row['test_rmse']} | "
        f"{row['mean_absolute_coverage_error']} | {row['mean_interval_width']} | {row['mean_interval_score']} | {row['best_epoch']} |"
        for row in result_rows
    ]
    (ROOT / "reports/gate4c_minimal_results.md").write_text("\n".join(result_md) + "\n")
    interval_md = [
        "# Gate 4c Interval Metrics", "",
        "| Head | Method | Coverage | PICP | MPIW | Abs coverage error | Interval score |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    interval_md += [
        f"| {row['head']} | {row['method']} | {row['coverage']:.2f} | {row['picp']:.6f} | "
        f"{row['mpiw']:.6f} | {row['absolute_coverage_error']:.6f} | {row['interval_score']:.6f} |"
        for row in interval_rows
    ]
    interval_md += ["", "Elementwise marginal empirical coverage under a fixed time-series split; no exact exchangeability guarantee is claimed."]
    (ROOT / "reports/gate4c_interval_metrics.md").write_text("\n".join(interval_md) + "\n")
    bin_md = [
        "# Gate 4c Severity-Bin Metrics", "",
        "| Head | Method | Coverage | Bin | Missing-rate range | PICP | MPIW | Abs error | Score |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    bin_md += [
        f"| {row['head']} | {row['method']} | {row['coverage']:.2f} | {row['bin']} | {row['missing_rate_range']} | "
        f"{row['picp']:.6f} | {row['mpiw']:.6f} | {row['absolute_coverage_error']:.6f} | {row['interval_score']:.6f} |"
        for row in bin_rows
    ]
    (ROOT / "reports/gate4c_conditioned_bin_metrics.md").write_text("\n".join(bin_md) + "\n")
    quantile_md = [
        "# Gate 4c Calibration Quantiles", "",
        "| Head | Method | Coverage | Bin | q-hat | n | k |", "|---|---|---:|---|---:|---:|---:|",
    ]
    quantile_md += [
        f"| {row['head']} | {row['method']} | {row['coverage']:.2f} | {row['bin']} | {row['q_hat']:.6f} | {row['n']} | {row['k']} |"
        for row in quantile_rows
    ]
    (ROOT / "reports/gate4c_calibration_quantiles.md").write_text("\n".join(quantile_md) + "\n")
    print(json.dumps({
        "status": "PASS", "scheduler": state["status"], "formal_runs": 2,
        "minimal_result_rows": len(result_rows), "interval_rows": len(interval_rows),
        "severity_bin_rows": len(bin_rows), "quantile_rows": len(quantile_rows),
    }, indent=2))


if __name__ == "__main__":
    main()

