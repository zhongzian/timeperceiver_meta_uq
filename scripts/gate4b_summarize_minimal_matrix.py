#!/usr/bin/env python
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "configs/queues/gate4b_minimal_probabilistic.yaml"


def load(path):
    return json.loads(Path(path).read_text())


def latest(variant, queue):
    task = next(item for item in queue["tasks"] if item["variant"] == variant)
    rows = []
    pattern = f"timeperceiver__etth1__gate4b__{variant}__random_rate0.3__seed2025__*/status.json"
    for path in (ROOT / "runs").glob(pattern):
        try:
            status, metrics = load(path), load(path.parent / "metrics.json")
            valid = (status.get("status") == "SUCCESS" and not metrics.get("smoke_only")
                     and metrics.get("training_protocol_hash") == task["training_protocol_hash"]
                     and metrics.get("metric_protocol_hash") == task["metric_protocol_hash"])
            if valid:
                rows.append((path.parent, metrics))
        except Exception:
            pass
    if not rows:
        raise RuntimeError(f"no matching formal SUCCESS run for {variant}")
    return sorted(rows, key=lambda item: item[0].name)[-1]


def main():
    state = load(ROOT / "runtime/gate4b_minimal_probabilistic_scheduler_state.json")
    if state.get("status") != "COMPLETE":
        raise RuntimeError(f"scheduler is not COMPLETE: {state.get('status')}")
    queue = load(QUEUE)
    with open(ROOT / "reports/gate3b_full_results.csv", newline="") as handle:
        baseline = next(row for row in csv.DictReader(handle)
                        if row["dataset"] == "etth1" and row["missing_type"] == "random"
                        and float(row["target_missing_rate"]) == 0.3 and int(row["seed"]) == 2025)
    rows = [{
        "variant": "official_baseline", "experiment_id": baseline["experiment_id"],
        "best_epoch": "NA", "best_validation_nll": "NA", "epochs_completed": "NA",
        "early_stopped": "NA", "test_mean_nll": "NA", "test_crps": "NA",
        "test_mse": float(baseline["mse"]), "test_mae": float(baseline["mae"]),
        "test_rmse": float(baseline["rmse"]), "mean_absolute_coverage_error": "NA",
        "mean_interval_width": "NA", "mean_interval_score": "NA",
        "trainable_parameter_count": 0, "total_parameter_count": "NA",
        "parameter_overhead_ratio": 0.0, "peak_gpu_memory_mib": "NA",
        "run_dir": baseline["run_dir"],
    }]
    metrics_by_variant = {}
    for variant in ("gaussian_head", "student_t_head"):
        run, metrics = latest(variant, queue)
        metrics_by_variant[variant] = metrics
        rows.append({key: metrics.get(key) for key in (
            "best_epoch", "best_validation_nll", "epochs_completed", "early_stopped",
            "test_mean_nll", "test_crps", "test_mse", "test_mae", "test_rmse",
            "mean_absolute_coverage_error", "mean_interval_width", "mean_interval_score",
            "trainable_parameter_count", "total_parameter_count", "parameter_overhead_ratio",
            "peak_gpu_memory_mib")} | {"variant": variant, "experiment_id": run.name, "run_dir": str(run)})
    fields = list(rows[0])
    with open(ROOT / "reports/gate4b_minimal_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    md = ["# Gate 4b Minimal Results", "", "Single seed (`2025`); no statistical-significance claim.", "",
          "| Variant | Best epoch | Val NLL | Test NLL | CRPS | MSE | MAE | RMSE | Coverage error | Mean width |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append("| " + " | ".join(str(row[key]) for key in (
            "variant", "best_epoch", "best_validation_nll", "test_mean_nll", "test_crps",
            "test_mse", "test_mae", "test_rmse", "mean_absolute_coverage_error", "mean_interval_width")) + " |")
    (ROOT / "reports/gate4b_minimal_results.md").write_text("\n".join(md) + "\n")

    interval_rows = []
    for variant, metrics in metrics_by_variant.items():
        for level in (50, 80, 90, 95):
            interval_rows.append({"variant": variant, "level": level,
                "picp": metrics[f"picp_{level}"], "mpiw": metrics[f"mpiw_{level}"],
                "coverage_gap": metrics[f"coverage_gap_{level}"],
                "interval_score": metrics[f"interval_score_{level}"]})
    with open(ROOT / "reports/gate4b_interval_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(interval_rows[0])); writer.writeheader(); writer.writerows(interval_rows)
    interval_md = ["# Gate 4b Interval Metrics", "", "| Variant | Level | PICP | MPIW | Coverage gap | Interval score |",
                   "|---|---:|---:|---:|---:|---:|"]
    interval_md += [f"| {r['variant']} | {r['level']} | {r['picp']:.6f} | {r['mpiw']:.6f} | {r['coverage_gap']:.6f} | {r['interval_score']:.6f} |" for r in interval_rows]
    (ROOT / "reports/gate4b_interval_metrics.md").write_text("\n".join(interval_md) + "\n")

    parameter_rows = [{key: row[key] for key in ("variant", "trainable_parameter_count", "total_parameter_count",
                                                   "parameter_overhead_ratio", "peak_gpu_memory_mib")} for row in rows]
    with open(ROOT / "reports/gate4b_parameter_comparison.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0])); writer.writeheader(); writer.writerows(parameter_rows)
    (ROOT / "reports/gate4b_parameter_comparison.md").write_text(
        "# Gate 4b Parameter Comparison\n\n" + "\n".join(
            f"- {row['variant']}: trainable={row['trainable_parameter_count']}, overhead={row['parameter_overhead_ratio']}, peak_mib={row['peak_gpu_memory_mib']}"
            for row in parameter_rows) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
