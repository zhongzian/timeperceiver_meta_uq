#!/usr/bin/env python3
"""Render paper figures exclusively from persisted formal evidence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/timeperceiver-final-mpl")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from final_synthesis_lib import ROOT, as_float, read_csv_rows, read_json, write_csv


DATASET_LABELS = {"etth1": "ETTh1", "weather": "Weather", "electricity": "Electricity"}
TYPE_COLORS = {"random": "#2f6f9f", "block": "#d58b27", "channel": "#3f8f5f", "mixed": "#b34d5d"}
HEAD_COLORS = {"gaussian_head": "#2f6f9f", "student_t_head": "#d58b27"}
METHOD_COLORS = {"native": "#2f6f9f", "marginal": "#b34d5d", "conditioned": "#3f8f5f"}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def save(fig: Any, base: str) -> None:
    target = ROOT / base
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def registry_stage(stage: str) -> list[dict[str, Any]]:
    registry = read_json("results/final_evidence_registry.json")
    return [row for row in registry["rows"] if row["stage"] == stage and row["paper_evidence"] is True]


def figure1() -> int:
    rows = registry_stage("gate3b_fixed_mask")
    data = [
        {
            "dataset": row["dataset"],
            "missing_type": row["missing_type"],
            "missing_rate": float(row["missing_rate"]),
            "mse": float(row["mse"]),
            "evidence_id": row["evidence_id"],
            "test_mask_sha256": row["test_mask_sha256"],
        }
        for row in rows
    ]
    data.sort(key=lambda row: (row["dataset"], row["missing_type"], row["missing_rate"]))
    write_csv("paper/figures/fig1_missingness_degradation_data.csv", data, list(data[0]))
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), constrained_layout=True)
    for axis, dataset in zip(axes, ("etth1", "weather", "electricity")):
        for missing_type in ("random", "block", "channel", "mixed"):
            subset = [row for row in data if row["dataset"] == dataset and row["missing_type"] == missing_type]
            axis.plot(
                [row["missing_rate"] for row in subset],
                [row["mse"] for row in subset],
                marker="o",
                color=TYPE_COLORS[missing_type],
                label=missing_type.capitalize(),
            )
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Missing rate")
        axis.set_ylabel("MSE (lower is better)")
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Fixed historical-input missingness degradation", fontsize=11)
    save(fig, "paper/figures/fig1_missingness_degradation")
    return len(data)


def figure2() -> int:
    source = read_csv_rows("reports/gate4b_multiseed_paired_deltas.csv")
    data = [
        {
            "dataset": row["dataset"],
            "experiment_seed": int(row["experiment_seed"]),
            "delta_nll_student_t_minus_gaussian": float(row["delta_nll"]),
            "nll_win": row["nll_win"],
        }
        for row in source
    ]
    data.sort(key=lambda row: (("etth1", "weather", "electricity").index(row["dataset"]), row["experiment_seed"]))
    write_csv("paper/figures/fig2_multiseed_nll_deltas_data.csv", data, list(data[0]))
    fig, axis = plt.subplots(figsize=(8.8, 3.5), constrained_layout=True)
    x = np.arange(len(data))
    colors = [TYPE_COLORS[{"etth1": "random", "weather": "block", "electricity": "channel"}[row["dataset"]]] for row in data]
    axis.axhline(0, color="#555555", linewidth=1)
    axis.scatter(x, [row["delta_nll_student_t_minus_gaussian"] for row in data], c=colors, s=35, zorder=3)
    axis.set_xticks(x, [f"{DATASET_LABELS[row['dataset']]}\n{row['experiment_seed']}" for row in data])
    axis.set_ylabel("Student-t - Gaussian NLL\n(negative is better)")
    axis.set_title("Paired NLL deltas for all 12 formal cells")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "paper/figures/fig2_multiseed_nll_deltas")
    return len(data)


def figure3(metric: str, suffix: str) -> int:
    rows = registry_stage("gate4b_multiseed")
    data = []
    for dataset in ("etth1", "weather", "electricity"):
        for variant in ("gaussian_head", "student_t_head"):
            values = [float(row[metric]) for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            data.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_sample_std_ddof1": float(np.std(values, ddof=1)),
                    "seed_count": len(values),
                }
            )
    base = f"paper/figures/{suffix}"
    write_csv(base + "_data.csv", data, list(data[0]))
    fig, axis = plt.subplots(figsize=(6.5, 3.7), constrained_layout=True)
    datasets = ("etth1", "weather", "electricity")
    centers = np.arange(len(datasets))
    width = 0.32
    for offset, variant in zip((-width / 2, width / 2), ("gaussian_head", "student_t_head")):
        subset = [next(row for row in data if row["dataset"] == dataset and row["variant"] == variant) for dataset in datasets]
        axis.bar(
            centers + offset,
            [row[f"{metric}_mean"] for row in subset],
            width,
            yerr=[row[f"{metric}_sample_std_ddof1"] for row in subset],
            capsize=4,
            color=HEAD_COLORS[variant],
            label="Gaussian" if variant == "gaussian_head" else "Student-t",
            alpha=0.9,
        )
    axis.set_xticks(centers, [DATASET_LABELS[dataset] for dataset in datasets])
    axis.set_ylabel(f"{metric.upper()} (lower is better)")
    axis.set_title(f"Multi-seed {metric.upper()} mean and sample std")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save(fig, base)
    return len(data)


def figure4() -> int:
    source = read_csv_rows("reports/gate4b_multiseed_df_diagnostics.csv")
    data = [
        {
            "dataset": row["dataset"],
            "experiment_seed": int(row["experiment_seed"]),
            "mean_df": float(row["mean_df"]),
            "median_df": float(row["median_df"]),
            "min_df": float(row["min_df"]),
            "max_df": float(row["max_df"]),
        }
        for row in source
        if row["experiment_seed"] != "cross_seed_summary"
    ]
    data.sort(key=lambda row: (("etth1", "weather", "electricity").index(row["dataset"]), row["experiment_seed"]))
    write_csv("paper/figures/fig4_student_t_df_data.csv", data, list(data[0]))
    fig, axis = plt.subplots(figsize=(8.8, 3.7), constrained_layout=True)
    x = np.arange(len(data))
    mean = np.array([row["mean_df"] for row in data])
    lower = mean - np.array([row["min_df"] for row in data])
    upper = np.array([row["max_df"] for row in data]) - mean
    colors = [TYPE_COLORS[{"etth1": "random", "weather": "block", "electricity": "channel"}[row["dataset"]]] for row in data]
    axis.errorbar(x, mean, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#666666", capsize=3, alpha=0.7)
    axis.scatter(x, mean, c=colors, s=35, zorder=3)
    axis.set_xticks(x, [f"{DATASET_LABELS[row['dataset']]}\n{row['experiment_seed']}" for row in data])
    axis.set_ylabel("Mean learned df")
    axis.set_title("Student-t per-target-channel degrees-of-freedom summaries")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "paper/figures/fig4_student_t_df")
    return len(data)


def figure5() -> int:
    rows = [row for row in registry_stage("gate4c_interval_calibration") if row["variant"] == "student_t_head"]
    data = [
        {
            "method": row["interval_method"],
            "coverage": float(row["coverage_level"]),
            "mpiw": float(row["mpiw"]),
            "absolute_coverage_error": float(row["absolute_coverage_error"]),
            "interval_score": float(row["interval_score"]),
            "evidence_id": row["evidence_id"],
        }
        for row in rows
    ]
    data.sort(key=lambda row: (row["coverage"], row["method"]))
    write_csv("paper/figures/fig5_conformal_tradeoff_data.csv", data, list(data[0]))
    fig, axis = plt.subplots(figsize=(6.1, 4.1), constrained_layout=True)
    markers = {0.9: "o", 0.95: "s"}
    for row in data:
        axis.scatter(
            row["mpiw"],
            row["absolute_coverage_error"],
            color=METHOD_COLORS[row["method"]],
            marker=markers[row["coverage"]],
            s=55,
        )
    axis.set_xlabel("MPIW (lower is narrower)")
    axis.set_ylabel("Absolute coverage error (lower is better)")
    axis.set_title("Student-t conformal coverage-width tradeoff")
    axis.grid(alpha=0.2)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=METHOD_COLORS[method], markeredgecolor="none", label=label)
        for method, label in (("native", "Native"), ("marginal", "Marginal"), ("conditioned", "Conditioned"))
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="#555555", linestyle="none", label="90% target"),
            Line2D([0], [0], marker="s", color="#555555", linestyle="none", label="95% target"),
        ]
    )
    axis.legend(handles=handles, frameon=False, loc="lower right", ncol=2)
    save(fig, "paper/figures/fig5_conformal_tradeoff")
    return len(data)


def figure6() -> int:
    source = read_csv_rows("reports/gate4c_mechanism_native_shift.csv")
    data = [
        {
            "head": row["head"],
            "coverage": float(row["coverage"]),
            "calibration_picp": float(row["calibration_picp"]),
            "test_picp": float(row["test_picp"]),
            "coverage_shift_test_minus_calibration": float(row["coverage_shift_test_minus_calibration"]),
        }
        for row in source
    ]
    write_csv("paper/figures/fig6_calibration_test_shift_data.csv", data, list(data[0]))
    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x = np.arange(len(data))
    width = 0.34
    axis.bar(x - width / 2, [row["calibration_picp"] for row in data], width, color="#b34d5d", label="Calibration")
    axis.bar(x + width / 2, [row["test_picp"] for row in data], width, color="#3f8f5f", label="Test")
    for index, row in enumerate(data):
        axis.plot([index - 0.45, index + 0.45], [row["coverage"], row["coverage"]], color="#333333", linestyle="--", linewidth=1)
    axis.set_xticks(x, [f"{'Gaussian' if row['head'] == 'gaussian_head' else 'Student-t'}\n{int(row['coverage'] * 100)}%" for row in data])
    axis.set_ylim(0.82, 0.98)
    axis.set_ylabel("Native PICP")
    axis.set_title("Calibration segment is descriptively harder than test")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    save(fig, "paper/figures/fig6_calibration_test_shift")
    return len(data)


def appendix_figures() -> dict[str, int]:
    counts: dict[str, int] = {}
    adapter = registry_stage("gate4a_adapter_ablation")
    data_a1 = [
        {
            "variant": row["variant"],
            "test_mse": float(row["mse"]),
            "test_mae": float(row["mae"]),
            "evidence_id": row["evidence_id"],
        }
        for row in adapter
    ]
    write_csv("paper/appendix/figA1_gate4a_adapter_data.csv", data_a1, list(data_a1[0]))
    fig, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    labels = {"official_baseline": "Official", "static_adapter": "Static", "missingness_aware_adapter": "Missing-aware"}
    axis.bar([labels[row["variant"]] for row in data_a1], [row["test_mse"] for row in data_a1], color=["#777777", "#2f6f9f", "#d58b27"])
    axis.set_ylabel("Test MSE (lower is better)")
    axis.set_title("Gate 4a adapter ablation")
    axis.grid(axis="y", alpha=0.2)
    save(fig, "paper/appendix/figA1_gate4a_adapter")
    counts["figA1_gate4a_adapter"] = len(data_a1)

    channel_source = read_csv_rows("reports/gate4c_mechanism_channel_metrics.csv")
    data_a2 = [
        {
            "coverage": float(row["coverage"]),
            "channel": int(row["channel"]),
            "native_picp": float(row["native_picp"]),
            "marginal_picp": float(row["marginal_picp"]),
            "conditioned_picp": float(row["conditioned_picp"]),
        }
        for row in channel_source
        if row["head"] == "student_t_head"
    ]
    write_csv("paper/appendix/figA2_gate4c_channel_coverage_data.csv", data_a2, list(data_a2[0]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for axis, coverage in zip(axes, (0.9, 0.95)):
        subset = [row for row in data_a2 if row["coverage"] == coverage]
        for method in ("native", "marginal", "conditioned"):
            axis.plot([row["channel"] for row in subset], [row[f"{method}_picp"] for row in subset], marker="o", color=METHOD_COLORS[method], label=method.capitalize())
        axis.axhline(coverage, color="#444444", linestyle="--", linewidth=1)
        axis.set_title(f"{int(coverage * 100)}% target")
        axis.set_xlabel("Target channel")
        axis.set_ylabel("PICP")
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False)
    save(fig, "paper/appendix/figA2_gate4c_channel_coverage")
    counts["figA2_gate4c_channel_coverage"] = len(data_a2)

    horizon_source = read_csv_rows("reports/gate4c_mechanism_horizon_metrics.csv")
    data_a3 = [
        {
            "coverage": float(row["coverage"]),
            "horizon": int(row["horizon"]),
            "native_picp": float(row["native_picp"]),
            "marginal_picp": float(row["marginal_picp"]),
            "conditioned_picp": float(row["conditioned_picp"]),
        }
        for row in horizon_source
        if row["head"] == "student_t_head"
    ]
    write_csv("paper/appendix/figA3_gate4c_horizon_coverage_data.csv", data_a3, list(data_a3[0]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for axis, coverage in zip(axes, (0.9, 0.95)):
        subset = [row for row in data_a3 if row["coverage"] == coverage]
        for method in ("native", "marginal", "conditioned"):
            axis.plot([row["horizon"] for row in subset], [row[f"{method}_picp"] for row in subset], color=METHOD_COLORS[method], label=method.capitalize())
        axis.axhline(coverage, color="#444444", linestyle="--", linewidth=1)
        axis.set_title(f"{int(coverage * 100)}% target")
        axis.set_xlabel("Forecast horizon")
        axis.set_ylabel("PICP")
        axis.grid(alpha=0.2)
    axes[-1].legend(frameon=False)
    save(fig, "paper/appendix/figA3_gate4c_horizon_coverage")
    counts["figA3_gate4c_horizon_coverage"] = len(data_a3)
    return counts


def main() -> None:
    if read_json("results/final_metrics_consistency.json").get("FINAL_METRICS_CONSISTENCY_PASS") is not True:
        raise RuntimeError("Formal metrics consistency has not passed")
    configure()
    counts = {
        "fig1_missingness_degradation": figure1(),
        "fig2_multiseed_nll_deltas": figure2(),
        "fig3a_multiseed_nll": figure3("nll", "fig3a_multiseed_nll"),
        "fig3b_multiseed_crps": figure3("crps", "fig3b_multiseed_crps"),
        "fig3c_multiseed_mse": figure3("mse", "fig3c_multiseed_mse"),
        "fig4_student_t_df": figure4(),
        "fig5_conformal_tradeoff": figure5(),
        "fig6_calibration_test_shift": figure6(),
    }
    counts.update(appendix_figures())
    print({"status": "PASS", "figure_data_rows": counts})


if __name__ == "__main__":
    main()
