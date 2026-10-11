#!/usr/bin/env python3
"""Eval-only Gate 5a PIT, QQ, tail, residual-bin, and channel diagnostics."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaln, ndtr, stdtr
from scipy.stats import kurtosis, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import DATASETS, DISPLAY_NAMES, HEADS, dump, load, write_csv, write_json_markdown
from gate5a_eval_common import forward_batch, load_model_and_data, prepare_batch, queue_run_map


CACHE = ROOT / "artifacts/gate5a_heavy_tail_diagnostics"
FIGURES = ROOT / "paper/figures/gate5a"
REPORT_DIR = ROOT / "reports/gate5a_heavy_tail_diagnostics"
CHUNK = 1_000_000


def df_from_run(run, variant, channels):
    if variant == "student_t_fixed_df8_head":
        return np.full(channels, 8.0, dtype=np.float32)
    if variant != "student_t_head":
        return None
    checkpoint = torch.load(run / "checkpoints/best_prob_head.pt", map_location="cpu")
    raw = checkpoint["prob_head_state_dict"]["probabilistic_head.raw_df"]
    return (2.01 + torch.nn.functional.softplus(raw)).reshape(-1).numpy().astype(np.float32)


def build_cache(dataset, variant, task, run, device):
    directory = CACHE / dataset / variant
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = load(manifest_path)
        if manifest.get("experiment_id") == run.name and manifest.get("status") == "PASS":
            return manifest
    directory.mkdir(parents=True, exist_ok=True)
    model, args, loader, test_mask, indices = load_model_and_data(task, run, device)
    shape = (len(test_mask), args.pred_len, args.enc_in)
    loc_mm = np.lib.format.open_memmap(directory / "loc.npy", mode="w+", dtype=np.float32, shape=shape)
    scale_mm = np.lib.format.open_memmap(directory / "scale.npy", mode="w+", dtype=np.float32, shape=shape)
    target_path = CACHE / dataset / "target.npy"
    target_mm = np.lib.format.open_memmap(target_path, mode="w+", dtype=np.float32, shape=shape)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            size = len(batch[0])
            prepared = prepare_batch(batch, test_mask[offset : offset + size], args, indices, device)
            output, target = forward_batch(model, prepared, args)
            loc_mm[offset : offset + size] = output["loc"].detach().cpu().numpy()
            scale_mm[offset : offset + size] = output["scale"].detach().cpu().numpy()
            target_mm[offset : offset + size] = target.detach().cpu().numpy()
            offset += size
    loc_mm.flush(); scale_mm.flush(); target_mm.flush()
    df = df_from_run(run, variant, args.enc_in)
    if df is not None:
        np.save(directory / "df.npy", df)
    manifest = {
        "status": "PASS", "dataset": dataset, "variant": variant, "experiment_id": run.name,
        "shape": list(shape), "loc": str((directory / "loc.npy").relative_to(ROOT)),
        "scale": str((directory / "scale.npy").relative_to(ROOT)),
        "target": str(target_path.relative_to(ROOT)),
        "df": str((directory / "df.npy").relative_to(ROOT)) if df is not None else None,
        "eval_only": True, "training": False, "backward": False,
    }
    dump(manifest_path, manifest)
    del model, loc_mm, scale_mm, target_mm
    torch.cuda.empty_cache()
    return manifest


def nll(variant, loc, scale, target, df=None):
    residual = (target - loc) / scale
    if variant == "gaussian_head":
        return 0.5 * math.log(2 * math.pi) + np.log(scale) + 0.5 * residual ** 2
    if variant == "laplace_head":
        return np.log(2 * scale) + np.abs(residual)
    df = np.asarray(df)
    return (
        gammaln((df + 1) / 2) * -1 + gammaln(df / 2)
        + 0.5 * np.log(df * math.pi) + np.log(scale)
        + ((df + 1) / 2) * np.log1p((residual ** 2) / df)
    )


def pit(variant, loc, scale, target, df=None):
    value = (target - loc) / scale
    if variant == "gaussian_head":
        return ndtr(value)
    if variant == "laplace_head":
        return np.where(value < 0, 0.5 * np.exp(value), 1 - 0.5 * np.exp(-value))
    return stdtr(df, value)


def load_arrays(dataset, variant):
    directory = CACHE / dataset / variant
    loc = np.load(directory / "loc.npy", mmap_mode="r")
    scale = np.load(directory / "scale.npy", mmap_mode="r")
    target = np.load(CACHE / dataset / "target.npy", mmap_mode="r")
    df_path = directory / "df.npy"
    df = np.load(df_path) if df_path.is_file() else None
    return loc, scale, target, df


def markdown_table(path, title, rows, fields):
    text = f"# {title}\n\n"
    text += "| " + " | ".join(fields) + " |\n"
    text += "| " + " | ".join("---" for _ in fields) + " |\n"
    for row in rows:
        text += "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n"
    Path(path).write_text(text)


def main():
    decision = load(ROOT / "results/gate5a_formal_decision.json")
    if decision.get("GATE5A_IMPLEMENTATION_PASS") is not True:
        raise RuntimeError("Gate 5a implementation is incomplete")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("WAITING_RESOURCE: CUDA required for eval-only cache generation")
    CACHE.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True); REPORT_DIR.mkdir(parents=True, exist_ok=True)
    mapping = queue_run_map()
    for dataset in DATASETS:
        for variant in HEADS:
            task, run = mapping[(dataset, variant)]
            build_cache(dataset, variant, task, run, device)

    pit_rows, qq_rows, tail_rows = [], [], []
    pit_plot_data, qq_plot_data = [], []
    rng = np.random.default_rng(52025)
    for dataset in DATASETS:
        for variant in HEADS:
            loc, scale, target, channel_df = load_arrays(dataset, variant)
            flat_size = loc.size
            hist = np.zeros(20, dtype=np.int64)
            total_sum = total_sq = 0.0
            low05 = high95 = 0
            surprise = {0.05: 0, 0.01: 0, 0.001: 0}
            sample_size = min(200_000, flat_size)
            sample_indices = np.sort(rng.choice(flat_size, size=sample_size, replace=False))
            sample_values = []
            lf, sf, tf = loc.reshape(-1), scale.reshape(-1), target.reshape(-1)
            channels = loc.shape[-1]
            for start in range(0, flat_size, CHUNK):
                stop = min(start + CHUNK, flat_size)
                if channel_df is None:
                    df_chunk = None
                else:
                    df_chunk = channel_df[np.arange(start, stop) % channels]
                values = np.clip(pit(variant, lf[start:stop], sf[start:stop], tf[start:stop], df_chunk), 0, 1)
                hist += np.histogram(values, bins=np.linspace(0, 1, 21))[0]
                total_sum += float(values.sum()); total_sq += float((values ** 2).sum())
                low05 += int((values < 0.05).sum()); high95 += int((values > 0.95).sum())
                two_sided = 2 * np.minimum(values, 1 - values)
                for threshold in surprise:
                    surprise[threshold] += int((two_sided < threshold).sum())
                selected = sample_indices[(sample_indices >= start) & (sample_indices < stop)] - start
                if len(selected):
                    sample_values.append(values[selected])
            count = flat_size
            mean = total_sum / count
            variance = total_sq / count - mean ** 2
            expected = count / 20
            calibration_error = float(np.abs(hist - expected).sum() / count)
            row = {
                "dataset": dataset, "head": DISPLAY_NAMES[variant], "variant": variant, "count": count,
                "mean": mean, "variance": variance, "pit_calibration_error": calibration_error,
                "fraction_pit_lt_0.05": low05 / count, "fraction_pit_gt_0.95": high95 / count,
                "fraction_pit_outside_0.05_0.95": (low05 + high95) / count,
                "histogram_20_bins": json.dumps(hist.tolist()),
            }
            pit_rows.append(row)
            for bin_index, value in enumerate(hist):
                pit_plot_data.append({"dataset": dataset, "variant": variant, "bin": bin_index, "count": int(value), "fraction": value / count})
            tail_rows.append({
                "dataset": dataset, "head": DISPLAY_NAMES[variant], "variant": variant,
                "pit_tail_prob_lt_0.05": surprise[0.05] / count,
                "pit_tail_prob_lt_0.01": surprise[0.01] / count,
                "pit_tail_prob_lt_0.001": surprise[0.001] / count,
            })
            sample = np.concatenate(sample_values)
            sample = np.clip(sample, 1e-6, 1 - 1e-6)
            from scipy.special import ndtri
            z = np.sort(ndtri(sample))
            probabilities = np.linspace(0.005, 0.995, 199)
            observed = np.quantile(z, probabilities)
            theoretical = ndtri(probabilities)
            rmse = float(np.sqrt(np.mean((observed - theoretical) ** 2)))
            qq_rows.append({"dataset": dataset, "head": DISPLAY_NAMES[variant], "variant": variant, "sample_count": len(sample), "qq_rmse": rmse})
            qq_plot_data.extend({"dataset": dataset, "variant": variant, "probability": float(p), "theoretical": float(t), "observed": float(o)} for p, t, o in zip(probabilities, theoretical, observed))

        gloc, gscale, target, _ = load_arrays(dataset, "gaussian_head")
        z = np.abs((target - gloc) / gscale)
        gaussian_tail = next(row for row in tail_rows if row["dataset"] == dataset and row["variant"] == "gaussian_head")
        gaussian_tail.update({"gaussian_z_gt_2": float((z > 2).mean()), "gaussian_z_gt_3": float((z > 3).mean()), "gaussian_z_gt_4": float((z > 4).mean())})

    write_csv(ROOT / "reports/gate5a_pit_metrics.csv", pit_rows)
    markdown_table(ROOT / "reports/gate5a_pit_metrics.md", "Gate 5a PIT Metrics", pit_rows, ["dataset", "head", "count", "mean", "variance", "pit_calibration_error", "fraction_pit_outside_0.05_0.95"])
    write_csv(ROOT / "reports/gate5a_qq_metrics.csv", qq_rows)
    write_csv(ROOT / "reports/gate5a_tail_exceedance.csv", tail_rows)
    markdown_table(ROOT / "reports/gate5a_tail_exceedance.md", "Gate 5a Tail Exceedance", tail_rows, list(tail_rows[0]))

    residual_bin_rows, channel_rows, correlation_rows = [], [], []
    bin_labels = ("0-50", "50-90", "90-95", "95-99", "99-100")
    for dataset in DATASETS:
        arrays = {variant: load_arrays(dataset, variant) for variant in HEADS}
        gloc, gscale, target, _ = arrays["gaussian_head"]
        residual = np.abs((target - gloc) / gscale)
        boundaries = np.quantile(residual, [0, .5, .9, .95, .99, 1])
        nll_values = {}
        for variant, (loc, scale, _, channel_df) in arrays.items():
            df_full = channel_df.reshape((1, 1, -1)) if channel_df is not None else None
            nll_values[variant] = nll(variant, loc, scale, target, df_full)
        for index, label in enumerate(bin_labels):
            selected = (residual >= boundaries[index]) & ((residual <= boundaries[index + 1]) if index == 4 else (residual < boundaries[index + 1]))
            means = {variant: float(values[selected].mean()) for variant, values in nll_values.items()}
            residual_bin_rows.append({
                "dataset": dataset, "residual_percentile_bin": label, "element_count": int(selected.sum()),
                "gaussian_mean_nll": means["gaussian_head"], "laplace_mean_nll": means["laplace_head"],
                "fixed_t_mean_nll": means["student_t_fixed_df8_head"], "learnable_t_mean_nll": means["student_t_head"],
                "laplace_minus_gaussian": means["laplace_head"] - means["gaussian_head"],
                "fixed_t_minus_gaussian": means["student_t_fixed_df8_head"] - means["gaussian_head"],
                "learnable_t_minus_gaussian": means["student_t_head"] - means["gaussian_head"],
                "learnable_t_minus_fixed_t": means["student_t_head"] - means["student_t_fixed_df8_head"],
            })
        learned_df = arrays["student_t_head"][3]
        for channel in range(target.shape[-1]):
            gaussian_residual = ((target[..., channel] - gloc[..., channel]) / gscale[..., channel]).reshape(-1)
            channel_rows.append({
                "dataset": dataset, "channel": channel,
                "gaussian_residual_excess_kurtosis": float(kurtosis(gaussian_residual, fisher=True, bias=False)),
                "learnable_student_t_df": float(learned_df[channel]),
                "learnable_t_minus_gaussian_mean_nll": float((nll_values["student_t_head"][..., channel] - nll_values["gaussian_head"][..., channel]).mean()),
                "learnable_t_minus_fixed_t_mean_nll": float((nll_values["student_t_head"][..., channel] - nll_values["student_t_fixed_df8_head"][..., channel]).mean()),
            })
        local = [row for row in channel_rows if row["dataset"] == dataset]
        vectors = {key: np.asarray([row[key] for row in local]) for key in (
            "gaussian_residual_excess_kurtosis", "learnable_student_t_df",
            "learnable_t_minus_gaussian_mean_nll", "learnable_t_minus_fixed_t_mean_nll",
        )}
        for left, right in (
            ("gaussian_residual_excess_kurtosis", "learnable_student_t_df"),
            ("gaussian_residual_excess_kurtosis", "learnable_t_minus_gaussian_mean_nll"),
            ("learnable_student_t_df", "learnable_t_minus_gaussian_mean_nll"),
        ):
            coefficient = spearmanr(vectors[left], vectors[right]).statistic
            correlation_rows.append({"dataset": dataset, "left": left, "right": right, "spearman_rho": float(coefficient), "descriptive_only": True, "statistical_significance_claim": False})

    write_csv(ROOT / "reports/gate5a_nll_gain_by_residual_bin.csv", residual_bin_rows)
    markdown_table(ROOT / "reports/gate5a_nll_gain_by_residual_bin.md", "Gate 5a NLL Gain by Residual Bin", residual_bin_rows, list(residual_bin_rows[0]))
    write_csv(ROOT / "reports/gate5a_channel_tail_analysis.csv", channel_rows)
    write_csv(ROOT / "reports/gate5a_channel_tail_correlations.csv", correlation_rows)
    write_json_markdown(ROOT / "reports/gate5a_channel_tail_analysis.md", "Gate 5a Channel Tail Analysis", {"status": "PASS", "descriptive_only": True, "statistical_significance_claim": False, "correlations": correlation_rows})

    write_csv(FIGURES / "fig_gate5a_pit_histograms_data.csv", pit_plot_data)
    write_csv(FIGURES / "fig_gate5a_qq_data.csv", qq_plot_data)
    write_csv(FIGURES / "fig_gate5a_tail_nll_gain_data.csv", residual_bin_rows)
    write_csv(FIGURES / "fig_gate5a_kurtosis_df_data.csv", channel_rows)
    fig, axes = plt.subplots(3, 4, figsize=(12, 7), constrained_layout=True)
    for ax, dataset, variant in zip(axes.flat, [d for d in DATASETS for _ in HEADS], list(HEADS) * 3):
        values = next(row for row in pit_rows if row["dataset"] == dataset and row["variant"] == variant)
        ax.bar(np.arange(20), json.loads(values["histogram_20_bins"])); ax.set_title(f"{dataset} {DISPLAY_NAMES[variant]}"); ax.set_xticks([])
    fig.savefig(FIGURES / "fig_gate5a_pit_histograms.png", dpi=320); fig.savefig(FIGURES / "fig_gate5a_pit_histograms.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        for variant in HEADS:
            rows = [row for row in qq_plot_data if row["dataset"] == dataset and row["variant"] == variant]
            ax.plot([r["theoretical"] for r in rows], [r["observed"] for r in rows], label=DISPLAY_NAMES[variant])
        ax.plot([-3, 3], [-3, 3], "k--", lw=.8); ax.set_title(dataset)
    axes[-1].legend(fontsize=7); fig.savefig(FIGURES / "fig_gate5a_qq.png", dpi=320); fig.savefig(FIGURES / "fig_gate5a_qq.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        rows = [row for row in residual_bin_rows if row["dataset"] == dataset]
        for key, label in (("laplace_minus_gaussian", "Laplace"), ("fixed_t_minus_gaussian", "Fixed-t8"), ("learnable_t_minus_gaussian", "Learnable-t")):
            ax.plot(bin_labels, [row[key] for row in rows], marker="o", label=label)
        ax.axhline(0, color="black", lw=.8); ax.set_title(dataset); ax.tick_params(axis="x", rotation=35)
    axes[-1].legend(fontsize=7); fig.savefig(FIGURES / "fig_gate5a_tail_nll_gain.png", dpi=320); fig.savefig(FIGURES / "fig_gate5a_tail_nll_gain.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for ax, dataset in zip(axes, DATASETS):
        rows = [row for row in channel_rows if row["dataset"] == dataset]
        ax.scatter([r["gaussian_residual_excess_kurtosis"] for r in rows], [r["learnable_student_t_df"] for r in rows], s=10); ax.set_title(dataset)
    fig.savefig(FIGURES / "fig_gate5a_kurtosis_df.png", dpi=320); fig.savefig(FIGURES / "fig_gate5a_kurtosis_df.pdf"); plt.close(fig)

    top = [row for row in residual_bin_rows if row["residual_percentile_bin"] in {"95-99", "99-100"}]
    bottom = [row for row in residual_bin_rows if row["residual_percentile_bin"] in {"0-50", "50-90"}]
    top_gain = -float(np.mean([row["learnable_t_minus_gaussian"] for row in top]))
    bottom_gain = -float(np.mean([row["learnable_t_minus_gaussian"] for row in bottom]))
    gaussian_heavy = float(np.mean([row.get("gaussian_z_gt_3", 0) for row in tail_rows if row["variant"] == "gaussian_head"])) > 0.0027
    correlations = [row["spearman_rho"] for row in correlation_rows if row["left"] == "gaussian_residual_excess_kurtosis" and row["right"] == "learnable_student_t_df"]
    mechanism = {
        "status": "PASS", "descriptive_only": True, "statistical_significance_claim": False,
        "GAUSSIAN_RESIDUAL_HEAVY_TAIL_OBSERVED": gaussian_heavy,
        "HEAVY_TAIL_GAIN_CONCENTRATED_IN_EXTREMES": top_gain > bottom_gain,
        "FIXED_T_EXPLAINS_MOST_STUDENT_T_GAIN": decision["GATE5A_FIXED_T_SIGNAL_POSITIVE"] and not decision["GATE5A_LEARNED_DF_SIGNAL_POSITIVE"],
        "LEARNED_DF_ADDS_VALUE_BEYOND_FIXED_T": decision["GATE5A_LEARNED_DF_SIGNAL_POSITIVE"],
        "LAPLACE_COMPETITIVE_WITH_STUDENT_T": decision["GATE5A_LAPLACE_SIGNAL_POSITIVE"],
        "PER_CHANNEL_DF_ASSOCIATED_WITH_CHANNEL_TAILS": float(np.mean(np.abs(correlations))) >= 0.3,
        "STUDENT_T_PARAMETER_COST_MINIMAL": None,
        "STUDENT_T_RUNTIME_COST_MINIMAL": None,
        "top_5_percent_student_t_nll_gain": top_gain,
        "bottom_90_percent_student_t_nll_gain": bottom_gain,
        "predeclared_rules": {
            "tail_concentrated": "mean absolute learnable-t NLL gain in top 5% > bottom 90%",
            "df_association": "mean absolute descriptive Spearman rho across datasets >= 0.3",
        },
    }
    dump(ROOT / "results/gate5a_mechanism_audit.json", mechanism)
    write_json_markdown(ROOT / "reports/gate5a_mechanism_audit.md", "Gate 5a Mechanism Audit", mechanism)
    print(json.dumps(mechanism, indent=2))


if __name__ == "__main__":
    main()
