#!/usr/bin/env python3
"""Create the minimal Gate 5b paper increment from verified formal evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from final_synthesis_lib import read_csv_rows, read_json, write_csv, write_json, write_table_bundle  # noqa: E402


DATASETS = ("etth1", "weather", "electricity")
LABELS = {"etth1": "ETTh1", "weather": "Weather", "electricity": "Electricity"}


def truth(value):
    return str(value).lower() == "true"


def main():
    decision = read_json("results/gate5b_formal_decision.json")
    if decision.get("GATE5B_IMPLEMENTATION_PASS") is not True:
        raise RuntimeError("Gate 5b implementation is incomplete")
    registry = read_json("results/final_evidence_registry.json")
    evidence = {
        (row["stage"], row["dataset"], row["variant"]): row["evidence_id"]
        for row in registry["rows"]
        if row["stage"] in {"gate5a_distribution_ablation", "gate5b_second_backbone"}
        and row.get("paper_evidence") is True
    }

    timeperceiver_results = read_csv_rows("reports/gate5a_distribution_results.csv")
    timeperceiver_pairs = {
        row["dataset"]: row
        for row in read_csv_rows("reports/gate5a_pairwise_deltas.csv")
        if row["comparison"] == "Fixed-t vs Gaussian"
    }
    patchtst_results = read_csv_rows("reports/gate5b_patchtst_probability_results.csv")
    patchtst_pairs = {row["dataset"]: row for row in read_csv_rows("reports/gate5b_patchtst_paired_deltas.csv")}

    table_rows = []
    figure_rows = []
    for dataset in DATASETS:
        tp = {
            row["variant"]: row
            for row in timeperceiver_results
            if row["dataset"] == dataset and row["variant"] in {"gaussian_head", "student_t_fixed_df8_head"}
        }
        pt = {row["variant"]: row for row in patchtst_results if row["dataset"] == dataset}
        for backbone, pair, values, stage in (
            ("TimePerceiver", timeperceiver_pairs[dataset], tp, "gate5a_distribution_ablation"),
            ("PatchTST", patchtst_pairs[dataset], pt, "gate5b_second_backbone"),
        ):
            gaussian = values["gaussian_head"]
            fixed = values["student_t_fixed_df8_head"]
            if backbone == "TimePerceiver":
                gaussian_nll, fixed_nll = float(gaussian["test_mean_nll"]), float(fixed["test_mean_nll"])
                gaussian_crps, fixed_crps = float(gaussian["test_crps"]), float(fixed["test_crps"])
                gaussian_mse, fixed_mse = float(gaussian["test_mse"]), float(fixed["test_mse"])
                nll_win = truth(pair["nll_win"])
                crps_noninferior = truth(pair["crps_noninferior"])
                point_noninferior = truth(pair["point_noninferior"])
                pair_pass = nll_win and crps_noninferior and point_noninferior
            else:
                gaussian_nll, fixed_nll = float(gaussian["nll"]), float(fixed["nll"])
                gaussian_crps, fixed_crps = float(gaussian["crps"]), float(fixed["crps"])
                gaussian_mse, fixed_mse = float(gaussian["mse"]), float(fixed["mse"])
                nll_win = truth(pair["nll_win"])
                crps_noninferior = truth(pair["crps_noninferior"])
                point_noninferior = truth(pair["point_noninferior"])
                pair_pass = truth(pair["full_pair_pass"])
            relative_mse_change = 100.0 * (fixed_mse - gaussian_mse) / gaussian_mse
            source_ids = ";".join((
                evidence[(stage, dataset, "gaussian_head")],
                evidence[(stage, dataset, "student_t_fixed_df8_head")],
            ))
            table_rows.append({
                "dataset": LABELS[dataset],
                "backbone": backbone,
                "gaussian_nll": gaussian_nll,
                "fixed_t8_nll": fixed_nll,
                "delta_nll": fixed_nll - gaussian_nll,
                "gaussian_crps": gaussian_crps,
                "fixed_t8_crps": fixed_crps,
                "delta_crps": fixed_crps - gaussian_crps,
                "gaussian_mse": gaussian_mse,
                "fixed_t8_mse": fixed_mse,
                "delta_mse": fixed_mse - gaussian_mse,
                "relative_mse_change_pct": f"{relative_mse_change:+.3f}%",
                "nll_win": nll_win,
                "crps_noninferior": crps_noninferior,
                "point_noninferior": point_noninferior,
                "full_pair_pass": pair_pass,
                "source_evidence_ids": source_ids,
            })
            figure_rows.append({
                "dataset": dataset,
                "backbone": backbone,
                "delta_nll": fixed_nll - gaussian_nll,
                "delta_crps": fixed_crps - gaussian_crps,
                "delta_mse": fixed_mse - gaussian_mse,
                "source_evidence_ids": source_ids,
            })

    fields = list(table_rows[0])
    write_table_bundle(
        "paper/tables/table10_second_backbone_replication",
        table_rows,
        fields,
        "Fixed-t8 Results Across the Two Evaluated Backbones",
        (
            "Deltas are fixed-t8 minus Gaussian within each backbone; negative values favor fixed-t8. "
            "The point-noninferiority threshold was pre-registered as a maximum 1% relative MSE increase. "
            "Absolute metrics are not used to rank backbones."
        ),
        labels={
            "dataset": "Dataset",
            "backbone": "Backbone",
            "gaussian_nll": "Gaussian NLL",
            "fixed_t8_nll": "Fixed-t NLL",
            "delta_nll": "Delta NLL",
            "gaussian_crps": "Gaussian CRPS",
            "fixed_t8_crps": "Fixed-t CRPS",
            "delta_crps": "Delta CRPS",
            "gaussian_mse": "Gaussian MSE",
            "fixed_t8_mse": "Fixed-t MSE",
            "delta_mse": "Delta MSE",
            "relative_mse_change_pct": "Relative MSE change",
            "nll_win": "NLL win",
            "crps_noninferior": "CRPS noninferior",
            "point_noninferior": "Point noninferior",
            "full_pair_pass": "Full-pair pass",
            "source_evidence_ids": "Evidence IDs",
        },
    )

    figure_dir = ROOT / "paper/figures/gate5b"
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(figure_dir / "fig_gate5b_cross_backbone_deltas_data.csv", figure_rows, list(figure_rows[0]))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    colors = {"TimePerceiver": "#2f6f9f", "PatchTST": "#b34d5d"}
    for axis, metric in zip(axes, ("delta_nll", "delta_crps", "delta_mse")):
        centers = np.arange(len(DATASETS))
        width = 0.34
        for offset, backbone in ((-width / 2, "TimePerceiver"), (width / 2, "PatchTST")):
            local = [next(row for row in figure_rows if row["dataset"] == dataset and row["backbone"] == backbone) for dataset in DATASETS]
            axis.bar(centers + offset, [row[metric] for row in local], width, label=backbone, color=colors[backbone])
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(centers, [LABELS[dataset] for dataset in DATASETS])
        axis.set_title(metric.replace("delta_", "Delta ").upper())
        axis.set_ylabel("Fixed-t8 - Gaussian (negative favors fixed-t8)")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].legend(frameon=False)
    fig.suptitle("Within-backbone fixed-t8 deltas")
    fig.savefig(figure_dir / "fig_gate5b_cross_backbone_deltas.png", dpi=320, facecolor="white")
    fig.savefig(figure_dir / "fig_gate5b_cross_backbone_deltas.pdf", facecolor="white")
    plt.close(fig)

    (figure_dir / "fig_gate5b_cross_backbone_deltas_caption.md").write_text(
        "**Cross-backbone metric deltas.** Bars show fixed-t8 minus Gaussian, with the zero line "
        "marking no change and negative values indicating improvement. PatchTST produces NLL and "
        "CRPS improvements across all three datasets, while Electricity exhibits a small positive "
        "MSE delta. No significance inference is attached to these descriptive comparisons.\n",
        encoding="utf-8",
    )

    (ROOT / "paper/sections/second_backbone_replication.md").write_text(
        """# Second-Backbone Replication

## Protocol

We evaluate supervised PatchTST with input length 384, prediction length 96, random historical missingness at rate 0.3, and seed 2025. Capacity-matched Gaussian and fixed Student-t (df=8) heads use the same frozen clean PatchTST backbone and the same fixed validation and test masks. Future targets remain unmasked and are not used for model selection.

## Results

- ETTh1: Gaussian versus fixed-t8 NLL is 0.892433 versus 0.857116 (delta -0.035317), CRPS is 0.338364 versus 0.335825 (delta -0.002538), and MSE is 0.426829 versus 0.421389 (delta -0.005439).
- Weather: Gaussian versus fixed-t8 NLL is 2.096352 versus 0.277655 (delta -1.818698), CRPS is 0.274037 versus 0.197832 (delta -0.076205), and MSE is 0.232359 versus 0.182180 (delta -0.050179).
- Electricity: Gaussian versus fixed-t8 NLL is 0.413581 versus 0.330148 (delta -0.083433), CRPS is 0.206058 versus 0.204371 (delta -0.001688), and MSE is 0.165829 versus 0.167809 (delta +0.001980).

## Gate Interpretation

Fixed-t8 records 3/3 NLL wins and 3/3 CRPS improvements/noninferiority outcomes. Point-prediction noninferiority and the full-pair gate each pass in 2/3 datasets. The formal decision is `SECOND_BACKBONE_POINT_TRADEOFF`.

## Electricity Tradeoff

The Electricity MSE increases from 0.165829 to 0.167809, corresponding to approximately 1.194%, which is slightly above the pre-registered 1% tolerance. This threshold miss remains visible despite improvements in NLL, CRPS, MAE, and coverage error.

## Tail Diagnostic

The residual-percentile diagnostic covers approximately 1.82e8 forecast elements. Most NLL gains are concentrated in the most extreme Gaussian-residual regions. This is descriptive analysis only, is not a formal gate, and is not a causal mechanism proof; the element count is not treated as an independent sample size for inference.

## Conclusion

The result supports transfer of the probabilistic-score direction across the two evaluated backbones, but not complete cross-backbone multi-metric consistency. Its scope is three datasets, random missing rate 0.3, and one PatchTST backbone/head seed.
""",
        encoding="utf-8",
    )
    payload = {
        "status": "PASS",
        "table10_rows": len(table_rows),
        "cross_backbone_figure_rows": len(figure_rows),
        "tail_figure_rows": len(read_csv_rows("reports/gate5b_patchtst_tail_nll_gain.csv")),
        "figure_caption_created": True,
        "decision": decision["decision"],
        "statistical_significance_claim": False,
        "broad_transfer_claim": False,
        "absolute_backbone_ranking": False,
    }
    write_json("results/gate5b_paper_update.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
