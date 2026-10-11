#!/usr/bin/env python3
"""Assemble anonymous Gate 7A manuscript assets from frozen Gate 6 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
TABLE_DIR = MANUSCRIPT / "tables"
FIGURE_DIR = MANUSCRIPT / "figures"
RESULT_PATH = ROOT / "results" / "gate7a_manuscript_assembly.json"
REPORT_PATH = ROOT / "reports" / "gate7a_manuscript_assembly.md"
MANIFEST_PATH = ROOT / "artifacts" / "gate7a_pre_submission" / "manuscript_assembly_manifest.json"


TABLE_SPECS = [
    {
        "source": "paper/tables/table1_official_clean_baselines.csv",
        "output": "table1_clean_baselines.tex",
        "columns": ["dataset", "mse", "mae", "rmse", "best_epoch"],
        "headers": ["Dataset", "MSE", "MAE", "RMSE", "Best epoch"],
    },
    {
        "source": "paper/tables/table2_fixed_mask_degradation.csv",
        "output": "table2_fixed_mask_degradation.tex",
        "columns": ["dataset", "missing_type", "missing_rate", "mse", "relative_clean_degradation_pct"],
        "headers": ["Dataset", "Mask", "Rate", "MSE", "Clean degradation"],
    },
    {
        "source": "paper/tables/table3_adapter_ablation.csv",
        "output": "table3_adapter_ablation.tex",
        "columns": ["variant", "trainable_parameters", "parameter_overhead_pct", "best_epoch", "validation_mse", "test_mse", "test_mae"],
        "headers": ["Variant", "Trainable", "Overhead", "Best epoch", "Val. MSE", "Test MSE", "Test MAE"],
    },
    {
        "source": "paper/tables/table4_student_t_multiseed.csv",
        "output": "table4_timeperceiver_multiseed.tex",
        "columns": [
            "dataset", "gaussian_nll_mean", "student_t_nll_mean", "mean_paired_delta_nll", "nll_wins",
            "gaussian_crps_mean", "student_t_crps_mean", "mean_paired_delta_crps",
            "gaussian_mse_mean", "student_t_mse_mean", "mean_paired_delta_mse", "point_noninferiority_count",
        ],
        "headers": [
            "Dataset", "G NLL", "T NLL", "$\\Delta$NLL", "NLL wins",
            "G CRPS", "T CRPS", "$\\Delta$CRPS", "G MSE", "T MSE", "$\\Delta$MSE", "Point NI",
        ],
    },
    {
        "source": "paper/tables/table5_etth1_cross_condition.csv",
        "output": "table5_etth1_cross_condition.tex",
        "columns": ["condition", "delta_nll", "delta_crps", "delta_mse", "nll_win", "crps_noninferior", "point_noninferior"],
        "headers": ["Condition", "$\\Delta$NLL", "$\\Delta$CRPS", "$\\Delta$MSE", "NLL win", "CRPS NI", "Point NI"],
    },
    {
        "source": "paper/tables/table6_conformal_calibration.csv",
        "output": "table6_conformal.tex",
        "columns": ["method", "nominal_coverage", "picp", "mpiw", "absolute_coverage_error", "interval_score"],
        "headers": ["Method", "Nominal", "PICP", "MPIW", "Coverage error", "Interval score"],
    },
    {
        "source": "paper/tables/table7_conformal_mechanism.csv",
        "output": "table7_conformal_mechanism.tex",
        "columns": ["head", "nominal_coverage", "native_calibration_picp", "native_test_picp", "coverage_shift", "formal_qhat", "delta_absolute_coverage_error"],
        "headers": ["Head", "Nominal", "Cal. PICP", "Test PICP", "Shift", "$\\hat q$", "$\\Delta$ coverage error"],
    },
    {
        "source": "paper/tables/table8_distribution_head_ablation.csv",
        "output": "table8_distribution_ablation.tex",
        "columns": ["dataset", "head", "nll", "crps", "mse", "coverage_error", "trainable_params"],
        "headers": ["Dataset", "Head", "NLL", "CRPS", "MSE", "Coverage error", "Trainable"],
    },
    {
        "source": "paper/tables/table9_distribution_efficiency.csv",
        "output": "table9_efficiency.tex",
        "columns": ["dataset", "head", "params", "forward_median_ms", "nll_median_ms", "crps_median_ms", "peak_memory_mib"],
        "headers": ["Dataset", "Head", "Params", "Forward ms", "NLL ms", "CRPS ms", "Peak MiB"],
    },
    {
        "source": "paper/tables/table10_second_backbone_replication.csv",
        "output": "table10_second_backbone.tex",
        "columns": ["dataset", "backbone", "delta_nll", "delta_crps", "delta_mse", "relative_mse_change_pct", "nll_win", "crps_noninferior", "point_noninferior", "full_pair_pass"],
        "headers": ["Dataset", "Backbone", "$\\Delta$NLL", "$\\Delta$CRPS", "$\\Delta$MSE", "Rel. MSE", "NLL win", "CRPS NI", "Point NI", "Full pair"],
    },
    {
        "source": "paper/appendix/tableA1_full_gate3b_matrix.csv",
        "output": "tableA1_etth1.tex",
        "columns": ["missing_type", "missing_rate", "mse", "mae", "rmse"],
        "headers": ["Mask", "Rate", "MSE", "MAE", "RMSE"],
        "filter": {"dataset": "ETTh1"},
    },
    {
        "source": "paper/appendix/tableA1_full_gate3b_matrix.csv",
        "output": "tableA1_weather.tex",
        "columns": ["missing_type", "missing_rate", "mse", "mae", "rmse"],
        "headers": ["Mask", "Rate", "MSE", "MAE", "RMSE"],
        "filter": {"dataset": "Weather"},
    },
    {
        "source": "paper/appendix/tableA1_full_gate3b_matrix.csv",
        "output": "tableA1_electricity.tex",
        "columns": ["missing_type", "missing_rate", "mse", "mae", "rmse"],
        "headers": ["Mask", "Rate", "MSE", "MAE", "RMSE"],
        "filter": {"dataset": "Electricity"},
    },
    {
        "source": "paper/appendix/tableA2_gate4c_all_heads.csv",
        "output": "tableA2_conformal_all_heads.tex",
        "columns": ["head", "method", "nominal_coverage", "picp", "mpiw", "absolute_coverage_error", "interval_score"],
        "headers": ["Head", "Method", "Nominal", "PICP", "MPIW", "Coverage error", "Interval score"],
    },
]


FIGURE_SPECS = [
    ("paper/figures/fig1_missingness_degradation.pdf", "fig1_missingness_degradation.pdf"),
    ("paper/figures/fig2_multiseed_nll_deltas.pdf", "fig2_multiseed_nll_deltas.pdf"),
    ("paper/figures/fig3a_multiseed_nll.pdf", "fig3a_multiseed_nll.pdf"),
    ("paper/figures/fig3b_multiseed_crps.pdf", "fig3b_multiseed_crps.pdf"),
    ("paper/figures/fig3c_multiseed_mse.pdf", "fig3c_multiseed_mse.pdf"),
    ("paper/figures/fig5_conformal_tradeoff.pdf", "fig5_conformal_tradeoff.pdf"),
    ("paper/figures/fig6_calibration_test_shift.pdf", "fig6_calibration_test_shift.pdf"),
    ("paper/figures/gate5a/fig_gate5a_tail_nll_gain.pdf", "fig_gate5a_tail_nll_gain.pdf"),
    ("paper/figures/gate5a/fig_gate5a_pit_histograms.pdf", "fig_gate5a_pit_histograms.pdf"),
    ("paper/figures/gate5a/fig_gate5a_qq.pdf", "fig_gate5a_qq.pdf"),
    ("paper/figures/gate5a/fig_gate5a_kurtosis_df.pdf", "fig_gate5a_kurtosis_df.pdf"),
    ("paper/figures/gate5b/fig_gate5b_cross_backbone_deltas.pdf", "fig_gate5b_cross_backbone_deltas.pdf"),
    ("paper/figures/gate5b/fig_gate5b_patchtst_tail_nll_gain.pdf", "fig_gate5b_patchtst_tail_nll_gain.pdf"),
    ("paper/appendix/figA1_gate4a_adapter.pdf", "figA1_gate4a_adapter.pdf"),
    ("paper/appendix/figA2_gate4c_channel_coverage.pdf", "figA2_gate4c_channel_coverage.pdf"),
    ("paper/appendix/figA3_gate4c_horizon_coverage.pdf", "figA3_gate4c_horizon_coverage.pdf"),
]


INTEGER_FIELDS = {
    "best_epoch", "trainable_parameters", "trainable_params", "params", "nll_wins",
    "crps_wins", "point_noninferiority_count",
}
BOOLEAN_FIELDS = {"nll_win", "crps_noninferior", "point_noninferior", "full_pair_pass"}
PERCENT_FIELDS = {"relative_clean_degradation_pct", "parameter_overhead_pct"}
RATE_FIELDS = {"missing_rate", "nominal_coverage"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_value(field: str, raw: str) -> str:
    value = raw.strip()
    if value == "" or value.upper() == "NA":
        return "NA"
    if field in BOOLEAN_FIELDS:
        return "Yes" if value.lower() == "true" else "No"
    if field == "head":
        value = {"Fixed-t8": "Fixed Student-t (df=8)", "Learnable-t": "Learned Student-t"}.get(value, value)
    if field == "condition":
        for prefix in ("random", "block", "channel", "mixed"):
            if value.startswith(prefix):
                value = prefix.title() + " " + value[len(prefix):]
                break
    if field == "missing_type":
        value = value.title()
    if field in INTEGER_FIELDS:
        return f"{int(float(value)):,}"
    if field in PERCENT_FIELDS:
        return f"{float(value):.2f}\\%"
    if field == "relative_mse_change_pct":
        return latex_escape(value)
    if field in RATE_FIELDS:
        return f"{float(value):.2f}"
    try:
        return f"{float(value):.6f}"
    except ValueError:
        return latex_escape(value)


def load_rows(path: Path, row_filter: dict[str, str] | None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if row_filter:
        rows = [row for row in rows if all(row.get(key) == value for key, value in row_filter.items())]
    return rows


def render_table(spec: dict[str, object], rows: list[dict[str, str]]) -> str:
    columns = list(spec["columns"])
    headers = list(spec["headers"])
    alignment = "l" + "r" * (len(columns) - 1)
    lines = [f"\\begin{{tabular}}{{@{{}}{alignment}@{{}}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(format_value(field, row[field]) for field in columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, object]] = []
    errors: list[str] = []

    for spec in TABLE_SPECS:
        source = ROOT / str(spec["source"])
        destination = TABLE_DIR / str(spec["output"])
        if not source.is_file():
            errors.append(f"missing table source: {source.relative_to(ROOT)}")
            continue
        rows = load_rows(source, spec.get("filter"))
        if not rows:
            errors.append(f"no rows selected from: {source.relative_to(ROOT)}")
            continue
        missing_columns = [column for column in spec["columns"] if column not in rows[0]]
        if missing_columns:
            errors.append(f"missing columns in {source.relative_to(ROOT)}: {missing_columns}")
            continue
        destination.write_text(render_table(spec, rows), encoding="utf-8")
        assets.append({
            "kind": "table",
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "destination": str(destination.relative_to(ROOT)),
            "destination_sha256": sha256(destination),
            "row_count": len(rows),
        })

    for source_rel, destination_name in FIGURE_SPECS:
        source = ROOT / source_rel
        destination = FIGURE_DIR / destination_name
        if not source.is_file():
            errors.append(f"missing figure source: {source_rel}")
            continue
        shutil.copyfile(source, destination)
        assets.append({
            "kind": "figure",
            "source": source_rel,
            "source_sha256": sha256(source),
            "destination": str(destination.relative_to(ROOT)),
            "destination_sha256": sha256(destination),
            "identical_copy": sha256(source) == sha256(destination),
        })

    status = "PASS" if not errors else "FAILED"
    payload = {
        "stage": "Gate 7A manuscript assembly",
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "frozen Gate 6 tables and figures; no metric recomputation",
        "table_count": sum(asset["kind"] == "table" for asset in assets),
        "figure_count": sum(asset["kind"] == "figure" for asset in assets),
        "assets": assets,
        "errors": errors,
    }
    write_json(MANIFEST_PATH, payload)
    write_json(RESULT_PATH, payload)

    report_lines = [
        "# Gate 7A Manuscript Assembly",
        "",
        f"**Status:** {status}",
        "",
        "The anonymous manuscript was assembled only from frozen Gate 6 table and figure assets. Metrics were not recomputed.",
        "",
        f"- Generated table fragments: {payload['table_count']}",
        f"- Copied PDF figures: {payload['figure_count']}",
        f"- Errors: {len(errors)}",
        "",
        "## Manifest",
        "",
        f"`{MANIFEST_PATH.relative_to(ROOT)}`",
    ]
    if errors:
        report_lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({key: payload[key] for key in ("status", "table_count", "figure_count", "errors")}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
