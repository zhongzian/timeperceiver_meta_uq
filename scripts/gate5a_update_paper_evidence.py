#!/usr/bin/env python3
"""Build Gate 5a paper tables and sections from verified canonical evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from gate5a_common import HEADS, dump, load, write_csv


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def markdown(path, title, rows, fields):
    text = f"# {title}\n\n"
    text += "| " + " | ".join(fields) + " |\n"
    text += "| " + " | ".join("---" for _ in fields) + " |\n"
    for row in rows:
        text += "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n"
    Path(path).write_text(text)


def latex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def latex(path, rows, fields):
    alignment = "l" + "r" * (len(fields) - 1)
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(field) for field in fields) + " \\\\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_escape(row.get(field, "")) for field in fields) + " \\\\" for row in rows
    )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    decision = load(ROOT / "results/gate5a_formal_decision.json")
    mechanism = load(ROOT / "results/gate5a_mechanism_audit.json")
    efficiency = load(ROOT / "results/gate5a_efficiency.json")
    if decision.get("GATE5A_IMPLEMENTATION_PASS") is not True or mechanism.get("status") != "PASS" or efficiency.get("status") != "PASS":
        raise RuntimeError("Gate 5a formal, mechanism, and efficiency outputs must all pass")
    registry = load(ROOT / "results/final_evidence_registry.json")
    evidence = {
        (row["dataset"], row["variant"]): row["evidence_id"]
        for row in registry["rows"] if row["stage"] == "gate5a_distribution_ablation"
    }
    if len(evidence) != 12:
        raise RuntimeError(f"expected 12 Gate 5a evidence rows, got {len(evidence)}")
    result_rows = read_csv(ROOT / "reports/gate5a_distribution_results.csv")
    table8 = []
    for row in result_rows:
        table8.append({
            "dataset": row["dataset"], "head": row["head"],
            "nll": row["test_mean_nll"], "crps": row["test_crps"], "mse": row["test_mse"],
            "coverage_error": row["mean_absolute_coverage_error"],
            "interval_score": row["mean_interval_score"],
            "trainable_params": row["trainable_parameter_count"],
            "source_evidence_ids": evidence[(row["dataset"], row["variant"])],
        })
    table8.sort(key=lambda row: (("etth1", "weather", "electricity").index(row["dataset"]), ("Gaussian", "Laplace", "Fixed-t8", "Learnable-t").index(row["head"])))
    fields8 = ["dataset", "head", "nll", "crps", "mse", "coverage_error", "interval_score", "trainable_params", "source_evidence_ids"]
    base8 = ROOT / "paper/tables/table8_distribution_head_ablation"
    write_csv(base8.with_suffix(".csv"), table8, fields8)
    markdown(base8.with_suffix(".md"), "Table 8: Distribution Head Ablation", table8, fields8)
    latex(base8.with_suffix(".tex"), table8, fields8)

    table9 = []
    for row in efficiency["rows"]:
        table9.append({
            "dataset": row["dataset"], "head": row["head"], "batch_size": row["batch_size"],
            "gpu_name": row["gpu_name"], "params": row["trainable_parameter_count"],
            "overhead": row["parameter_overhead_ratio"], "forward_median_ms": row["forward_median_ms"],
            "nll_median_ms": row["forward_nll_median_ms"], "interval_median_ms": row["forward_interval_median_ms"],
            "crps_median_ms": row["forward_crps64_median_ms"], "peak_memory_mib": row["benchmark_peak_allocated_mib"],
            "source_evidence_ids": evidence[(row["dataset"], row["variant"])],
        })
    fields9 = ["dataset", "head", "batch_size", "gpu_name", "params", "overhead", "forward_median_ms", "nll_median_ms", "interval_median_ms", "crps_median_ms", "peak_memory_mib", "source_evidence_ids"]
    base9 = ROOT / "paper/tables/table9_distribution_efficiency"
    write_csv(base9.with_suffix(".csv"), table9, fields9)
    markdown(base9.with_suffix(".md"), "Table 9: Distribution Efficiency", table9, fields9)
    latex(base9.with_suffix(".tex"), table9, fields9)

    section_ablation = f"""# Distribution Head Ablation

Gate 5a compares matched Gaussian, Laplace, fixed-df Student-t, and per-channel learnable-df Student-t heads on frozen official TimePerceiver checkpoints for ETTh1, Weather, and Electricity under the corrected random-0.3 fixed-mask protocol. The formal decision is `{decision['decision']}`. All heads share the same location and scale projections; Laplace and fixed-t8 have exactly the Gaussian trainable capacity, while learnable-t adds one df parameter per target channel.

The comparison uses a single probability-head training seed with fixed backbones. It does not establish formal significance, superiority beyond the evaluated scope, or novelty of Student-t forecasting itself. Table 8 reports every distribution and dataset rather than selecting only the best head. Fixed df=8 passes the matched Gaussian gates in all three datasets, whereas learned per-channel df does not add a consistent benefit over fixed df=8.
"""
    (ROOT / "paper/sections/distribution_head_ablation.md").write_text(section_ablation)
    section_mechanism = f"""# Heavy-Tail Mechanism

PIT, distribution-normalized QQ, Gaussian-residual exceedance, residual-percentile NLL decomposition, and channel-level kurtosis/df analyses are secondary descriptive evidence. They do not replace the formal metric gates and do not support p-value or significance claims.

- Gaussian residual heavy tail observed: `{mechanism['GAUSSIAN_RESIDUAL_HEAVY_TAIL_OBSERVED']}`
- Heavy-tail gain concentrated in extremes: `{mechanism['HEAVY_TAIL_GAIN_CONCENTRATED_IN_EXTREMES']}`
- Fixed-t explains most Student-t gain: `{mechanism['FIXED_T_EXPLAINS_MOST_STUDENT_T_GAIN']}`
- Learned df adds value beyond fixed-t: `{mechanism['LEARNED_DF_ADDS_VALUE_BEYOND_FIXED_T']}`
- Per-channel df associated with channel tails: `{mechanism['PER_CHANNEL_DF_ASSOCIATED_WITH_CHANNEL_TAILS']}`

The controlled latency benchmark uses the same GPU, batch, precision, warmup, and iteration count within each dataset. Formal training wall times from different GPU models are retained but not treated as strict speed comparisons.
"""
    (ROOT / "paper/sections/heavy_tail_mechanism.md").write_text(section_mechanism)
    payload = {
        "status": "PASS", "table8_rows": len(table8), "table9_rows": len(table9),
        "diagnostics_secondary_descriptive_evidence": True,
        "statistical_significance_claim": False,
        "canonical_gate5a_rows": len(evidence),
    }
    dump(ROOT / "results/gate5a_paper_evidence_update.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
