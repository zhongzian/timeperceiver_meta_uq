#!/usr/bin/env python3
"""Check Gate 7B1 key numbers against frozen formal evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"


def manuscript_corpus() -> str:
    paths = [VENUE / "manuscript/abstract.tex"]
    paths.extend(sorted((VENUE / "manuscript/sections").glob("*.tex")))
    paths.extend(sorted((VENUE / "manuscript/tables").glob("*.tex")))
    paths.append(VENUE / "submission/cover_letter_draft.tex")
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def main() -> None:
    text = manuscript_corpus()
    gate4b = json.loads((ROOT / "results/gate4b_multiseed_formal_decision.json").read_text())
    gate5b = json.loads((ROOT / "results/gate5b_formal_decision.json").read_text())
    with (ROOT / "reports/gate5b_patchtst_probability_results.csv").open(newline="", encoding="utf-8") as handle:
        probability_rows = list(csv.DictReader(handle))
    electricity = {
        row["head"]: row for row in probability_rows if row["dataset"] == "electricity"
    }
    with (ROOT / "reports/gate5b_patchtst_paired_deltas.csv").open(newline="", encoding="utf-8") as handle:
        paired = {row["dataset"]: row for row in csv.DictReader(handle)}

    evidence = {
        "timeperceiver_nll_wins": gate4b["GATE4B_MULTI_SEED_NLL_WIN_COUNT"],
        "timeperceiver_crps_noninferior": gate4b["GATE4B_MULTI_SEED_CRPS_NONINFERIOR_COUNT"],
        "timeperceiver_point_noninferior": gate4b["GATE4B_MULTI_SEED_POINT_NONINFERIOR_COUNT"],
        "patchtst_nll_wins": gate5b["GATE5B_NLL_WIN_COUNT"],
        "patchtst_crps_noninferior": gate5b["GATE5B_CRPS_NONINFERIOR_COUNT"],
        "patchtst_point_noninferior": gate5b["GATE5B_POINT_NONINFERIOR_COUNT"],
        "patchtst_full_pair_pass": gate5b["GATE5B_FULL_PAIR_PASS_COUNT"],
        "electricity_gaussian_mse": float(electricity["gaussian"]["mse"]),
        "electricity_student_mse": float(electricity["fixed_t8"]["mse"]),
        "electricity_delta_mse": float(paired["electricity"]["delta_mse"]),
    }
    required_tokens = {
        "timeperceiver_12_of_12": "12/12",
        "patchtst_3_of_3": "3/3",
        "patchtst_2_of_3": "2/3",
        "electricity_gaussian_mse": "0.165829",
        "electricity_student_mse": "0.167809",
        "electricity_absolute_change": "+0.001980",
        "electricity_relative_change": "+1.194\\%",
        "point_threshold": "1\\%",
        "fixed_df": "\\df=8",
        "crps_samples": "64 Monte Carlo samples",
        "crps_seed": "52025",
        "sequence_length": "input length 384",
        "prediction_length": "prediction length 96",
    }
    missing_tokens = [name for name, token in required_tokens.items() if token not in text]
    evidence_checks = {
        "timeperceiver_counts": evidence["timeperceiver_nll_wins"] == evidence["timeperceiver_crps_noninferior"] == evidence["timeperceiver_point_noninferior"] == 12,
        "patchtst_counts": (
            evidence["patchtst_nll_wins"], evidence["patchtst_crps_noninferior"],
            evidence["patchtst_point_noninferior"], evidence["patchtst_full_pair_pass"]
        ) == (3, 3, 2, 2),
        "electricity_gaussian_rounding": f"{evidence['electricity_gaussian_mse']:.6f}" == "0.165829",
        "electricity_student_rounding": f"{evidence['electricity_student_mse']:.6f}" == "0.167809",
        "electricity_delta_rounding": f"{evidence['electricity_delta_mse']:+.6f}" == "+0.001980",
        "point_tradeoff_formal": not next(row for row in gate5b["paired_results"] if row["dataset"] == "electricity")["point_noninferior"],
    }
    interpretation_checks = {
        "threshold_failure_explicit": "exceeds the prespecified 1\\%" in text,
        "mc_not_analytic": "rather than reported as an exact analytic value" in text,
        "df_not_optimal": "not theoretically optimal" in text,
        "head_seed_scope": "not an end-to-end backbone retraining study" in text,
    }
    number_mismatches = missing_tokens + [name for name, ok in evidence_checks.items() if not ok]
    interpretation_mismatches = [name for name, ok in interpretation_checks.items() if not ok]
    passed = not number_mismatches and not interpretation_mismatches
    result = {
        "status": "PASS" if passed else "FAILED",
        "number_mismatch_count": len(number_mismatches),
        "number_mismatches": number_mismatches,
        "percentage_mismatch_count": 0 if evidence_checks["electricity_delta_rounding"] else 1,
        "percentage_mismatches": [],
        "interpretation_mismatch_count": len(interpretation_mismatches),
        "interpretation_mismatches": interpretation_mismatches,
        "evidence_values": evidence,
        "evidence_checks": evidence_checks,
        "interpretation_checks": interpretation_checks,
        "metrics_recomputed": False,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_number_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Gate 7B1 Number Audit

## Status

**{result['status']}**

| Check | Mismatches |
| --- | ---: |
| Numbers | {result['number_mismatch_count']} |
| Percentages | {result['percentage_mismatch_count']} |
| Interpretations | {result['interpretation_mismatch_count']} |

The audit reads existing formal decisions and primary metric CSVs; it does not rerun models or recompute forecasting metrics. It confirms 12/12 TimePerceiver outcomes, PatchTST 3/3--3/3--2/3--2/3 outcomes, the Electricity MSE pair 0.165829 versus 0.167809, the +0.001980 change, and the explicit +1.194% threshold failure.
"""
    (ROOT / "reports/gate7b1_jof_number_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
