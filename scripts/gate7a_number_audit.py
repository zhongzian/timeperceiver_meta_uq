#!/usr/bin/env python3
"""Verify Gate 7A manuscript numbers and interpretations against formal assets."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
RESULT = ROOT / "results" / "gate7a_number_audit.json"
REPORT = ROOT / "reports" / "gate7a_number_audit.md"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> int:
    abstract = (MANUSCRIPT / "abstract.tex").read_text(encoding="utf-8")
    setup = (MANUSCRIPT / "sections" / "05_experimental_setup.tex").read_text(encoding="utf-8")
    results_text = (MANUSCRIPT / "sections" / "06_results.tex").read_text(encoding="utf-8")
    negative_text = (MANUSCRIPT / "sections" / "07_ablation_negative_results.tex").read_text(encoding="utf-8")
    limitations = (MANUSCRIPT / "sections" / "08_limitations.tex").read_text(encoding="utf-8")
    all_prose = "\n".join([abstract, setup, results_text, negative_text, limitations])

    gate4a = load_json("results/gate4a_formal_decision.json")
    gate4b = load_json("results/gate4b_multiseed_formal_decision.json")
    gate4c = load_json("results/gate4c_formal_decision.json")
    gate5a = load_json("results/gate5a_formal_decision.json")
    gate5b = load_json("results/gate5b_formal_decision.json")
    assembly = load_json("results/gate7a_manuscript_assembly.json")

    number_mismatches: list[dict[str, object]] = []
    percentage_mismatches: list[dict[str, object]] = []
    interpretation_mismatches: list[str] = []

    expected_decisions = {
        "gate4b_nll": (gate4b.get("GATE4B_MULTI_SEED_NLL_WIN_COUNT"), 12),
        "gate4b_crps": (gate4b.get("GATE4B_MULTI_SEED_CRPS_NONINFERIOR_COUNT"), 12),
        "gate4b_point": (gate4b.get("GATE4B_MULTI_SEED_POINT_NONINFERIOR_COUNT"), 12),
        "gate4b_dataset": (gate4b.get("GATE4B_MULTI_SEED_DATASET_MEAN_PASS_COUNT"), 3),
        "gate5a_fixed_nll": (gate5a.get("GATE5A_FIXED_T_NLL_WIN_COUNT"), 3),
        "gate5a_fixed_crps": (gate5a.get("GATE5A_FIXED_T_CRPS_NONINFERIOR_COUNT"), 3),
        "gate5a_fixed_point": (gate5a.get("GATE5A_FIXED_T_POINT_NONINFERIOR_COUNT"), 3),
        "gate5b_nll": (gate5b.get("GATE5B_NLL_WIN_COUNT"), 3),
        "gate5b_crps": (gate5b.get("GATE5B_CRPS_NONINFERIOR_COUNT"), 3),
        "gate5b_point": (gate5b.get("GATE5B_POINT_NONINFERIOR_COUNT"), 2),
        "gate5b_full": (gate5b.get("GATE5B_FULL_PAIR_PASS_COUNT"), 2),
    }
    for name, (actual, expected) in expected_decisions.items():
        if actual != expected:
            number_mismatches.append({"field": name, "actual": actual, "expected": expected})

    required_phrases = {
        "timeperceiver 12/12 nll": ("12/12 NLL wins", results_text),
        "timeperceiver 12/12 crps": ("12/12 CRPS-noninferior", results_text),
        "timeperceiver 12/12 point": ("12/12 point-noninferior", results_text),
        "timeperceiver 3/3 means": ("3/3 dataset-level mean gates", results_text),
        "patchtst nll 3/3": ("3/3 NLL wins", results_text),
        "patchtst crps 3/3": ("3/3 CRPS-noninferior", results_text),
        "patchtst point 2/3": ("2/3 point-noninferior", results_text),
        "patchtst full 2/3": ("2/3 full paired passes", results_text),
        "electricity gaussian mse": ("0.165829", results_text),
        "electricity fixed mse": ("0.167809", results_text),
        "electricity delta mse": ("+0.001980", results_text),
        "electricity percent": ("+1.194\\%", results_text),
        "point threshold": ("1\\% tolerance", results_text),
        "gate5b decision": ("SECOND\\_BACKBONE\\_POINT\\_TRADEOFF", results_text),
        "input and horizon": ("input length 384 and prediction length 96", setup),
        "mask rates": ("0.1, 0.3, 0.5, and 0.7", setup),
        "mask seed": ("test-mask seed is 2025", setup),
        "head seeds": ("2021, 2022, 2023, and 2025", setup),
        "crps protocol": ("64 Monte Carlo (MC) samples with MC seed 52025", setup),
        "fixed df": ("degrees of freedom fixed at 8", abstract),
        "abstract tradeoff": ("1.194\\%", abstract),
    }
    for name, (phrase, text) in required_phrases.items():
        if phrase not in text:
            number_mismatches.append({"field": name, "actual": "missing", "expected": phrase})

    expected_interpretations = [
        (gate5a.get("decision") == "FIXED_HEAVY_TAIL_SUFFICIENT", "Gate 5a decision is not FIXED_HEAVY_TAIL_SUFFICIENT"),
        (gate5b.get("decision") == "SECOND_BACKBONE_POINT_TRADEOFF", "Gate 5b decision is not SECOND_BACKBONE_POINT_TRADEOFF"),
        (gate4a.get("GATE4A_MISSINGNESS_CONDITIONING_SIGNAL") == "NEGATIVE", "Gate 4a negative decision changed"),
        (gate4c.get("GATE4C_SIGNAL_POSITIVE") is False, "Gate 4c negative decision changed"),
        ("adds no generalization value beyond matched static capacity" in negative_text, "Gate 4a negative result missing from prose"),
        ("worsens both absolute coverage error and interval score" in negative_text, "Gate 4c negative result missing from prose"),
        ("complete multi-metric gate does not" in results_text, "Gate 5b complete-gate failure missing from prose"),
        ("probability-head training seed" in setup, "TimePerceiver seed scope missing"),
        ("rather than full end-to-end backbone retraining" in setup, "TimePerceiver fixed-backbone scope missing"),
    ]
    interpretation_mismatches.extend(message for condition, message in expected_interpretations if not condition)

    patch_results = load_csv("reports/gate5b_patchtst_probability_results.csv")
    patch_pairs = {row["dataset"]: row for row in load_csv("reports/gate5b_patchtst_paired_deltas.csv")}
    table10 = [row for row in load_csv("paper/tables/table10_second_backbone_replication.csv") if row["backbone"] == "PatchTST"]
    patch_by_key = {(row["dataset"], row["variant"]): row for row in patch_results}
    for row in table10:
        dataset = row["dataset"].lower()
        gaussian = patch_by_key[(dataset, "gaussian_head")]
        fixed = patch_by_key[(dataset, "student_t_fixed_df8_head")]
        pair = patch_pairs[dataset]
        comparisons = {
            "gaussian_nll": gaussian["test_mean_nll"],
            "fixed_t8_nll": fixed["test_mean_nll"],
            "delta_nll": pair["delta_nll"],
            "gaussian_crps": gaussian["test_crps"],
            "fixed_t8_crps": fixed["test_crps"],
            "delta_crps": pair["delta_crps"],
            "gaussian_mse": gaussian["test_mse"],
            "fixed_t8_mse": fixed["test_mse"],
            "delta_mse": pair["delta_mse"],
        }
        for field, expected in comparisons.items():
            if not close(row[field], expected):
                number_mismatches.append({"field": f"table10.{dataset}.{field}", "actual": row[field], "expected": expected})
        expected_percent = 100.0 * float(pair["delta_mse"]) / float(gaussian["test_mse"])
        actual_percent = float(row["relative_mse_change_pct"].rstrip("%"))
        if not math.isclose(actual_percent, expected_percent, rel_tol=0, abs_tol=0.0005):
            percentage_mismatches.append({"field": f"table10.{dataset}.relative_mse", "actual": actual_percent, "expected": expected_percent})

    gate5a_rows = load_csv("reports/gate5a_distribution_results.csv")
    gate5a_lookup = {(row["dataset"], row["head"]): row for row in gate5a_rows}
    table8 = load_csv("paper/tables/table8_distribution_head_ablation.csv")
    head_map = {"Gaussian": "Gaussian", "Laplace": "Laplace", "Fixed-t8": "Fixed-t8", "Learnable-t": "Learnable-t"}
    verified_parameter_rows = 0
    for row in table8:
        source = gate5a_lookup.get((row["dataset"], head_map[row["head"]]))
        if source is None:
            number_mismatches.append({"field": f"table8.{row['dataset']}.{row['head']}", "actual": "no source", "expected": "formal Gate 5a row"})
            continue
        for field, source_field in (("nll", "test_mean_nll"), ("crps", "test_crps"), ("mse", "test_mse"), ("trainable_params", "trainable_parameter_count")):
            if not close(row[field], source[source_field]):
                number_mismatches.append({"field": f"table8.{row['dataset']}.{row['head']}.{field}", "actual": row[field], "expected": source[source_field]})
        verified_parameter_rows += 1

    for asset in assembly.get("assets", []):
        source = ROOT / asset["source"]
        destination = ROOT / asset["destination"]
        if not source.is_file() or sha256(source) != asset["source_sha256"]:
            number_mismatches.append({"field": "assembly_source_hash", "actual": str(source), "expected": asset["source_sha256"]})
        if not destination.is_file() or sha256(destination) != asset["destination_sha256"]:
            number_mismatches.append({"field": "assembly_destination_hash", "actual": str(destination), "expected": asset["destination_sha256"]})

    numeric_token_count = len(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\\?%)?", all_prose))
    passed = not number_mismatches and not percentage_mismatches and not interpretation_mismatches
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_NUMBER_AUDIT_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "numeric_prose_token_count": numeric_token_count,
        "verified_table_asset_count": sum(asset.get("kind") == "table" for asset in assembly.get("assets", [])),
        "verified_figure_asset_count": sum(asset.get("kind") == "figure" for asset in assembly.get("assets", [])),
        "verified_gate5a_parameter_rows": verified_parameter_rows,
        "verified_gate5b_rows": len(table10),
        "number_mismatch_count": len(number_mismatches),
        "percentage_mismatch_count": len(percentage_mismatches),
        "interpretation_mismatch_count": len(interpretation_mismatches),
        "number_mismatches": number_mismatches,
        "percentage_mismatches": percentage_mismatches,
        "interpretation_mismatches": interpretation_mismatches,
        "special_gate_checks": {
            "gate4b_12_of_12": gate4b.get("GATE4B_MULTI_SEED_FULL_PAIR_PASS_COUNT") == 12,
            "gate5a_fixed_heavy_tail": gate5a.get("decision") == "FIXED_HEAVY_TAIL_SUFFICIENT",
            "gate5b_nll_3_of_3": gate5b.get("GATE5B_NLL_WIN_COUNT") == 3,
            "gate5b_crps_3_of_3": gate5b.get("GATE5B_CRPS_NONINFERIOR_COUNT") == 3,
            "gate5b_point_2_of_3": gate5b.get("GATE5B_POINT_NONINFERIOR_COUNT") == 2,
            "gate5b_full_2_of_3": gate5b.get("GATE5B_FULL_PAIR_PASS_COUNT") == 2,
            "electricity_plus_1_194_percent": "+1.194\\%" in results_text,
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text("\n".join([
        "# Gate 7A Number Audit",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Numeric prose tokens inventoried: {numeric_token_count}",
        f"- Frozen table assets verified: {payload['verified_table_asset_count']}",
        f"- Frozen figure assets verified: {payload['verified_figure_asset_count']}",
        f"- Gate 5a metric/parameter rows verified: {verified_parameter_rows}",
        f"- Gate 5b rows verified: {len(table10)}",
        f"- Number mismatches: {len(number_mismatches)}",
        f"- Percentage mismatches: {len(percentage_mismatches)}",
        f"- Interpretation mismatches: {len(interpretation_mismatches)}",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "number_mismatches": len(number_mismatches),
        "percentage_mismatches": len(percentage_mismatches),
        "interpretation_mismatches": len(interpretation_mismatches),
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
