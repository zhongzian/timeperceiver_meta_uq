#!/usr/bin/env python3
"""Verify every displayed Gate 5b value against immutable formal sources."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("etth1", "weather", "electricity")
LABELS = {"etth1": "ETTh1", "weather": "Weather", "electricity": "Electricity"}
TABLE_FIELDS = {
    "dataset",
    "backbone",
    "gaussian_nll",
    "fixed_t8_nll",
    "delta_nll",
    "gaussian_crps",
    "fixed_t8_crps",
    "delta_crps",
    "gaussian_mse",
    "fixed_t8_mse",
    "delta_mse",
    "relative_mse_change_pct",
    "nll_win",
    "crps_noninferior",
    "point_noninferior",
    "full_pair_pass",
}


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def close(left: object, right: object, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def add_number_mismatch(items: list[dict[str, object]], dataset: str, field: str, actual: object, expected: object) -> None:
    items.append({"dataset": dataset, "field": field, "actual": actual, "expected": expected})


def main() -> None:
    result_rows = read_csv(ROOT / "reports/gate5b_patchtst_probability_results.csv")
    pair_rows = read_csv(ROOT / "reports/gate5b_patchtst_paired_deltas.csv")
    decision = read_json(ROOT / "results/gate5b_formal_decision.json")
    table_rows = read_csv(ROOT / "paper/tables/table10_second_backbone_replication.csv")
    second_text = (ROOT / "paper/sections/second_backbone_replication.md").read_text(encoding="utf-8")
    main_text = (ROOT / "paper/sections/main_results.md").read_text(encoding="utf-8")

    by_result = {(row["dataset"], row["variant"]): row for row in result_rows}
    by_pair = {row["dataset"]: row for row in pair_rows}
    patch_rows = {
        row["dataset"].strip().lower(): row
        for row in table_rows
        if row.get("backbone") == "PatchTST"
    }
    number_mismatches: list[dict[str, object]] = []
    interpretation_mismatches: list[str] = []
    required_tokens: list[str] = []

    if not table_rows or not TABLE_FIELDS.issubset(table_rows[0]):
        interpretation_mismatches.append("Table 10 is missing one or more required Gate 6 fields")

    for dataset in DATASETS:
        gaussian = by_result[(dataset, "gaussian_head")]
        fixed = by_result[(dataset, "student_t_fixed_df8_head")]
        pair = by_pair[dataset]
        table = patch_rows.get(dataset)
        if table is None:
            interpretation_mismatches.append(f"Table 10 missing PatchTST row for {dataset}")
            continue

        expected_numbers = {
            "gaussian_nll": float(gaussian["nll"]),
            "fixed_t8_nll": float(fixed["nll"]),
            "delta_nll": float(pair["delta_nll"]),
            "gaussian_crps": float(gaussian["crps"]),
            "fixed_t8_crps": float(fixed["crps"]),
            "delta_crps": float(pair["delta_crps"]),
            "gaussian_mse": float(gaussian["mse"]),
            "fixed_t8_mse": float(fixed["mse"]),
            "delta_mse": float(pair["delta_mse"]),
        }
        for field, expected in expected_numbers.items():
            if not close(table[field], expected):
                add_number_mismatch(number_mismatches, dataset, field, table[field], expected)

        relative_mse = 100.0 * expected_numbers["delta_mse"] / expected_numbers["gaussian_mse"]
        displayed_relative = float(table["relative_mse_change_pct"].rstrip("%"))
        if not math.isclose(displayed_relative, relative_mse, rel_tol=0.0, abs_tol=0.0005):
            add_number_mismatch(
                number_mismatches,
                dataset,
                "relative_mse_change_pct",
                table["relative_mse_change_pct"],
                relative_mse,
            )

        for field in ("nll_win", "crps_noninferior", "point_noninferior", "full_pair_pass"):
            if truth(table[field]) != truth(pair[field]):
                interpretation_mismatches.append(
                    f"{dataset} {field}: table={table[field]} source={pair[field]}"
                )

        required_tokens.extend(
            [
                f"{expected_numbers['gaussian_nll']:.6f}",
                f"{expected_numbers['fixed_t8_nll']:.6f}",
                f"{expected_numbers['delta_nll']:+.6f}",
                f"{expected_numbers['gaussian_crps']:.6f}",
                f"{expected_numbers['fixed_t8_crps']:.6f}",
                f"{expected_numbers['delta_crps']:+.6f}",
                f"{expected_numbers['gaussian_mse']:.6f}",
                f"{expected_numbers['fixed_t8_mse']:.6f}",
                f"{expected_numbers['delta_mse']:+.6f}",
            ]
        )

    for token in required_tokens:
        if token not in second_text:
            number_mismatches.append(
                {"dataset": "manuscript", "field": "required_display_token", "actual": "missing", "expected": token}
            )
    if "1.194%" not in second_text or "+1.194%" not in (ROOT / "paper/tables/table10_second_backbone_replication.md").read_text(encoding="utf-8"):
        number_mismatches.append(
            {"dataset": "electricity", "field": "relative_mse_display", "actual": "missing", "expected": "+1.194%"}
        )

    expected_decision = {
        "GATE5B_NLL_WIN_COUNT": 3,
        "GATE5B_CRPS_NONINFERIOR_COUNT": 3,
        "GATE5B_POINT_NONINFERIOR_COUNT": 2,
        "GATE5B_FULL_PAIR_PASS_COUNT": 2,
        "decision": "SECOND_BACKBONE_POINT_TRADEOFF",
        "GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE": False,
        "CROSS_BACKBONE_DIRECTION_CONSISTENT": False,
    }
    for key, expected in expected_decision.items():
        if decision.get(key) != expected:
            interpretation_mismatches.append(f"formal decision {key}: {decision.get(key)!r} != {expected!r}")

    required_interpretations = [
        ("3/3 NLL wins", second_text),
        ("3/3 CRPS improvements", second_text),
        ("2/3 datasets", second_text),
        ("pre-registered 1% tolerance", second_text),
        ("SECOND_BACKBONE_POINT_TRADEOFF", second_text),
        ("3/3 NLL wins", main_text),
        ("2/3 point-noninferiority", main_text),
        ("2/3 full-pair", main_text),
        ("complete multi-metric gate does not", main_text),
    ]
    for phrase, text in required_interpretations:
        if phrase not in text:
            interpretation_mismatches.append(f"manuscript phrase missing: {phrase}")

    passed = not number_mismatches and not interpretation_mismatches
    payload = {
        "status": "PASS" if passed else "BLOCKED_NUMBER_OR_GATE_MISMATCH",
        "GATE6_NUMBER_VERIFICATION_PASS": passed,
        "generated_at": generated_at(),
        "source_files": [
            "reports/gate5b_patchtst_probability_results.csv",
            "reports/gate5b_patchtst_paired_deltas.csv",
            "results/gate5b_formal_decision.json",
        ],
        "verified_patchtst_rows": len(patch_rows),
        "verified_display_tokens": len(required_tokens) + 1,
        "number_mismatch_count": len(number_mismatches),
        "gate_interpretation_mismatch_count": len(interpretation_mismatches),
        "number_mismatches": number_mismatches,
        "gate_interpretation_mismatches": interpretation_mismatches,
    }
    result_path = ROOT / "results/gate6_number_verification.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Gate 6 Manuscript Number Verification",
        "",
        f"**Status: {payload['status']}**",
        "",
        f"- PatchTST table rows verified: {len(patch_rows)}",
        f"- Display tokens verified: {payload['verified_display_tokens']}",
        f"- Number mismatches: {len(number_mismatches)}",
        f"- Gate interpretation mismatches: {len(interpretation_mismatches)}",
    ]
    if number_mismatches:
        report.extend(["", "## Number Mismatches", "", "```json", json.dumps(number_mismatches, indent=2), "```"])
    if interpretation_mismatches:
        report.extend(["", "## Gate Interpretation Mismatches", ""])
        report.extend(f"- {item}" for item in interpretation_mismatches)
    (ROOT / "reports/gate6_number_verification.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "number_mismatches": len(number_mismatches),
                "gate_interpretation_mismatches": len(interpretation_mismatches),
            },
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
