#!/usr/bin/env python3
"""Verify canonical evidence provenance and paper-evidence exclusions."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from final_synthesis_lib import (
    ROOT,
    generated_at,
    read_csv_rows,
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)


EXPECTED_STAGE_COUNTS = {
    "gate2_clean_baseline": 3,
    "gate3b_fixed_mask": 48,
    "gate4a_adapter_ablation": 3,
    "gate4b_legacy_unmasked_audit": 2,
    "gate4b_expanded_conditions": 12,
    "gate4b_cross_dataset": 4,
    "gate4b_multiseed": 24,
    "gate4c_probability_head": 2,
    "gate4c_interval_calibration": 12,
    "gate4c_mechanism_audit": 4,
    "gate5a_distribution_ablation": 12,
    "gate5b_second_backbone": 6,
}


def source_row_exists(row: dict[str, Any]) -> bool:
    source = resolve_path(row.get("source_report"))
    if source is None or not source.is_file() or not row.get("source_row_or_key"):
        return False
    match = re.search(r"(?:^|;)row=(\d+)", str(row["source_row_or_key"]))
    if match is None:
        return False
    source_rows = read_csv_rows(source)
    row_number = int(match.group(1))
    return 2 <= row_number <= len(source_rows) + 1


def hash_checks(row: dict[str, Any]) -> list[str]:
    mismatches = []
    pairs = [
        ("checkpoint_path", "checkpoint_sha256"),
        ("source_checkpoint_path", "source_checkpoint_sha256"),
        ("validation_mask_path", "validation_mask_sha256"),
        ("calibration_mask_path", "calibration_mask_sha256"),
        ("test_mask_path", "test_mask_sha256"),
    ]
    for path_key, hash_key in pairs:
        path_value, expected = row.get(path_key), row.get(hash_key)
        if not path_value and not expected:
            continue
        actual = sha256_file(path_value)
        if not actual or actual != expected:
            mismatches.append(f"{row['evidence_id']}:{path_key}")
    return mismatches


def main() -> None:
    registry_path = ROOT / "results/final_evidence_registry.json"
    registry = read_json(registry_path)
    rows = registry["rows"]
    paper_rows = [row for row in rows if row.get("paper_evidence") is True]

    key_fields = (
        "stage",
        "dataset",
        "missing_type",
        "missing_rate",
        "experiment_seed",
        "variant",
        "interval_method",
        "coverage_level",
    )
    keys = [tuple(row.get(field, "NA") for field in key_fields) for row in rows]
    duplicates = [list(key) for key, count in Counter(keys).items() if count > 1]
    missing_sources = [row["evidence_id"] for row in rows if not resolve_path(row.get("source_report")) or not resolve_path(row.get("source_report")).is_file()]
    missing_source_rows = [row["evidence_id"] for row in rows if not source_row_exists(row)]
    hash_mismatches = [problem for row in rows for problem in hash_checks(row)]
    missing_trace_fields = [
        row["evidence_id"]
        for row in paper_rows
        if not all(
            row.get(field)
            for field in (
                "experiment_id",
                "checkpoint_path",
                "checkpoint_sha256",
                "training_protocol_hash",
                "metric_protocol_hash",
            )
        )
    ]
    masked_paper_rows = [row for row in paper_rows if row["missing_type"] != "clean"]
    missing_mask_trace = [
        row["evidence_id"]
        for row in masked_paper_rows
        if not row.get("test_mask_path") or not row.get("test_mask_sha256")
    ]
    invalid_probability_rows = [
        row["evidence_id"]
        for row in paper_rows
        if row["stage"].startswith(("gate4b", "gate5a", "gate5b"))
        and (
            row.get("input_mask_applied") is not True
            or row.get("mask_application_verified") is not True
            or row.get("legacy_unmasked_invalid") is not False
        )
    ]
    stage_counts = Counter(row["stage"] for row in rows)
    stage_count_mismatches = {
        stage: {"expected": expected, "actual": stage_counts.get(stage, 0)}
        for stage, expected in EXPECTED_STAGE_COUNTS.items()
        if stage_counts.get(stage, 0) != expected
    }

    checks = {
        "expected_total_rows_132": len(rows) == 132,
        "expected_stage_counts": not stage_count_mismatches,
        "invalid_legacy_unmasked_paper_rows_zero": not any(
            row.get("legacy_unmasked_invalid") is True for row in paper_rows
        ),
        "invalid_legacy_rows_retained_2": sum(row.get("legacy_unmasked_invalid") is True for row in rows) == 2,
        "smoke_paper_rows_zero": not any(row.get("is_smoke") is True for row in paper_rows),
        "failed_paper_rows_zero": not any(row.get("status") not in {"SUCCESS", "SUCCEEDED"} for row in paper_rows),
        "invalid_paper_rows_zero": not any(row.get("is_valid") is not True for row in paper_rows),
        "missing_source_file_zero": not missing_sources,
        "missing_source_row_or_key_zero": not missing_source_rows,
        "hash_mismatch_zero": not hash_mismatches,
        "duplicate_evidence_key_zero": not duplicates,
        "paper_trace_fields_complete": not missing_trace_fields,
        "masked_paper_rows_have_test_mask": not missing_mask_trace,
        "gate4b_probability_validity": not invalid_probability_rows,
        "source_manifest_pass": read_json("results/final_source_manifest.json").get("status") == "PASS",
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "FAILED",
        "FINAL_EVIDENCE_REGISTRY_PASS": passed,
        "generated_at": generated_at(),
        "counts": {
            "total_evidence_rows": len(rows),
            "valid_formal_rows": sum(row.get("is_valid") is True and row.get("is_formal") is True for row in rows),
            "paper_evidence_rows": len(paper_rows),
            "invalid_legacy_rows_retained": sum(row.get("legacy_unmasked_invalid") is True for row in rows),
            "invalid_legacy_rows_in_paper": sum(row.get("legacy_unmasked_invalid") is True for row in paper_rows),
            "smoke_rows_in_paper": sum(row.get("is_smoke") is True for row in paper_rows),
            "duplicate_keys": len(duplicates),
            "missing_sources": len(missing_sources),
            "missing_source_rows": len(missing_source_rows),
            "hash_mismatches": len(hash_mismatches),
        },
        "checks": checks,
        "details": {
            "stage_counts": dict(stage_counts),
            "stage_count_mismatches": stage_count_mismatches,
            "duplicate_keys": duplicates,
            "missing_sources": missing_sources,
            "missing_source_rows": missing_source_rows,
            "hash_mismatches": hash_mismatches,
            "missing_trace_fields": missing_trace_fields,
            "missing_mask_trace": missing_mask_trace,
            "invalid_probability_rows": invalid_probability_rows,
        },
    }
    write_json("results/final_evidence_registry_verification.json", payload)
    registry["status"] = "PASS" if passed else "FAILED"
    registry["verification"] = payload["counts"]
    write_json(registry_path, registry)

    lines = [
        "# Final Evidence Registry Verification",
        "",
        f"**Status: {payload['status']}**",
        "",
        f"- Total rows: {len(rows)}",
        f"- Valid formal rows: {payload['counts']['valid_formal_rows']}",
        f"- Invalid legacy rows retained: {payload['counts']['invalid_legacy_rows_retained']}",
        f"- Invalid legacy rows in paper: {payload['counts']['invalid_legacy_rows_in_paper']}",
        f"- Smoke rows in paper: {payload['counts']['smoke_rows_in_paper']}",
        f"- Duplicate keys: {payload['counts']['duplicate_keys']}",
        f"- Missing sources: {payload['counts']['missing_sources']}",
        f"- Hash mismatches: {payload['counts']['hash_mismatches']}",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items())
    (ROOT / "reports/final_evidence_registry_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], **payload["counts"]}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
