#!/usr/bin/env python3
"""Verify paper artifacts, protected formal sources, claims, and final status."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image

from final_synthesis_lib import (
    PROTECTED_FORMAL_SOURCES,
    ROOT,
    UPSTREAM_COMMIT,
    generated_at,
    read_csv_rows,
    read_json,
    resolve_path,
    sha256_file,
    upstream_state,
    write_json,
)


MAIN_TABLES = {
    "table1_official_clean_baselines": 3,
    "table2_fixed_mask_degradation": 18,
    "table3_adapter_ablation": 3,
    "table4_student_t_multiseed": 3,
    "table5_etth1_cross_condition": 6,
    "table6_conformal_calibration": 6,
    "table7_conformal_mechanism": 4,
    "table8_distribution_head_ablation": 12,
    "table9_distribution_efficiency": 12,
    "table10_second_backbone_replication": 6,
}
APPENDIX_TABLES = {"tableA1_full_gate3b_matrix": 48, "tableA2_gate4c_all_heads": 12}
MAIN_FIGURES = {
    "fig1_missingness_degradation": 48,
    "fig2_multiseed_nll_deltas": 12,
    "fig3a_multiseed_nll": 6,
    "fig3b_multiseed_crps": 6,
    "fig3c_multiseed_mse": 6,
    "fig4_student_t_df": 12,
    "fig5_conformal_tradeoff": 6,
    "fig6_calibration_test_shift": 4,
    "gate5a/fig_gate5a_pit_histograms": 240,
    "gate5a/fig_gate5a_qq": 2388,
    "gate5a/fig_gate5a_tail_nll_gain": 15,
    "gate5a/fig_gate5a_kurtosis_df": 349,
    "gate5b/fig_gate5b_patchtst_tail_nll_gain": 15,
    "gate5b/fig_gate5b_cross_backbone_deltas": 6,
}
APPENDIX_FIGURES = {
    "figA1_gate4a_adapter": 3,
    "figA2_gate4c_channel_coverage": 14,
    "figA3_gate4c_horizon_coverage": 192,
}
SECTIONS = [
    "abstract_draft.md",
    "contributions.md",
    "experimental_setup.md",
    "main_results.md",
    "ablation_and_negative_results.md",
    "limitations.md",
    "reproducibility.md",
    "distribution_head_ablation.md",
    "heavy_tail_mechanism.md",
    "second_backbone_replication.md",
    "conclusion.md",
]

TABLE10_REQUIRED_FIELDS = {
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


def table_paths(base: str, appendix: bool = False) -> list[Path]:
    directory = ROOT / ("paper/appendix" if appendix else "paper/tables")
    return [directory / f"{base}{suffix}" for suffix in (".csv", ".md", ".tex")]


def figure_paths(base: str, appendix: bool = False) -> list[Path]:
    directory = ROOT / ("paper/appendix" if appendix else "paper/figures")
    return [directory / f"{base}{suffix}" for suffix in (".png", ".pdf", "_data.csv")]


def latex_fragment_valid(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return (
        text.count(r"\begin{tabular}") == 1
        and text.count(r"\end{tabular}") == 1
        and text.index(r"\begin{tabular}") < text.index(r"\end{tabular}")
        and text.count("\\\\") >= 2
    )


def markdown_table_valid(path: Path) -> bool:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    return len(lines) >= 3 and bool(re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", lines[1]))


def png_valid(path: Path) -> tuple[bool, dict[str, Any]]:
    with Image.open(path) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0, 0))
        minimum_dpi = min(dpi) if isinstance(dpi, tuple) and dpi else 0
        return width > 600 and height > 300 and minimum_dpi >= 299, {
            "width": width,
            "height": height,
            "dpi": list(dpi) if isinstance(dpi, tuple) else dpi,
        }


def all_source_ids(cell: str) -> list[str]:
    return [value for value in str(cell).split(";") if value]


def verify_tables(valid_evidence_ids: set[str]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    passed = True
    for appendix, specifications in ((False, MAIN_TABLES), (True, APPENDIX_TABLES)):
        for name, expected_rows in specifications.items():
            csv_path, md_path, tex_path = table_paths(name, appendix)
            files_exist = all(path.is_file() and path.stat().st_size > 0 for path in (csv_path, md_path, tex_path))
            rows = read_csv_rows(csv_path) if csv_path.is_file() else []
            trace_ids = [evidence_id for row in rows for evidence_id in all_source_ids(row.get("source_evidence_ids", ""))]
            checks = {
                "files_exist_nonempty": files_exist,
                "expected_rows": len(rows) == expected_rows,
                "markdown_parses": md_path.is_file() and markdown_table_valid(md_path),
                "latex_fragment_valid": tex_path.is_file() and latex_fragment_valid(tex_path),
                "all_rows_have_evidence_trace": len(rows) == expected_rows and all(row.get("source_evidence_ids") for row in rows),
                "all_evidence_ids_valid": bool(trace_ids) and all(evidence_id in valid_evidence_ids for evidence_id in trace_ids),
            }
            if name == "table10_second_backbone_replication":
                checks["gate6_required_fields_present"] = bool(rows) and TABLE10_REQUIRED_FIELDS.issubset(rows[0])
            details[name] = {"rows": len(rows), "checks": checks}
            passed = passed and all(checks.values())
    return passed, details


def verify_figures() -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    passed = True
    for appendix, specifications in ((False, MAIN_FIGURES), (True, APPENDIX_FIGURES)):
        for name, expected_rows in specifications.items():
            png_path, pdf_path, data_path = figure_paths(name, appendix)
            files_exist = all(path.is_file() and path.stat().st_size > 0 for path in (png_path, pdf_path, data_path))
            data_rows = read_csv_rows(data_path) if data_path.is_file() else []
            png_ok, png_details = png_valid(png_path) if png_path.is_file() else (False, {})
            checks = {
                "files_exist_nonempty": files_exist,
                "expected_data_rows": len(data_rows) == expected_rows,
                "png_300_dpi_nonblank_dimensions": png_ok,
                "pdf_nonempty": pdf_path.is_file() and pdf_path.stat().st_size > 1000,
            }
            if name == "gate5b/fig_gate5b_cross_backbone_deltas":
                caption = ROOT / "paper/figures/gate5b/fig_gate5b_cross_backbone_deltas_caption.md"
                checks["gate6_caption_present"] = caption.is_file() and caption.stat().st_size > 100
            details[name] = {"data_rows": len(data_rows), "png": png_details, "checks": checks}
            passed = passed and all(checks.values())
    return passed, details


def verify_claims(valid_evidence_ids: set[str]) -> tuple[bool, dict[str, Any]]:
    claims = read_json("results/final_claim_evidence_registry.json")
    rows = claims["claims"]
    missing_sources = []
    missing_evidence = []
    prohibited_hits = []
    patterns = [
        "state-of-the-art",
        "statistically significant improvement",
        "universal superiority",
        "guaranteed calibration",
        "conformal prediction does not work",
        "missingness-aware adapters are generally ineffective",
    ]
    for claim in rows:
        for source in claim["source_files"].split(";"):
            if not resolve_path(source) or not resolve_path(source).is_file():
                missing_sources.append(f"{claim['claim_id']}:{source}")
        for evidence_id in all_source_ids(claim["source_evidence_ids"]):
            if evidence_id not in valid_evidence_ids:
                missing_evidence.append(f"{claim['claim_id']}:{evidence_id}")
        lowered = claim["claim_text"].lower()
        prohibited_hits.extend(f"{claim['claim_id']}:{pattern}" for pattern in patterns if pattern in lowered)
    details = {
        "claim_count": len(rows),
        "supported_count": sum(claim["supported"] is True for claim in rows),
        "unsupported_count": sum(claim["supported"] is not True for claim in rows),
        "missing_sources": missing_sources,
        "missing_evidence_ids": missing_evidence,
        "prohibited_wording_hits": prohibited_hits,
    }
    passed = (
        len(rows) == 9
        and details["supported_count"] == 9
        and details["unsupported_count"] == 0
        and not missing_sources
        and not missing_evidence
        and not prohibited_hits
        and claims.get("FINAL_CLAIM_EVIDENCE_PASS") is True
    )
    return passed, details


def verify_sections(legacy_ids: list[str]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {}
    passed = True
    prohibited_affirmative = [
        "state-of-the-art",
        "statistically significant improvement",
        "guaranteed calibration",
        "universally superior",
    ]
    for name in SECTIONS:
        path = ROOT / "paper/sections" / name
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        checks = {
            "exists_nonempty": len(text.strip()) > 100,
            "legacy_experiment_ids_absent": not any(experiment_id in text for experiment_id in legacy_ids),
            "prohibited_affirmative_wording_absent": not any(pattern in text.lower() for pattern in prohibited_affirmative),
        }
        details[name] = checks
        passed = passed and all(checks.values())
    return passed, details


def verify_sources() -> tuple[bool, dict[str, Any]]:
    before = read_json("results/final_formal_source_hashes_before.json")["hashes"]
    after = {path: sha256_file(path) for path in PROTECTED_FORMAL_SOURCES}
    manifest = read_json("results/final_source_manifest.json")
    manifest_mismatches = [
        row["relative_path"]
        for row in manifest["rows"]
        if sha256_file(row["relative_path"]) != row["sha256"]
    ]
    upstream = upstream_state()
    patchtst_root = ROOT / "upstream/PatchTST"
    patchtst_commit = subprocess.check_output(
        ["git", "-C", str(patchtst_root), "rev-parse", "HEAD"], text=True
    ).strip()
    patchtst_status = subprocess.check_output(
        ["git", "-C", str(patchtst_root), "status", "--short"], text=True
    ).splitlines()
    patchtst_tracked = [line for line in patchtst_status if not line.startswith("??")]
    details = {
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "protected_hashes_unchanged": before == after,
        "source_manifest_mismatches": manifest_mismatches,
        "source_manifest_status": manifest.get("status"),
        "upstream": upstream,
        "patchtst_commit": patchtst_commit,
        "patchtst_tracked_status": patchtst_tracked,
    }
    passed = (
        before == after
        and not manifest_mismatches
        and manifest.get("status") == "PASS"
        and upstream["commit"] == UPSTREAM_COMMIT
        and upstream["tracked_diff_empty"] is True
        and patchtst_commit == "204c21efe0b39603ad6e2ca640ef5896646ab1a9"
        and not patchtst_tracked
    )
    return passed, details


def update_final_status(payload: dict[str, Any]) -> None:
    status = read_json("results/final_project_status.json")
    status.update(
        {
            "status": "PAPER_READY_EVIDENCE_SYNTHESIS_COMPLETE",
            "final_synthesis_pass": True,
            "paper_artifact_verification_pass": True,
            "FINAL_EVIDENCE_REGISTRY_PASS": True,
            "FINAL_METRICS_CONSISTENCY_PASS": True,
            "FINAL_CLAIM_EVIDENCE_PASS": True,
            "FINAL_PAPER_ARTIFACT_PASS": True,
            "FINAL_SYNTHESIS_PASS": True,
            "generated_at": generated_at(),
            "verification_path": "results/paper_artifact_verification.json",
        }
    )
    write_json("results/final_project_status.json", status)
    report = f"""# Final Project Status

**Status: {status['status']}**

- Final synthesis pass: True
- Main contribution: `{status['main_contribution']}`
- Valid formal evidence rows: {status['valid_formal_evidence_count']}
- Invalid legacy rows retained and excluded: {status['invalid_legacy_evidence_count']}
- Main tables: {status['main_table_count']}
- Appendix tables: {status['appendix_table_count']}
- Main figures: {status['main_figure_count']}
- Appendix figures: {status['appendix_figure_count']}
- Claims supported: {status['supported_claim_count']}/{status['claim_count']}
- Metrics consistency: PASS
- Source manifest: PASS
- Paper artifact verification: PASS
- Upstream tracked diff empty: True

This status means that metrics, tables, figures, and narrative drafts have passed internal evidence integration. It does not mean paper acceptance, significance proof, or external review.
"""
    (ROOT / "reports/final_project_status.md").write_text(report, encoding="utf-8")
    synthesis_path = ROOT / "reports/final_project_synthesis.md"
    synthesis = synthesis_path.read_text(encoding="utf-8")
    synthesis = synthesis.replace(
        "Canonical registry, source manifest, metric consistency, tables, claims, and narrative sections are built. Figure generation and final paper-artifact verification determine the terminal synthesis status.",
        "Canonical registry, source manifest, metric consistency, tables, figures, claims, and narrative sections passed final paper-artifact verification. The synthesis status is `PAPER_READY_EVIDENCE_SYNTHESIS_COMPLETE`.",
    )
    synthesis_path.write_text(synthesis, encoding="utf-8")


def main() -> None:
    registry = read_json("results/final_evidence_registry.json")
    valid_evidence_ids = {
        row["evidence_id"] for row in registry["rows"] if row["paper_evidence"] is True and row["is_valid"] is True
    }
    legacy_ids = [row["experiment_id"] for row in registry["rows"] if row["legacy_unmasked_invalid"] is True]
    tables_pass, table_details = verify_tables(valid_evidence_ids)
    figures_pass, figure_details = verify_figures()
    claims_pass, claim_details = verify_claims(valid_evidence_ids)
    sections_pass, section_details = verify_sections(legacy_ids)
    sources_pass, source_details = verify_sources()
    registry_pass = read_json("results/final_evidence_registry_verification.json").get("FINAL_EVIDENCE_REGISTRY_PASS") is True
    metrics_pass = read_json("results/final_metrics_consistency.json").get("FINAL_METRICS_CONSISTENCY_PASS") is True
    legacy_contamination = []
    for directory in (ROOT / "paper/tables", ROOT / "paper/figures", ROOT / "paper/sections", ROOT / "paper/appendix"):
        for path in directory.glob("**/*"):
            if path.suffix not in {".csv", ".md", ".tex"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for experiment_id in legacy_ids:
                if experiment_id in text:
                    legacy_contamination.append(f"{path.relative_to(ROOT)}:{experiment_id}")
    checks = {
        "FINAL_EVIDENCE_REGISTRY_PASS": registry_pass,
        "FINAL_METRICS_CONSISTENCY_PASS": metrics_pass,
        "FINAL_CLAIM_EVIDENCE_PASS": claims_pass,
        "all_required_tables_pass": tables_pass,
        "all_required_figures_pass": figures_pass,
        "all_required_sections_pass": sections_pass,
        "legacy_unmasked_evidence_in_paper_zero": not legacy_contamination,
        "protected_formal_sources_unchanged": sources_pass,
    }
    passed = all(checks.values())
    payload = {
        "status": "PASS" if passed else "BLOCKED_PAPER_ARTIFACT_VERIFICATION",
        "FINAL_PAPER_ARTIFACT_PASS": passed,
        "generated_at": generated_at(),
        "checks": checks,
        "tables": table_details,
        "figures": figure_details,
        "claims": claim_details,
        "sections": section_details,
        "sources": source_details,
        "legacy_contamination": legacy_contamination,
    }
    write_json("results/paper_artifact_verification.json", payload)
    lines = ["# Paper Artifact Verification", "", f"**Status: {payload['status']}**", "", "## Final Checks", ""]
    lines.extend(f"- {name}: {'PASS' if value else 'FAIL'}" for name, value in checks.items())
    lines.extend(["", "## Counts", "", f"- Main tables: {len(MAIN_TABLES)}", f"- Appendix tables: {len(APPENDIX_TABLES)}", f"- Main figures: {len(MAIN_FIGURES)}", f"- Appendix figures: {len(APPENDIX_FIGURES)}", f"- Claims: {claim_details['claim_count']}", f"- Legacy contamination: {len(legacy_contamination)}", ""])
    (ROOT / "reports/paper_artifact_verification.md").write_text("\n".join(lines), encoding="utf-8")
    if passed:
        update_final_status(payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
