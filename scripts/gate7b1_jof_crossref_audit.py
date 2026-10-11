#!/usr/bin/env python3
"""Validate Gate 7B1 labels, references, asset paths, and main-item counts."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
MANUSCRIPT = VENUE / "manuscript"


def read_many(paths: list[Path]) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def refs_and_labels(text: str) -> tuple[set[str], set[str]]:
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", text))
    return labels, refs


def main() -> None:
    main_paths = [MANUSCRIPT / "abstract.tex"] + sorted((MANUSCRIPT / "sections").glob("*.tex"))
    supp_paths = sorted((MANUSCRIPT / "appendix").glob("*.tex"))
    main_text = read_many(main_paths)
    supp_text = read_many(supp_paths)
    main_labels, main_refs = refs_and_labels(main_text)
    supp_labels, supp_refs = refs_and_labels(supp_text)
    undefined_main = sorted(main_refs - main_labels)
    undefined_supp = sorted(supp_refs - supp_labels)
    duplicate_main = sorted({label for label in main_labels if sum(path.read_text().count(f"\\label{{{label}}}") for path in main_paths) > 1})
    duplicate_supp = sorted({label for label in supp_labels if sum(path.read_text().count(f"\\label{{{label}}}") for path in supp_paths) > 1})

    graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", main_text + supp_text)
    missing_graphics = sorted({name for name in graphics if not (MANUSCRIPT / "figures" / Path(name).name).is_file()})
    table_inputs = re.findall(r"\\input\{(?:\.\./manuscript/)?tables/([^}]+)\}", main_text + supp_text)
    missing_tables = sorted({name for name in table_inputs if not (MANUSCRIPT / "tables" / f"{name}.tex").is_file()})

    main_table_count = len(re.findall(r"\\begin\{table\}", main_text))
    main_figure_count = len(re.findall(r"\\begin\{figure\}", main_text))
    section_count = len(re.findall(r"\\section\{", main_text))
    expected_tables = {
        "tab:fixed-mask-degradation", "tab:timeperceiver-multiseed",
        "tab:distribution-ablation", "tab:patchtst", "tab:efficiency",
    }
    expected_figures = {
        "fig:missingness-degradation", "fig:timeperceiver-nll",
        "fig:distribution-ablation", "fig:cross-backbone", "fig:tail-gains",
    }
    checks = {
        "main_sections_10": section_count == 10,
        "main_tables_5": main_table_count == 5,
        "main_figures_5": main_figure_count == 5,
        "expected_main_tables": expected_tables.issubset(main_labels),
        "expected_main_figures": expected_figures.issubset(main_labels),
        "undefined_main_references_zero": not undefined_main,
        "undefined_supp_references_zero": not undefined_supp,
        "duplicate_labels_zero": not duplicate_main and not duplicate_supp,
        "missing_graphics_zero": not missing_graphics,
        "missing_tables_zero": not missing_tables,
    }
    passed = all(checks.values())
    result = {
        "status": "PASS" if passed else "FAILED",
        "main_section_count": section_count,
        "main_table_count": main_table_count,
        "main_figure_count": main_figure_count,
        "supplement_section_count": len(re.findall(r"\\section\{", supp_text)),
        "undefined_reference_count": len(undefined_main) + len(undefined_supp),
        "undefined_main_references": undefined_main,
        "undefined_supplement_references": undefined_supp,
        "duplicate_label_count": len(duplicate_main) + len(duplicate_supp),
        "missing_asset_count": len(missing_graphics) + len(missing_tables),
        "missing_graphics": missing_graphics,
        "missing_tables": missing_tables,
        "checks": checks,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_crossref_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Gate 7B1 Cross-Reference Audit

## Status

**{result['status']}**

| Check | Value |
| --- | ---: |
| Main sections | {section_count} |
| Main tables | {main_table_count} |
| Main figures | {main_figure_count} |
| Supplement sections | {result['supplement_section_count']} |
| Undefined references | {result['undefined_reference_count']} |
| Duplicate labels | {result['duplicate_label_count']} |
| Missing assets | {result['missing_asset_count']} |
"""
    (ROOT / "reports/gate7b1_jof_crossref_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
