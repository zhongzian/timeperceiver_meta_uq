#!/usr/bin/env python3
"""Validate Gate 7A LaTeX labels, references, inputs, and figure assets."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
RESULT = ROOT / "results" / "gate7a_cross_reference_validation.json"
REPORT = ROOT / "reports" / "gate7a_cross_reference_validation.md"

MAIN_TABLE_LABELS = [
    "tab:clean-baselines",
    "tab:fixed-mask-degradation",
    "tab:timeperceiver-multiseed",
    "tab:distribution-ablation",
    "tab:efficiency",
    "tab:second-backbone",
]
MAIN_FIGURE_LABELS = [
    "fig:missingness-degradation",
    "fig:multiseed-nll-deltas",
    "fig:multiseed-summary",
    "fig:cross-backbone-deltas",
    "fig:tail-nll",
    "fig:conformal-tradeoff",
]
MAIN_ORDER_FILES = [
    "sections/05_experimental_setup.tex",
    "sections/06_results.tex",
    "sections/07_ablation_negative_results.tex",
]


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def ordered(text: str, labels: list[str]) -> bool:
    positions = [text.find(f"\\label{{{label}}}") for label in labels]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def resolve_asset(value: str, suffix: str | None = None) -> Path:
    path = MANUSCRIPT / value
    if suffix and not path.suffix:
        path = path.with_suffix(suffix)
    return path


def main() -> int:
    tex_files = sorted(path for path in MANUSCRIPT.rglob("*.tex") if "build" not in path.parts)
    documents = {str(path.relative_to(MANUSCRIPT)): strip_comments(path.read_text(encoding="utf-8")) for path in tex_files}
    joined = "\n".join(documents.values())

    labels = re.findall(r"\\label\{([^}]+)\}", joined)
    refs = re.findall(r"\\(?:ref|eqref|pageref|autoref)\{([^}]+)\}", joined)
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    undefined = sorted(set(refs) - set(labels))

    figure_values = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", joined)
    table_values = [value for value in re.findall(r"\\input\{([^}]+)\}", joined) if value.startswith("tables/")]
    missing_figures = sorted(value for value in figure_values if not resolve_asset(value).is_file())
    missing_tables = sorted(value for value in table_values if not resolve_asset(value, ".tex").is_file())

    missing_main_labels = sorted(set(MAIN_TABLE_LABELS + MAIN_FIGURE_LABELS) - set(labels))
    unreferenced_main = sorted(label for label in MAIN_TABLE_LABELS + MAIN_FIGURE_LABELS if label not in refs)
    broken_appendix = sorted(ref for ref in refs if ref.startswith("app:") and ref not in labels)

    ordered_text = "\n".join(documents[path] for path in MAIN_ORDER_FILES)
    wrong_table_numbering = [] if ordered(ordered_text, MAIN_TABLE_LABELS) else ["main table label order"]
    wrong_figure_numbering = [] if ordered(ordered_text, MAIN_FIGURE_LABELS) else ["main figure label order"]

    errors = (
        undefined + duplicates + missing_figures + missing_tables + missing_main_labels
        + unreferenced_main + broken_appendix + wrong_table_numbering + wrong_figure_numbering
    )
    passed = not errors
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_CROSS_REFERENCE_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tex_file_count": len(tex_files),
        "label_count": len(labels),
        "reference_count": len(refs),
        "figure_asset_count": len(figure_values),
        "table_asset_count": len(table_values),
        "undefined_reference_count": len(undefined),
        "duplicate_label_count": len(duplicates),
        "missing_figure_count": len(missing_figures),
        "missing_table_count": len(missing_tables),
        "unreferenced_main_asset_count": len(unreferenced_main),
        "broken_appendix_reference_count": len(broken_appendix),
        "wrong_table_numbering_count": len(wrong_table_numbering),
        "wrong_figure_numbering_count": len(wrong_figure_numbering),
        "undefined_references": undefined,
        "duplicate_labels": duplicates,
        "missing_figures": missing_figures,
        "missing_tables": missing_tables,
        "missing_main_labels": missing_main_labels,
        "unreferenced_main_assets": unreferenced_main,
        "broken_appendix_references": broken_appendix,
        "wrong_table_numbering": wrong_table_numbering,
        "wrong_figure_numbering": wrong_figure_numbering,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    REPORT.write_text(
        "\n".join([
            "# Gate 7A Cross-Reference Validation",
            "",
            f"**Status:** {payload['status']}",
            "",
            f"- Undefined references: {len(undefined)}",
            f"- Duplicate labels: {len(duplicates)}",
            f"- Missing figures: {len(missing_figures)}",
            f"- Missing tables: {len(missing_tables)}",
            f"- Unreferenced main assets: {len(unreferenced_main)}",
            f"- Broken appendix references: {len(broken_appendix)}",
            f"- Wrong table numbering/order: {len(wrong_table_numbering)}",
            f"- Wrong figure numbering/order: {len(wrong_figure_numbering)}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "undefined": len(undefined),
        "duplicates": len(duplicates),
        "missing_assets": len(missing_figures) + len(missing_tables),
        "unreferenced_main": len(unreferenced_main),
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
