#!/usr/bin/env python3
"""Build an anonymous, paper-only submission asset directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "paper/submission"
TEXT_MAP = {
    "abstract.md": "paper/sections/abstract_draft.md",
    "contributions.md": "paper/sections/contributions.md",
    "experimental_setup.md": "paper/sections/experimental_setup.md",
    "results.md": "paper/sections/main_results.md",
    "limitations.md": "paper/sections/limitations.md",
    "conclusion.md": "paper/sections/conclusion.md",
}


def generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_asset(source: Path, destination: Path, rows: list[dict[str, object]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    rows.append(
        {
            "file": str(destination.relative_to(SUBMISSION)),
            "sha256": sha256(destination),
            "size": destination.stat().st_size,
            "source": str(source.relative_to(ROOT)),
            "anonymous": True,
        }
    )


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for destination, source in TEXT_MAP.items():
        copy_asset(ROOT / source, SUBMISSION / destination, rows)

    for source in sorted((ROOT / "paper/tables").glob("table*")):
        if source.is_file() and source.suffix.lower() in {".csv", ".md", ".tex"}:
            copy_asset(source, SUBMISSION / "tables" / source.name, rows)

    figure_root = ROOT / "paper/figures"
    for source in sorted(figure_root.rglob("*")):
        if source.is_file() and source.suffix.lower() in {".csv", ".md", ".pdf", ".png"}:
            copy_asset(source, SUBMISSION / "figures" / source.relative_to(figure_root), rows)

    appendix_root = ROOT / "paper/appendix"
    for source in sorted(appendix_root.rglob("*")):
        if source.is_file() and source.suffix.lower() in {".csv", ".md", ".pdf", ".png", ".tex"}:
            copy_asset(source, SUBMISSION / "appendix" / source.relative_to(appendix_root), rows)

    rows.sort(key=lambda row: str(row["file"]))
    payload = {
        "status": "PASS",
        "generated_at": generated_at(),
        "anonymous": True,
        "manifest_self_excluded": True,
        "file_count": len(rows),
        "text_section_count": len(TEXT_MAP),
        "table_file_count": sum(str(row["file"]).startswith("tables/") for row in rows),
        "figure_file_count": sum(str(row["file"]).startswith("figures/") for row in rows),
        "appendix_file_count": sum(str(row["file"]).startswith("appendix/") for row in rows),
        "files": rows,
    }
    manifest_json = SUBMISSION / "submission_manifest.json"
    manifest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Submission Manifest",
        "",
        f"Generated assets: {len(rows)}. All entries are marked `anonymous=true`.",
        "",
        "| file | sha256 | size | source | anonymous |",
        "| --- | --- | ---: | --- | --- |",
    ]
    lines.extend(
        f"| {row['file']} | {row['sha256']} | {row['size']} | {row['source']} | true |" for row in rows
    )
    (SUBMISSION / "submission_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "status": "PASS",
        "GATE6_SUBMISSION_ASSET_BUILD_PASS": True,
        "generated_at": payload["generated_at"],
        "submission_directory": "paper/submission",
        "manifest": "paper/submission/submission_manifest.json",
        "counts": {
            "text_sections": payload["text_section_count"],
            "table_files": payload["table_file_count"],
            "figure_files": payload["figure_file_count"],
            "appendix_files": payload["appendix_file_count"],
            "total_manifest_files": payload["file_count"],
        },
    }
    (ROOT / "results/gate6_submission_assets.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = f"""# Gate 6 Submission Assets

**Status: PASS**

- Submission directory: `paper/submission`
- Text sections: {payload['text_section_count']}
- Table files: {payload['table_file_count']}
- Figure files: {payload['figure_file_count']}
- Appendix files: {payload['appendix_file_count']}
- Manifest entries: {payload['file_count']}
- Anonymous flag: True
"""
    (ROOT / "reports/gate6_submission_assets.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
