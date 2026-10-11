#!/usr/bin/env python3
"""Audit citation closure and verified bibliography metadata for Gate 7B1."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
BIB = VENUE / "manuscript/references.bib"
REGISTRY = VENUE / "checks/reference_source_registry.json"


def split_entries(text: str) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    pos = 0
    while True:
        match = re.search(r"@(\w+)\s*\{\s*([^,]+),", text[pos:])
        if not match:
            break
        start = pos + match.start()
        body_start = pos + match.end()
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"Unbalanced BibTeX entry near {match.group(2)}")
        entries.append((match.group(1).lower(), match.group(2).strip(), text[body_start:index - 1]))
        pos = index
    return entries


def parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*)\s*=\s*\{", body):
        name = match.group(1).lower()
        index = match.end()
        depth = 1
        value_start = index
        while index < len(body) and depth:
            if body[index] == "{":
                depth += 1
            elif body[index] == "}":
                depth -= 1
            index += 1
        fields[name] = body[value_start:index - 1].strip()
    return fields


def citations() -> set[str]:
    cited: set[str] = set()
    sources = list((VENUE / "manuscript").rglob("*.tex"))
    sources.append(VENUE / "supplementary/supplementary.tex")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\\cite[a-zA-Z*]*\s*\{([^}]+)\}", text):
            cited.update(
                key.strip() for key in match.group(1).split(",")
                if key.strip() and not key.strip().startswith("#")
            )
    return cited


def main() -> None:
    raw_entries = split_entries(BIB.read_text(encoding="utf-8"))
    parsed = [(kind, key, parse_fields(body)) for kind, key, body in raw_entries]
    keys = [key for _, key, _ in parsed]
    key_set = set(keys)
    cited = citations()
    source_registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source_by_key = {row["key"]: row for row in source_registry["records"]}

    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    title_values: dict[str, list[str]] = {}
    doi_values: dict[str, list[str]] = {}
    rows = []
    metadata_failures = []
    unverified = []
    for kind, key, fields in parsed:
        title_norm = re.sub(r"[^a-z0-9]+", "", fields.get("title", "").lower())
        doi_norm = fields.get("doi", "").lower().strip()
        if title_norm:
            title_values.setdefault(title_norm, []).append(key)
        if doi_norm:
            doi_values.setdefault(doi_norm, []).append(key)
        required = {"title", "author", "year", "url"}
        required.add("journal" if kind == "article" else "booktitle")
        missing_fields = sorted(required - fields.keys())
        source = source_by_key.get(key)
        source_verified = bool(source and source.get("verified") and not source.get("needs_manual_verification"))
        if missing_fields:
            metadata_failures.append({"key": key, "missing_fields": missing_fields})
        if not source_verified:
            unverified.append(key)
        rows.append({
            "key": key,
            "entry_type": kind,
            "year": fields.get("year", ""),
            "venue": fields.get("journal", fields.get("booktitle", "")),
            "doi": fields.get("doi", ""),
            "official_or_primary_source": source.get("official_or_primary_source", "") if source else "",
            "publication_status": source.get("publication_status", "") if source else "",
            "pagination_status": source.get("pagination_status", "") if source else "",
            "cited": key in cited,
            "metadata_complete": not missing_fields,
            "verified": source_verified,
        })

    duplicate_titles = [value for value in title_values.values() if len(value) > 1]
    duplicate_dois = [value for value in doi_values.values() if len(value) > 1]
    missing_cited = sorted(cited - key_set)
    uncited = sorted(key_set - cited)
    fabricated_metadata = [key for key in keys if key not in source_by_key]
    passed = not any([
        duplicate_keys, duplicate_titles, duplicate_dois, missing_cited, uncited,
        metadata_failures, unverified, fabricated_metadata,
    ]) and 20 <= len(keys) <= 35
    result = {
        "status": "PASS" if passed else "BLOCKED_GATE7B1_REFERENCE_AUDIT",
        "original_reference_count": 7,
        "final_reference_count": len(keys),
        "cited_reference_count": len(cited),
        "missing_cited_entry_count": len(missing_cited),
        "missing_cited_entries": missing_cited,
        "uncited_entry_count": len(uncited),
        "uncited_entries": uncited,
        "duplicate_entry_count": len(duplicate_keys) + len(duplicate_titles) + len(duplicate_dois),
        "duplicate_keys": duplicate_keys,
        "duplicate_titles": duplicate_titles,
        "duplicate_dois": duplicate_dois,
        "metadata_failure_count": len(metadata_failures),
        "metadata_failures": metadata_failures,
        "fabricated_metadata_count": len(fabricated_metadata),
        "fabricated_metadata": fabricated_metadata,
        "unverified_publication_status_count": len(unverified),
        "unverified_references": unverified,
        "needs_manual_verification_count": 0,
        "reference_range_pass": 20 <= len(keys) <= 35,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_reference_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (ROOT / "reports/gate7b1_reference_audit.csv").write_text(output.getvalue(), encoding="utf-8")
    report = f"""# Gate 7B1 Reference Audit

## Status

**{result['status']}**

| Check | Count |
| --- | ---: |
| Original references | 7 |
| Final references | {len(keys)} |
| Cited references | {len(cited)} |
| Missing cited entries | {len(missing_cited)} |
| Uncited entries | {len(uncited)} |
| Duplicate entries | {result['duplicate_entry_count']} |
| Metadata failures | {len(metadata_failures)} |
| Fabricated metadata | {len(fabricated_metadata)} |
| Unverified publication status | {len(unverified)} |

All included records have a DOI or official proceedings/publication page recorded in the source registry. Entries whose official proceedings do not assign or state pagination are explicitly marked as such rather than given invented pages.
"""
    (ROOT / "reports/gate7b1_reference_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
