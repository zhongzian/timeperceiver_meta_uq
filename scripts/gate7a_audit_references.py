#!/usr/bin/env python3
"""Audit manuscript citations against conservatively verified bibliography metadata."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
RESULT = ROOT / "results" / "gate7a_reference_audit.json"
REPORT_CSV = ROOT / "reports" / "gate7a_reference_audit.csv"
REPORT_MD = ROOT / "reports" / "gate7a_reference_audit.md"

VERIFIED = {
    "lee2025timeperceiver": {
        "title": "TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting",
        "year": "2025",
        "status": "published conference paper",
        "source": "https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html",
    },
    "nie2023patchtst": {
        "title": "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers",
        "year": "2023",
        "status": "published conference paper; arXiv identifier retained",
        "source": "https://openreview.net/forum?id=Jbdc0vTOcol",
    },
    "salinas2020deepar": {
        "title": "{DeepAR}: Probabilistic Forecasting with Autoregressive Recurrent Networks",
        "year": "2020",
        "status": "published journal article",
        "source": "https://doi.org/10.1016/j.ijforecast.2019.07.001",
    },
    "gneiting2007proper": {
        "title": "Strictly Proper Scoring Rules, Prediction, and Estimation",
        "year": "2007",
        "status": "published journal article",
        "source": "https://doi.org/10.1198/016214506000001437",
    },
    "cao2018brits": {
        "title": "{BRITS}: Bidirectional Recurrent Imputation for Time Series",
        "year": "2018",
        "status": "published conference paper; arXiv identifier retained",
        "source": "https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
    },
    "tashiro2021csdi": {
        "title": "{CSDI}: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation",
        "year": "2021",
        "status": "published conference paper; arXiv identifier retained",
        "source": "https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html",
    },
    "stankeviciute2021conformal": {
        "title": "Conformal Time-Series Forecasting",
        "year": "2021",
        "status": "published conference paper",
        "source": "https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html",
    },
}


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.replace("{", "").replace("}", "").lower())


def parse_bibtex(text: str) -> list[dict[str, str]]:
    entries = []
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,]+),\s*(.*?)(?=^\})", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        fields: dict[str, str] = {"entry_type": match.group(1).lower(), "key": match.group(2).strip()}
        for field in re.finditer(r"(?m)^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$", match.group(3)):
            fields[field.group(1).lower()] = field.group(2).strip()
        entries.append(fields)
    return entries


def main() -> int:
    bib_path = MANUSCRIPT / "references.bib"
    entries = parse_bibtex(bib_path.read_text(encoding="utf-8"))
    tex = "\n".join(
        path.read_text(encoding="utf-8")
        for path in MANUSCRIPT.rglob("*.tex")
        if "build" not in path.parts and "tables" not in path.parts
    )
    cited = sorted({key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex) for key in group.split(",")})
    keys = [entry["key"] for entry in entries]
    by_key = {entry["key"]: entry for entry in entries}

    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    title_groups: dict[str, list[str]] = {}
    for entry in entries:
        title_groups.setdefault(normalize_title(entry.get("title", "")), []).append(entry["key"])
    duplicate_titles = sorted(group for title, group in title_groups.items() if title and len(group) > 1)
    missing_cited = sorted(set(cited) - set(keys))
    uncited = sorted(set(keys) - set(cited))

    rows = []
    incomplete = []
    fabricated = []
    manual = []
    for entry in entries:
        key = entry["key"]
        venue = entry.get("journal") or entry.get("booktitle") or ""
        locator = entry.get("doi") or entry.get("eprint") or entry.get("url") or ""
        missing_fields = [name for name, value in (
            ("title", entry.get("title")),
            ("author", entry.get("author")),
            ("year", entry.get("year")),
            ("venue", venue),
            ("doi/arXiv/url", locator),
        ) if not value]
        verified = VERIFIED.get(key)
        metadata_match = bool(
            verified
            and normalize_title(entry.get("title", "")) == normalize_title(verified["title"])
            and entry.get("year") == verified["year"]
        )
        if missing_fields:
            incomplete.append({"key": key, "missing_fields": missing_fields})
        if not metadata_match:
            fabricated.append(key)
        needs_manual = bool(missing_fields or not metadata_match)
        if needs_manual:
            manual.append(key)
        rows.append({
            "citation_key": key,
            "title": entry.get("title", ""),
            "authors": entry.get("author", ""),
            "year": entry.get("year", ""),
            "venue": venue,
            "doi_or_arxiv": entry.get("doi") or entry.get("eprint") or "",
            "url": entry.get("url", ""),
            "cited_in_manuscript": key in cited,
            "duplicate_entry": key in duplicate_keys or any(key in group for group in duplicate_titles),
            "metadata_complete": not missing_fields,
            "publication_status": verified["status"] if verified else "unverified",
            "verification_source": verified["source"] if verified else "",
            "needs_manual_verification": needs_manual,
        })

    passed = not missing_cited and not duplicate_keys and not duplicate_titles and not fabricated
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_REFERENCE_AUDIT_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bibliography_entry_count": len(entries),
        "cited_entry_count": len(cited),
        "missing_cited_entry_count": len(missing_cited),
        "duplicate_bibliography_entry_count": len(duplicate_keys) + len(duplicate_titles),
        "uncited_bibliography_entry_count": len(uncited),
        "incomplete_metadata_count": len(incomplete),
        "fabricated_metadata_count": len(fabricated),
        "manual_verification_required_count": len(manual),
        "missing_cited_entries": missing_cited,
        "duplicate_keys": duplicate_keys,
        "duplicate_titles": duplicate_titles,
        "uncited_bibliography_entries": uncited,
        "incomplete_metadata": incomplete,
        "fabricated_or_unverified_metadata": fabricated,
        "manual_verification_required": manual,
        "entries": rows,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    REPORT_MD.write_text("\n".join([
        "# Gate 7A Reference Audit",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Cited entries: {len(cited)}",
        f"- Missing cited entries: {len(missing_cited)}",
        f"- Duplicate entries: {len(duplicate_keys) + len(duplicate_titles)}",
        f"- Uncited bibliography entries: {len(uncited)}",
        f"- Incomplete metadata: {len(incomplete)}",
        f"- Fabricated or unverified metadata: {len(fabricated)}",
        f"- Manual verification required: {len(manual)}",
        "",
        "Metadata was checked against publisher proceedings, OpenReview, or DOI landing pages listed in the CSV.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "cited": len(cited),
        "missing": len(missing_cited),
        "duplicates": len(duplicate_keys) + len(duplicate_titles),
        "incomplete": len(incomplete),
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
