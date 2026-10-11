#!/usr/bin/env python3
"""Audit Gate 7A prose for unsupported claims and terminology drift."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript"
RESULT = ROOT / "results" / "gate7a_language_audit.json"
REPORT = ROOT / "reports" / "gate7a_language_audit.md"

PROHIBITED = {
    "state-of-the-art": r"\bstate[- ]of[- ]the[- ]art\b|\bSOTA\b",
    "statistical-significance claim": r"\bstatistically significant\b|\bsignificant(?:ly)? (?:improvement|gain|outperformance)\b",
    "universal claim": r"\b(?:is|are|proves?|establishes?) universally?\b|\buniversal solution\b",
    "backbone-independent claim": r"\bbackbone[- ]independent\b",
    "joint-calibration claim": r"\bjoint(?:ly)? calibrated\b|\bjoint calibration (?:is|was|holds)\b",
    "guarantee claim": r"\bguarantee[sd]? (?:uncertainty|coverage|performance|improvement)\b",
    "unsupported causal claim": r"\b(?:proves?|demonstrates?) that .{0,80}\bcauses?\b",
}
TERMINOLOGY_CONFLICTS = {
    "fixed-t": r"\bfixed[- ]t(?:8)?\b",
    "fixed student t": r"\bfixed student t\b",
    "t head": r"(?<!student-)\bt head\b",
}
KNOWN_ABBREVIATIONS = {
    "BRITS", "CRPS", "CSDI", "CUDA", "MAE", "MC", "MPIW", "MSE", "NLL", "PICP", "RMSE", "SECOND",
}


def strip_latex(text: str) -> str:
    text = "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())
    text = re.sub(r"\\(?:cite\w*|ref|eqref|label|url|texttt)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("$", " ").replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", text)


def line_hits(pattern: str, files: dict[str, str]) -> list[dict[str, object]]:
    hits = []
    regex = re.compile(pattern, re.IGNORECASE)
    for name, text in files.items():
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in regex.finditer(line):
                hits.append({"file": name, "line": line_number, "text": match.group(0)})
    return hits


def main() -> int:
    source_paths = [MANUSCRIPT / "abstract.tex", MANUSCRIPT / "anonymous_metadata.tex"]
    source_paths.extend(sorted((MANUSCRIPT / "sections").glob("*.tex")))
    source_paths.extend(sorted((MANUSCRIPT / "appendix").glob("*.tex")))
    files = {str(path.relative_to(MANUSCRIPT)): path.read_text(encoding="utf-8") for path in source_paths}
    prose = strip_latex("\n".join(files.values()))

    prohibited_hits = {name: line_hits(pattern, files) for name, pattern in PROHIBITED.items()}
    prohibited_hits = {name: hits for name, hits in prohibited_hits.items() if hits}
    terminology_hits = {name: line_hits(pattern, files) for name, pattern in TERMINOLOGY_CONFLICTS.items()}
    terminology_hits = {name: hits for name, hits in terminology_hits.items() if hits}

    seen_abbreviations = sorted(set(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", prose)))
    undefined = sorted(abbreviation for abbreviation in seen_abbreviations if abbreviation.rstrip("-") not in KNOWN_ABBREVIATIONS)

    sentences = re.split(r"(?<=[.!?])\s+", prose)
    overlong = []
    for sentence in sentences:
        words = re.findall(r"\b[\w'-]+\b", sentence)
        if len(words) > 45:
            overlong.append({"word_count": len(words), "text": sentence[:240]})

    passive_pattern = re.compile(r"\b(?:is|are|was|were|be|been)\s+(?:[A-Za-z]+ed|shown|known|given|built|fit|set|used)\b", re.IGNORECASE)
    passive_count = len(passive_pattern.findall(prose))
    prohibited_count = sum(len(hits) for hits in prohibited_hits.values())
    terminology_count = sum(len(hits) for hits in terminology_hits.values())
    passed = prohibited_count == 0 and terminology_count == 0 and len(undefined) == 0
    payload = {
        "status": "PASS" if passed else "FAILED",
        "GATE7A_LANGUAGE_AUDIT_PASS": passed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_terms": [
            "missing historical observations",
            "fixed missing-history evaluation",
            "fixed Student-t head",
            "fixed df=8",
            "probability-head training seed",
            "point noninferiority",
            "probabilistic-score improvement",
            "point-prediction tradeoff",
        ],
        "prohibited_unsupported_claim_count": prohibited_count,
        "terminology_conflict_count": terminology_count,
        "undefined_abbreviation_count": len(undefined),
        "overlong_sentence_count": len(overlong),
        "passive_voice_heuristic_count": passive_count,
        "prohibited_hits": prohibited_hits,
        "terminology_conflicts": terminology_hits,
        "seen_abbreviations": seen_abbreviations,
        "undefined_abbreviations": undefined,
        "overlong_sentences": overlong,
        "passive_voice_note": "Heuristic count is reported for editorial review and is not a failure criterion.",
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Gate 7A Language Audit",
        "",
        f"**Status:** {payload['status']}",
        "",
        f"- Prohibited unsupported wording: {prohibited_count}",
        f"- Terminology conflicts: {terminology_count}",
        f"- Undefined abbreviations: {len(undefined)}",
        f"- Overlong sentences (>45 words): {len(overlong)}",
        f"- Passive-voice heuristic matches: {passive_count}",
        "",
        "Passive-voice and sentence-length counts are editorial diagnostics, not automatic evidence failures.",
    ]
    if overlong:
        lines.extend(["", "## Overlong Sentences", ""])
        lines.extend(f"- {item['word_count']} words: {item['text']}" for item in overlong)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "prohibited": prohibited_count,
        "terminology": terminology_count,
        "undefined": len(undefined),
        "overlong": len(overlong),
    }, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
