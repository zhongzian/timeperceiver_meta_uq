#!/usr/bin/env python3
"""Generate the Gate 7B0 venue-fit audit from reviewed official-source records."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "artifacts/gate7b0_venue_sources/venue_records.json"
RESULT_PATH = ROOT / "results/gate7b0_venue_audit.json"

EXPECTED_VENUES = [
    "Applied Intelligence",
    "Neural Computing and Applications",
    "Journal of Forecasting",
    "Engineering Applications of Artificial Intelligence",
]

SCORE_ORDER = [
    "scope_fit",
    "method_novelty_fit",
    "experimental_evidence_fit",
    "forecasting_community_fit",
    "manuscript_format_fit",
    "negative_reproducibility_fit",
    "submission_policy_compatibility",
    "risk_adjusted_acceptance_fit",
]

SCORE_LABELS = {
    "scope_fit": "Scope",
    "method_novelty_fit": "Novelty",
    "experimental_evidence_fit": "Experiments",
    "forecasting_community_fit": "Forecast fit",
    "manuscript_format_fit": "Format",
    "negative_reproducibility_fit": "Negative/repro",
    "submission_policy_compatibility": "Policy",
    "risk_adjusted_acceptance_fit": "Risk-adjusted",
}

SUBMISSION_FIELDS = [
    "article_type",
    "manuscript_length_limit",
    "word_limit",
    "page_limit",
    "abstract_limit",
    "reference_limit",
    "figure_limit",
    "table_limit",
    "supplementary_material_policy",
    "initial_submission_format",
    "latex_template",
    "reference_style",
    "double_blind_policy",
    "anonymous_submission_required",
    "author_details_required_at_initial_submission",
    "data_availability_statement",
    "code_availability_statement",
    "conflict_of_interest_statement",
    "funding_statement",
    "author_contribution_statement",
    "generative_ai_disclosure",
    "ethics_statement",
    "highlights_required",
    "graphical_abstract_required",
    "cover_letter_required",
    "open_access_model",
    "mandatory_apc",
    "optional_apc",
    "submission_fee",
    "fee_information_verified",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def md(value: Any) -> str:
    if isinstance(value, bool):
        text = "YES" if value else "NO"
    elif isinstance(value, list):
        text = "; ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def score_value(venue: dict[str, Any], key: str) -> int:
    return int(venue["score"][key]["score"])


def score_total(venue: dict[str, Any]) -> int:
    return sum(score_value(venue, key) for key in SCORE_ORDER)


def primary_eligible(venue: dict[str, Any], highest: int) -> bool:
    return (
        score_value(venue, "scope_fit") >= 20
        and score_value(venue, "method_novelty_fit") >= 13
        and score_value(venue, "experimental_evidence_fit") >= 11
        and not venue["unresolved_scope_blocker"]
        and not venue["mandatory_format_blocker"]
        and score_total(venue) >= highest - 3
    )


def recommendation(venues: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(venues, key=lambda venue: (-score_total(venue), venue["venue_name"]))
    highest = score_total(ranked[0])
    eligible = [venue for venue in ranked if primary_eligible(venue, highest)]
    if not eligible:
        raise ValueError("No venue satisfies the preregistered primary recommendation rule")
    primary = eligible[0]
    backup_candidates = [
        venue
        for venue in ranked
        if venue["venue_name"] != primary["venue_name"]
        and not venue["unresolved_scope_blocker"]
        and not venue["mandatory_format_blocker"]
    ]
    backups = backup_candidates[:2]
    excluded = [
        venue for venue in ranked if venue["venue_name"] not in {primary["venue_name"], *(item["venue_name"] for item in backups)}
    ]
    return {
        "primary": primary["venue_name"],
        "backup_1": backups[0]["venue_name"],
        "backup_2": backups[1]["venue_name"],
        "not_recommended": [venue["venue_name"] for venue in excluded],
        "highest_total": highest,
        "eligible_primary_venues": [venue["venue_name"] for venue in eligible],
        "rule": "scope>=20, novelty>=13, experiments>=11, no scope/format blocker, and within three points of the highest total",
    }


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_comparison(venues: list[dict[str, Any]]) -> None:
    csv_fields = [
        "venue",
        *SCORE_ORDER,
        "total",
        "overall_risk",
        "unresolved_scope_blocker",
        "estimated_revision_effort",
        "recommended_role",
    ]
    rows = []
    for venue in venues:
        row: dict[str, Any] = {"venue": venue["venue_name"]}
        row.update({key: score_value(venue, key) for key in SCORE_ORDER})
        row.update(
            {
                "total": score_total(venue),
                "overall_risk": venue["overall_risk"],
                "unresolved_scope_blocker": venue["unresolved_scope_blocker"],
                "estimated_revision_effort": venue["estimated_revision_effort"],
                "recommended_role": venue["recommended_role"],
            }
        )
        rows.append(row)
    write_csv(ROOT / "reports/gate7b0_venue_comparison.csv", csv_fields, rows)

    headers = ["Venue", *[SCORE_LABELS[key] for key in SCORE_ORDER], "Total", "Risk", "Role"]
    lines = [
        "# Gate 7B0 Venue Comparison",
        "",
        "Scores rank documented suitability; they are not acceptance probabilities.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * 9 + ["---", "---"]) + " |",
    ]
    for venue in venues:
        values = [
            venue["venue_name"],
            *[score_value(venue, key) for key in SCORE_ORDER],
            score_total(venue),
            venue["overall_risk"],
            venue["recommended_role"],
        ]
        lines.append("| " + " | ".join(md(value) for value in values) + " |")
    lines.extend(["", "## Score reasons", ""])
    for venue in sorted(venues, key=lambda item: -score_total(item)):
        lines.extend([f"### {venue['venue_name']}", ""])
        for key in SCORE_ORDER:
            entry = venue["score"][key]
            lines.append(f"- {SCORE_LABELS[key]} ({entry['score']}/{entry['max']}): {entry['reason']}")
        lines.append("")
    write_text(ROOT / "reports/gate7b0_venue_comparison.md", "\n".join(lines))


def render_scope_fit(venues: list[dict[str, Any]]) -> None:
    lines = [
        "# Gate 7B0 Scope-Fit Audit",
        "",
        "The audit uses the venue-neutral manuscript facts frozen at Gate 7A. No manuscript text or formal evidence was changed.",
        "",
    ]
    for venue in venues:
        lines.extend(
            [
                f"## {venue['venue_name']}",
                "",
                f"- Official scope: {venue['official_scope_source']}",
                f"- Retrieved: {venue['retrieved_at']}",
                f"- Scope summary: {venue['scope_summary']}",
                f"- Methodology fit: {venue['methodology_fit']}",
                f"- Forecasting fit: {venue['forecasting_fit']}",
                f"- Probabilistic forecasting fit: {venue['probabilistic_forecasting_fit']}",
                f"- Missing-data fit: {venue['missing_data_fit']}",
                f"- Applied-AI fit: {venue['applied_ai_fit']}",
                f"- Negative-results fit: {venue['negative_results_fit']}",
                f"- Unresolved scope blocker: {venue['unresolved_scope_blocker']}",
                "",
                "Recent official-publication ecology:",
                "",
            ]
        )
        for article in venue["recent_relevant_article_examples"]:
            lines.append(
                f"- {article['year']}, [{article['title']}]({article['official_publication_source']}): "
                f"{article['why_relevant']}"
            )
        lines.append("")
    write_text(ROOT / "reports/gate7b0_scope_fit.md", "\n".join(lines))


def render_submission_requirements(venues: list[dict[str, Any]]) -> None:
    lines = [
        "# Gate 7B0 Submission Requirements",
        "",
        "`NOT_STATED` and `MANUAL_VERIFICATION_REQUIRED` are deliberate uncertainty markers, not inferred requirements.",
        "",
    ]
    for venue in venues:
        lines.extend(
            [
                f"## {venue['venue_name']}",
                "",
                f"Official guidelines: {venue['official_guidelines_source']} (retrieved {venue['retrieved_at']})",
                "",
                "| Requirement | Current official-record finding |",
                "| --- | --- |",
            ]
        )
        for field in SUBMISSION_FIELDS:
            lines.append(f"| {field.replace('_', ' ')} | {md(venue[field])} |")
        lines.extend(["", "Manual checks:", ""])
        lines.extend(f"- {item}" for item in venue["manual_checks_remaining"])
        lines.append("")
    write_text(ROOT / "reports/gate7b0_submission_requirements.md", "\n".join(lines))


def render_risks(venues: list[dict[str, Any]]) -> None:
    lines = [
        "# Gate 7B0 Submission-Risk Assessment",
        "",
        "No acceptance rate or review-duration estimate is used. Risk levels are qualitative consequences of documented scope, policy, and recent-paper fit.",
        "",
    ]
    for venue in venues:
        lines.extend(
            [
                f"## {venue['venue_name']} - {venue['overall_risk']}",
                "",
                "| Risk dimension | Level | Evidence-based interpretation |",
                "| --- | --- | --- |",
                f"| Desk-reject scope | {venue['risk_levels']['desk_reject_scope_risk']} | {md(venue['major_scope_risk'])} |",
                f"| Novelty | {venue['risk_levels']['novelty_rejection_risk']} | {md(venue['major_novelty_risk'])} |",
                f"| Experimental sufficiency | {venue['risk_levels']['experimental_sufficiency_risk']} | {md(venue['major_experiment_risk'])} |",
                f"| Format adaptation | {venue['risk_levels']['format_adaptation_risk']} | {md(venue['major_format_risk'])} |",
                f"| Policy compliance | {venue['risk_levels']['policy_compliance_risk']} | {md(venue['major_policy_risk'])} |",
                "",
                f"Estimated revision effort: **{venue['estimated_revision_effort']}**",
                "",
            ]
        )
    write_text(ROOT / "reports/gate7b0_risk_assessment.md", "\n".join(lines))


def render_recommendation(venues: list[dict[str, Any]], rec: dict[str, Any]) -> None:
    by_name = {venue["venue_name"]: venue for venue in venues}
    primary = by_name[rec["primary"]]
    backup_1 = by_name[rec["backup_1"]]
    backup_2 = by_name[rec["backup_2"]]
    lines = [
        "# Gate 7B0 Recommendation",
        "",
        "## Recommended primary venue",
        "",
        f"- Venue: **{primary['venue_name']}**",
        f"- Total score: **{score_total(primary)}/100**",
        f"- Risk level: **{primary['overall_risk']}**",
        f"- Why it fits: {primary['scope_summary']}",
        f"- Main weaknesses: {primary['major_novelty_risk']} {primary['major_experiment_risk']}",
        "- Required manuscript changes:",
        *[f"  - {item}" for item in primary["required_manuscript_changes"]],
        "- Manual checks remaining:",
        *[f"  - {item}" for item in primary["manual_checks_remaining"]],
        "",
        "## Backup venues",
        "",
        f"### Backup 1: {backup_1['venue_name']}",
        "",
        "Choose when: the author prefers explicit neural-computing and benchmark scope over a forecasting-specialist readership.",
        f"Main advantage: {backup_1['scope_summary']}",
        f"Main risk: {backup_1['major_novelty_risk']}",
        f"Adaptation effort: {backup_1['estimated_revision_effort']}",
        "",
        f"### Backup 2: {backup_2['venue_name']}",
        "",
        "Choose when: the author wants a broad applied-intelligence audience and can foreground the protocol as an intelligent-systems contribution.",
        f"Main advantage: {backup_2['probabilistic_forecasting_fit']}",
        f"Main risk: {backup_2['major_novelty_risk']}",
        f"Adaptation effort: {backup_2['estimated_revision_effort']}",
        "",
        "## Venues not recommended",
        "",
    ]
    for name in rec["not_recommended"]:
        venue = by_name[name]
        lines.append(f"- **{name}**: {venue['major_scope_risk']}")
    lines.extend(
        [
            "",
            "## Submission cascade",
            "",
            f"{rec['primary']} -> {rec['backup_1']} -> {rec['backup_2']}",
            "",
            "Submit sequentially, never simultaneously.",
            "",
            "## Decision boundary",
            "",
            "Gate 7B0 recommends a venue. It does not submit or adapt the manuscript.",
        ]
    )
    write_text(ROOT / "reports/gate7b0_recommendation.md", "\n".join(lines))


def render_decision_brief(venues: list[dict[str, Any]], rec: dict[str, Any]) -> None:
    lines = [
        "# Gate 7B0 Venue Decision Brief",
        "",
        "| Venue | Scope fit | Novelty fit | Experiment fit | Format effort | Risk | Total | Recommendation |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |",
    ]
    for venue in venues:
        lines.append(
            f"| {venue['venue_name']} | {score_value(venue, 'scope_fit')}/25 | "
            f"{score_value(venue, 'method_novelty_fit')}/20 | "
            f"{score_value(venue, 'experimental_evidence_fit')}/15 | "
            f"{venue['estimated_revision_effort']} | {venue['overall_risk']} | "
            f"{score_total(venue)} | {venue['recommended_role']} |"
        )
    lines.extend(
        [
            "",
            f"**Primary recommendation:** {rec['primary']}",
            "",
            "Best direct fit for the manuscript's probabilistic-forecasting question and heavy-tail evidence. Adaptation should strengthen forecasting and statistical framing, not invent theoretical novelty.",
            "",
            f"**Backup 1:** {rec['backup_1']}",
            "",
            "Strong explicit fit for intelligent forecasting, neural computing, and benchmark evidence.",
            "",
            f"**Backup 2:** {rec['backup_2']}",
            "",
            "Broad applied-AI fit, with a higher novelty-mismatch risk than the first two choices.",
            "",
            "**Not recommended:** Engineering Applications of Artificial Intelligence, because the present manuscript lacks the journal's required real-world engineering application.",
            "",
            "## Manual verification items",
            "",
            "- Verify current-year JCR categories/quartiles from JCR.",
            "- Verify current-year Scopus categories/quartiles from Scopus and do not substitute a favorable category.",
            "- Verify current CAS large/small category and partition from an authorized source.",
            "- Confirm current optional APC, submission-fee status, and portal article type for the selected venue.",
            "- Confirm the selected venue's final title-page, data/code statement, and AI-disclosure workflow.",
            "",
            "Gate 7B0 only recommends a venue. Gate 7B1 starts after the author selects exactly one venue.",
        ]
    )
    write_text(ROOT / "paper/venue_selection/decision_brief.md", "\n".join(lines))


def render_requirements_matrix(venues: list[dict[str, Any]]) -> None:
    fields = ["venue_name", "official_guidelines_source", *SUBMISSION_FIELDS]
    rows = [{field: venue[field] for field in fields} for venue in venues]
    write_csv(ROOT / "paper/venue_selection/venue_requirements_matrix.csv", fields, rows)

    compact_fields = [
        "article_type",
        "abstract_limit",
        "manuscript_length_limit",
        "initial_submission_format",
        "double_blind_policy",
        "data_availability_statement",
        "generative_ai_disclosure",
        "open_access_model",
    ]
    lines = [
        "# Venue Requirements Matrix",
        "",
        "Unknown and changing items remain explicitly marked for manual verification.",
        "",
        "| Venue | Article type | Abstract | Length | Initial format | Review | Data statement | AI disclosure | OA model |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for venue in venues:
        values = [venue["venue_name"], *[venue[field] for field in compact_fields]]
        lines.append("| " + " | ".join(md(value) for value in values) + " |")
    lines.extend(["", "The complete field matrix is stored in `venue_requirements_matrix.csv`."])
    write_text(ROOT / "paper/venue_selection/venue_requirements_matrix.md", "\n".join(lines))


def render_source_artifacts(venues: list[dict[str, Any]]) -> None:
    inventory_rows = []
    for venue in venues:
        lines = [
            f"# Official Source Record: {venue['venue_name']}",
            "",
            f"Retrieved: {venue['retrieved_at']}",
            "",
            "This is a concise research record, not a copy of the web pages.",
            "",
        ]
        for source in venue["sources"]:
            lines.extend(
                [
                    f"- [{source['title']}]({source['url']})",
                    f"  - Type: {source['source_type']}",
                    f"  - Retrieved: {source['retrieved_at']}",
                ]
            )
            inventory_rows.append(
                {
                    "venue": venue["venue_name"],
                    "source_type": source["source_type"],
                    "title": source["title"],
                    "url": source["url"],
                    "retrieved_at": source["retrieved_at"],
                }
            )
        write_text(ROOT / f"artifacts/gate7b0_venue_sources/{venue['slug']}.md", "\n".join(lines))
    write_csv(
        ROOT / "artifacts/gate7b0_venue_sources/source_inventory.csv",
        ["venue", "source_type", "title", "url", "retrieved_at"],
        inventory_rows,
    )


def main() -> int:
    records = load_json(RECORD_PATH)
    venues = records["venues"]
    names = [venue["venue_name"] for venue in venues]
    if names != EXPECTED_VENUES:
        raise ValueError(f"Expected venues in fixed order {EXPECTED_VENUES}, found {names}")
    expected_max = sum(records["score_weights"].values())
    if expected_max != 100:
        raise ValueError(f"Score weights sum to {expected_max}, expected 100")
    for venue in venues:
        calculated = score_total(venue)
        if calculated != venue["total_score"]:
            raise ValueError(f"{venue['venue_name']} score is {calculated}, recorded {venue['total_score']}")

    rec = recommendation(venues)
    render_comparison(venues)
    render_scope_fit(venues)
    render_submission_requirements(venues)
    render_risks(venues)
    render_recommendation(venues, rec)
    render_decision_brief(venues, rec)
    render_requirements_matrix(venues)
    render_source_artifacts(venues)

    payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "AUDIT_COMPLETE",
        "retrieved_at": records["retrieved_at"],
        "execution": {
            "training": False,
            "inference": False,
            "gpu": False,
            "scheduler": False,
            "manuscript_modified": False,
            "submission_package_modified": False,
            "formal_evidence_modified": False,
        },
        "manuscript_baseline": records["manuscript_baseline"],
        "score_weights": records["score_weights"],
        "ranking_policy": records["ranking_policy"],
        "recommendation": rec,
        "venues": venues,
        "artifacts": [
            "reports/gate7b0_venue_comparison.csv",
            "reports/gate7b0_venue_comparison.md",
            "reports/gate7b0_scope_fit.md",
            "reports/gate7b0_submission_requirements.md",
            "reports/gate7b0_risk_assessment.md",
            "reports/gate7b0_recommendation.md",
            "paper/venue_selection/decision_brief.md",
            "paper/venue_selection/venue_requirements_matrix.csv",
            "paper/venue_selection/venue_requirements_matrix.md",
            "artifacts/gate7b0_venue_sources/source_inventory.csv",
        ],
    }
    write_json(RESULT_PATH, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "venues": len(venues),
                "primary": rec["primary"],
                "backup_1": rec["backup_1"],
                "backup_2": rec["backup_2"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
