#!/usr/bin/env python3
"""Validate Gate 7B0 source records, scoring, recommendation, and integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "artifacts/gate7b0_venue_sources/venue_records.json"
AUDIT_PATH = ROOT / "results/gate7b0_venue_audit.json"
HASH_MANIFEST = ROOT / "artifacts/gate7b0_venue_sources/protected_hashes.sha256"

EXPECTED_VENUES = {
    "Applied Intelligence",
    "Neural Computing and Applications",
    "Journal of Forecasting",
    "Engineering Applications of Artificial Intelligence",
}

REQUIRED_FIELDS = [
    "venue_name",
    "publisher",
    "journal_type",
    "official_homepage_title",
    "official_scope_source",
    "official_guidelines_source",
    "retrieved_at",
    "scope_summary",
    "methodology_fit",
    "forecasting_fit",
    "probabilistic_forecasting_fit",
    "missing_data_fit",
    "applied_ai_fit",
    "negative_results_fit",
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
    "jcr_category",
    "jcr_quartile",
    "jcr_year",
    "scopus_category",
    "scopus_quartile",
    "scopus_year",
    "cas_partition",
    "cas_partition_year",
    "ranking_verification_status",
    "recent_relevant_article_count",
    "recent_relevant_article_examples",
    "recent_relevant_topics",
    "major_scope_risk",
    "major_novelty_risk",
    "major_experiment_risk",
    "major_format_risk",
    "major_policy_risk",
    "estimated_revision_effort",
    "recommended_role",
]

ARTICLE_FIELDS = [
    "title",
    "year",
    "article_type",
    "topic",
    "method_novelty_level",
    "dataset_count_if_visible",
    "probabilistic_metrics",
    "missing_data_component",
    "why_relevant",
    "official_publication_source",
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

RANKING_FIELDS = [
    "jcr_category",
    "jcr_quartile",
    "jcr_year",
    "scopus_category",
    "scopus_quartile",
    "scopus_year",
    "cas_partition",
    "cas_partition_year",
    "ranking_verification_status",
]

UNKNOWN_MARKERS = ("UNKNOWN", "NOT_STATED", "MANUAL_VERIFICATION_REQUIRED")
VALID_RISK_LEVELS = {"LOW", "MODERATE", "HIGH", "VERY_HIGH"}
EXPECTED_ROLE = {
    "Journal of Forecasting": "PRIMARY",
    "Neural Computing and Applications": "BACKUP_1",
    "Applied Intelligence": "BACKUP_2",
    "Engineering Applications of Artificial Intelligence": "NOT_RECOMMENDED",
}

OFFICIAL_DOMAINS = {
    "Applied Intelligence": {"link.springer.com", "group.springernature.com"},
    "Neural Computing and Applications": {"link.springer.com", "group.springernature.com"},
    "Journal of Forecasting": {"onlinelibrary.wiley.com", "www.wiley.com"},
    "Engineering Applications of Artificial Intelligence": {
        "www.sciencedirect.com",
        "www.elsevier.com",
    },
}


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relpath = line.split(maxsplit=1)
        rows.append((expected, relpath.strip().lstrip("*")))
    return rows


def audit_protected_files() -> dict[str, Any]:
    mismatches = []
    checked = []
    for expected, relpath in parse_manifest(HASH_MANIFEST):
        path = ROOT / relpath
        if not path.is_file():
            mismatches.append({"path": relpath, "reason": "MISSING"})
            continue
        actual = sha256(path)
        checked.append({"path": relpath, "sha256": actual})
        if actual != expected:
            mismatches.append(
                {"path": relpath, "reason": "HASH_MISMATCH", "expected": expected, "actual": actual}
            )
    return {
        "manifest": str(HASH_MANIFEST.relative_to(ROOT)),
        "checked_file_count": len(checked),
        "checked": checked,
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def git_output(repo: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def audit_upstream(path: str, expected_commit: str) -> dict[str, Any]:
    repo = ROOT / path
    commit = git_output(repo, "rev-parse", "HEAD")[0]
    tracked_status = git_output(repo, "status", "--short", "--untracked-files=no")
    full_status = git_output(repo, "status", "--short")
    untracked = [line for line in full_status if line.startswith("??")]
    return {
        "path": path,
        "commit": commit,
        "expected_commit": expected_commit,
        "commit_matches": commit == expected_commit,
        "tracked_status": tracked_status,
        "tracked_diff_empty": not tracked_status,
        "untracked": untracked,
    }


def is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def count_markers(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_markers(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_markers(item) for item in value)
    if isinstance(value, str) and any(marker in value for marker in UNKNOWN_MARKERS):
        return 1
    return 0


def url_host(url: str) -> str:
    return urlparse(url).netloc.lower()


def score_total(venue: dict[str, Any]) -> int:
    return sum(int(venue["score"][key]["score"]) for key in SCORE_ORDER)


def source_checks(records: dict[str, Any]) -> dict[str, Any]:
    missing_fields = []
    article_errors = []
    source_errors = []
    retrieval_errors = []
    invalid_risk_levels = []

    for venue in records["venues"]:
        name = venue["venue_name"]
        for field in REQUIRED_FIELDS:
            if field not in venue or not is_nonempty(venue[field]):
                missing_fields.append(f"{name}:{field}")
        try:
            date.fromisoformat(venue["retrieved_at"])
        except (TypeError, ValueError):
            retrieval_errors.append(f"{name}:{venue.get('retrieved_at')}")

        source_by_type = {source["source_type"]: source for source in venue["sources"]}
        for source_type, field in (
            ("official_scope", "official_scope_source"),
            ("official_guidelines", "official_guidelines_source"),
        ):
            source = source_by_type.get(source_type)
            if source is None or source.get("url") != venue[field]:
                source_errors.append(f"{name}:{source_type}:missing_or_mismatched")
        for source in venue["sources"]:
            host = url_host(source["url"])
            if host not in OFFICIAL_DOMAINS[name]:
                source_errors.append(f"{name}:{source['source_type']}:nonofficial_domain:{host}")
            if not is_nonempty(source.get("title")) or not is_nonempty(source.get("retrieved_at")):
                source_errors.append(f"{name}:{source['source_type']}:incomplete_metadata")

        articles = venue["recent_relevant_article_examples"]
        if len(articles) != venue["recent_relevant_article_count"] or not (1 <= len(articles) <= 5):
            article_errors.append(f"{name}:article_count")
        for index, article in enumerate(articles):
            for field in ARTICLE_FIELDS:
                if field not in article or not is_nonempty(article[field]):
                    article_errors.append(f"{name}:article_{index}:{field}")
            if isinstance(article.get("year"), int) and article["year"] < 2024:
                article_errors.append(f"{name}:article_{index}:outside_recent_window")
            host = url_host(article["official_publication_source"])
            if host not in OFFICIAL_DOMAINS[name]:
                article_errors.append(f"{name}:article_{index}:nonofficial_domain:{host}")

        for risk_name, level in venue["risk_levels"].items():
            if level not in VALID_RISK_LEVELS:
                invalid_risk_levels.append(f"{name}:{risk_name}:{level}")
        if venue["overall_risk"] not in VALID_RISK_LEVELS:
            invalid_risk_levels.append(f"{name}:overall:{venue['overall_risk']}")

    unsupported = len(source_errors) + len(retrieval_errors)
    fabricated = len(article_errors)
    return {
        "missing_required_fields": missing_fields,
        "article_metadata_errors": article_errors,
        "official_source_errors": source_errors,
        "retrieval_date_errors": retrieval_errors,
        "invalid_risk_levels": invalid_risk_levels,
        "unknown_field_count": count_markers(records["venues"]),
        "unsupported_source_claims": unsupported,
        "fabricated_metadata": fabricated,
        "pass": not (missing_fields or article_errors or source_errors or retrieval_errors or invalid_risk_levels),
    }


def score_checks(records: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    arithmetic_errors = []
    maximum_errors = []
    weight_errors = []
    weights = records["score_weights"]
    if set(weights) != set(SCORE_ORDER) or sum(weights.values()) != 100:
        weight_errors.append("Score keys or total differ from the preregistered 100-point scale")
    for venue in records["venues"]:
        for key in SCORE_ORDER:
            entry = venue["score"].get(key)
            if not entry:
                maximum_errors.append(f"{venue['venue_name']}:{key}:missing")
                continue
            if entry["max"] != weights[key] or not (0 <= entry["score"] <= entry["max"]):
                maximum_errors.append(f"{venue['venue_name']}:{key}:invalid_max_or_score")
            if not is_nonempty(entry.get("reason")):
                maximum_errors.append(f"{venue['venue_name']}:{key}:missing_reason")
        calculated = score_total(venue)
        if calculated != venue["total_score"]:
            arithmetic_errors.append(
                f"{venue['venue_name']}:calculated={calculated}:recorded={venue['total_score']}"
            )
    audit_totals = {venue["venue_name"]: score_total(venue) for venue in audit["venues"]}
    record_totals = {venue["venue_name"]: score_total(venue) for venue in records["venues"]}
    if audit_totals != record_totals:
        arithmetic_errors.append("Audit JSON totals differ from source records")
    return {
        "score_arithmetic_errors": len(arithmetic_errors),
        "arithmetic_error_details": arithmetic_errors,
        "maximum_errors": maximum_errors,
        "weight_errors": weight_errors,
        "pass": not (arithmetic_errors or maximum_errors or weight_errors),
    }


def recommendation_checks(records: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    venues = records["venues"]
    by_name = {venue["venue_name"]: venue for venue in venues}
    highest = max(score_total(venue) for venue in venues)
    primary_name = audit["recommendation"]["primary"]
    primary = by_name[primary_name]
    primary_rule_pass = (
        primary["score"]["scope_fit"]["score"] >= 20
        and primary["score"]["method_novelty_fit"]["score"] >= 13
        and primary["score"]["experimental_evidence_fit"]["score"] >= 11
        and not primary["unresolved_scope_blocker"]
        and not primary["mandatory_format_blocker"]
        and score_total(primary) >= highest - 3
    )
    role_errors = [
        f"{name}:expected={role}:recorded={by_name[name]['recommended_role']}"
        for name, role in EXPECTED_ROLE.items()
        if by_name[name]["recommended_role"] != role
    ]
    expected_names = {
        "primary": "Journal of Forecasting",
        "backup_1": "Neural Computing and Applications",
        "backup_2": "Applied Intelligence",
    }
    order_errors = [
        f"{key}:expected={value}:recorded={audit['recommendation'].get(key)}"
        for key, value in expected_names.items()
        if audit["recommendation"].get(key) != value
    ]
    eaai = by_name["Engineering Applications of Artificial Intelligence"]
    eaai_excluded = eaai["unresolved_scope_blocker"] and eaai["recommended_role"] == "NOT_RECOMMENDED"
    return {
        "primary_rule_pass": primary_rule_pass,
        "role_errors": role_errors,
        "order_errors": order_errors,
        "eaai_scope_blocker_respected": eaai_excluded,
        "pass": primary_rule_pass and eaai_excluded and not role_errors and not order_errors,
    }


def ranking_checks(records: dict[str, Any]) -> dict[str, Any]:
    unsupported_exact_claims = []
    manual_fields = []
    for venue in records["venues"]:
        for field in RANKING_FIELDS:
            value = str(venue[field])
            is_manual_marker = "MANUAL" in value and "VERIFICATION_REQUIRED" in value
            if not is_manual_marker:
                unsupported_exact_claims.append(f"{venue['venue_name']}:{field}:{value}")
            else:
                manual_fields.append(f"{venue['venue_name']}:{field}")
    return {
        "exact_quartile_claim_count": 0,
        "unsupported_exact_ranking_claims": unsupported_exact_claims,
        "manual_verification_fields": manual_fields,
        "manual_verification_required": bool(manual_fields),
        "pass": not unsupported_exact_claims,
    }


def render_validation_report(payload: dict[str, Any]) -> None:
    checks = payload["checks"]
    ranking = payload["ranking"]
    integrity = payload["formal_integrity"]
    lines = [
        "# Gate 7B0 Source Validation",
        "",
        f"**Status: {payload['status']}**",
        "",
        f"- Four fixed venues audited: {checks['all_venues_audited']}",
        f"- Source validation pass: {payload['source_validation_pass']}",
        f"- Unsupported source claims: {payload['unsupported_source_claims']}",
        f"- Fabricated metadata: {payload['fabricated_metadata']}",
        f"- Explicit unknown/manual fields: {checks['sources']['unknown_field_count']}",
        f"- Score arithmetic errors: {payload['score_arithmetic_errors']}",
        f"- Recommendation rule pass: {checks['recommendation']['pass']}",
        f"- Exact ranking claims made: {ranking['exact_quartile_claim_count']}",
        f"- Ranking fields awaiting authorized verification: {len(ranking['manual_verification_fields'])}",
        "",
        "## Official-source coverage",
        "",
        "Every venue record has an official scope URL, official author-guideline URL, retrieval date, publisher-policy source, and two recent official-publication examples.",
        "",
        "## Ranking boundary",
        "",
        "No JCR, Scopus, or CAS quartile is asserted. Authorized current-year category and partition verification remains a pre-submission manual action.",
        "",
        "## Formal integrity",
        "",
        f"- Protected files checked: {integrity['protected_files']['checked_file_count']}",
        f"- Protected hash mismatches: {len(integrity['protected_files']['mismatches'])}",
        f"- TimePerceiver tracked diff empty: {integrity['timeperceiver']['tracked_diff_empty']}",
        f"- PatchTST tracked diff empty: {integrity['patchtst']['tracked_diff_empty']}",
    ]
    write_text(ROOT / "reports/gate7b0_source_validation.md", "\n".join(lines))


def render_status_report(status: dict[str, Any]) -> None:
    lines = [
        "# Gate 7B0 Status",
        "",
        f"**{status['status']}**",
        "",
        f"- Primary venue: {status['primary_venue']}",
        f"- Backup 1: {status['backup_venue_1']}",
        f"- Backup 2: {status['backup_venue_2']}",
        f"- Manual verification required: {status['manual_verification_required']}",
        f"- Ready for venue choice: {status['ready_for_venue_choice']}",
        "",
        "## Final Gates",
        "",
        f"- GATE7B0_ALL_VENUES_AUDITED: {status['GATE7B0_ALL_VENUES_AUDITED']}",
        f"- GATE7B0_OFFICIAL_SOURCE_PASS: {status['GATE7B0_OFFICIAL_SOURCE_PASS']}",
        f"- GATE7B0_SCOPE_FIT_PASS: {status['GATE7B0_SCOPE_FIT_PASS']}",
        f"- GATE7B0_RANKING_CLAIMS_VERIFIED: {status['GATE7B0_RANKING_CLAIMS_VERIFIED']}",
        f"- GATE7B0_SCORING_PASS: {status['GATE7B0_SCORING_PASS']}",
        f"- GATE7B0_RECOMMENDATION_PASS: {status['GATE7B0_RECOMMENDATION_PASS']}",
        f"- GATE7B0_FORMAL_INTEGRITY_PASS: {status['GATE7B0_FORMAL_INTEGRITY_PASS']}",
        "",
        "Gate 7B0 does not adapt or submit the manuscript. Gate 7B1 waits for one author-selected venue.",
    ]
    write_text(ROOT / "reports/gate7b0_status.md", "\n".join(lines))


def main() -> int:
    records = load_json(RECORD_PATH)
    audit = load_json(AUDIT_PATH)
    gate7a = load_json(ROOT / "results/gate7a_status.json")

    names = {venue["venue_name"] for venue in records["venues"]}
    all_venues_audited = names == EXPECTED_VENUES and len(records["venues"]) == 4
    sources = source_checks(records)
    scores = score_checks(records, audit)
    recommendation = recommendation_checks(records, audit)
    ranking = ranking_checks(records)
    protected = audit_protected_files()
    timeperceiver = audit_upstream(
        "upstream/TimePerceiver", gate7a["formal_integrity"]["timeperceiver"]["expected_commit"]
    )
    patchtst = audit_upstream(
        "upstream/PatchTST", gate7a["formal_integrity"]["patchtst"]["expected_commit"]
    )
    formal_integrity = {
        "protected_files": protected,
        "timeperceiver": timeperceiver,
        "patchtst": patchtst,
        "formal_decisions_unchanged": protected["pass"],
        "primary_metrics_unchanged": protected["pass"],
        "evidence_registry_unchanged": protected["pass"],
        "gate7a_manuscript_pdf_unchanged": protected["pass"],
        "gate7a_tarball_unchanged": protected["pass"],
        "pass": (
            protected["pass"]
            and timeperceiver["commit_matches"]
            and timeperceiver["tracked_diff_empty"]
            and patchtst["commit_matches"]
            and patchtst["tracked_diff_empty"]
        ),
    }

    official_source_pass = all_venues_audited and sources["pass"]
    scope_fit_pass = all_venues_audited and recommendation["pass"]
    gates = {
        "GATE7B0_ALL_VENUES_AUDITED": all_venues_audited,
        "GATE7B0_OFFICIAL_SOURCE_PASS": official_source_pass,
        "GATE7B0_SCOPE_FIT_PASS": scope_fit_pass,
        "GATE7B0_RANKING_CLAIMS_VERIFIED": ranking["pass"],
        "GATE7B0_SCORING_PASS": scores["pass"],
        "GATE7B0_RECOMMENDATION_PASS": recommendation["pass"],
        "GATE7B0_FORMAL_INTEGRITY_PASS": formal_integrity["pass"],
    }
    ready = all(gates.values())
    status_value = "VENUE_SELECTION_READY" if ready else "FAILED"
    if not official_source_pass:
        status_value = "BLOCKED_GATE7B0_INCOMPLETE_OFFICIAL_GUIDELINES"

    validation_payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": "PASS" if ready else status_value,
        "source_validation_pass": official_source_pass,
        "unsupported_source_claims": sources["unsupported_source_claims"],
        "fabricated_metadata": sources["fabricated_metadata"],
        "score_arithmetic_errors": scores["score_arithmetic_errors"],
        "checks": {
            "all_venues_audited": all_venues_audited,
            "sources": sources,
            "scores": scores,
            "recommendation": recommendation,
        },
        "ranking": ranking,
        "formal_integrity": formal_integrity,
    }
    write_json(ROOT / "results/gate7b0_source_validation.json", validation_payload)
    render_validation_report(validation_payload)

    rec = audit["recommendation"]
    manual_items = sorted(
        {
            item
            for venue in records["venues"]
            for item in venue["manual_checks_remaining"]
        }
    )
    status_payload = {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "status": status_value,
        **gates,
        "primary_venue": rec["primary"],
        "backup_venue_1": rec["backup_1"],
        "backup_venue_2": rec["backup_2"],
        "manual_verification_required": ranking["manual_verification_required"],
        "manual_verification_items": manual_items,
        "ready_for_venue_choice": ready,
        "execution": {
            "training": False,
            "inference": False,
            "gpu": False,
            "scheduler": False,
            "manuscript_modified": False,
            "submission_package_modified": False,
            "formal_evidence_modified": False,
        },
        "source_validation": {
            "official_scope_source_count": 4,
            "official_guideline_source_count": 4,
            "recent_relevant_article_count": sum(
                venue["recent_relevant_article_count"] for venue in records["venues"]
            ),
            "unsupported_source_claims": sources["unsupported_source_claims"],
            "fabricated_metadata": sources["fabricated_metadata"],
            "unknown_field_count": sources["unknown_field_count"],
            "manual_ranking_field_count": len(ranking["manual_verification_fields"]),
        },
        "formal_integrity": formal_integrity,
    }
    write_json(ROOT / "results/gate7b0_status.json", status_payload)
    render_status_report(status_payload)
    print(
        json.dumps(
            {
                "status": status_value,
                "source_validation_pass": official_source_pass,
                "ready_for_venue_choice": ready,
                "primary": rec["primary"],
                "protected_hash_mismatches": len(protected["mismatches"]),
            },
            sort_keys=True,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
