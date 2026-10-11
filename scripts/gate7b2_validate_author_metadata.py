#!/usr/bin/env python3
"""Validate Gate 7B2 author metadata without exposing private field values."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
AUTHOR_METADATA = VENUE / "author_input/author_metadata.yaml"
PREVIEW = VENUE / "submission_final_preview"
FREEZE_MANIFEST = ROOT / "artifacts/gate7b2_pre_metadata/source_manifest.json"
VALIDATION_JSON = ROOT / "results/gate7b2_author_metadata_validation.json"
VALIDATION_MD = ROOT / "reports/gate7b2_author_metadata_validation.md"
STATUS_JSON = ROOT / "results/gate7b2_status.json"
STATUS_MD = ROOT / "reports/gate7b2_status.md"

INPUT_PLACEHOLDER = "AUTHOR_INPUT_REQUIRED"
CONFIRMATION_PLACEHOLDER = "AUTHOR_CONFIRMATION_REQUIRED"
PORTAL_PLACEHOLDER = "MANUAL_PORTAL_VERIFICATION_REQUIRED"
PLACEHOLDERS = {INPUT_PLACEHOLDER, CONFIRMATION_PLACEHOLDER, PORTAL_PLACEHOLDER}

EXPECTED_GRANTS = ["62272189", "62372206", "62402193", "2024AFB421", "XJ2026001501"]
EXPECTED_FUNDERS = {
    "National Natural Science Foundation of China": ["62272189", "62372206", "62402193"],
    "Hubei Provincial Natural Science Foundation of China": ["2024AFB421"],
    "Fundamental Research Funds for the Central Universities, Central China Normal University": [
        "XJ2026001501"
    ],
}
EXPECTED_FUNDING_STATEMENT = (
    "This work was supported by the National Natural Science Foundation of China "
    "(Nos. 62272189, 62372206, and 62402193), the Hubei Provincial Natural "
    "Science Foundation of China (No. 2024AFB421), and the Fundamental Research "
    "Funds for the Central Universities, Central China Normal University "
    "(No. XJ2026001501)."
)
EXPECTED_AI_STATEMENT = (
    "During the preparation of this manuscript, the authors used OpenAI ChatGPT "
    "and Codex for language editing, manuscript organization, and code-development "
    "assistance. All generated content and code changes were reviewed and verified "
    "by the authors, who remain fully responsible for the scientific claims, analyses, "
    "interpretation, and final manuscript. No AI system is listed as an author."
)
REQUIRED_TOP_LEVEL = {
    "manuscript",
    "authors",
    "affiliations",
    "corresponding_author",
    "funding",
    "acknowledgements",
    "conflict_of_interest",
    "author_contributions",
    "data_availability",
    "code_availability",
    "ethics",
    "generative_ai",
    "submission_confirmations",
    "reviewers",
    "portal",
}
EXPECTED_CREDIT_ROLES = {
    "conceptualization",
    "methodology",
    "software",
    "validation",
    "formal_analysis",
    "investigation",
    "resources",
    "data_curation",
    "writing_original_draft",
    "writing_review_editing",
    "visualization",
    "supervision",
    "project_administration",
    "funding_acquisition",
}


def import_yaml():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        fallback = Path("/mnt/data/zza/anaconda3/lib/python3.11/site-packages")
        if not fallback.is_dir():
            raise
        sys.path.insert(0, str(fallback))
        import yaml  # type: ignore
    return yaml


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: object) -> str:
    return " ".join(str(value).split())


def placeholder_counts(value: object) -> Counter[str]:
    counts: Counter[str] = Counter()
    if isinstance(value, dict):
        for item in value.values():
            counts.update(placeholder_counts(item))
    elif isinstance(value, list):
        for item in value:
            counts.update(placeholder_counts(item))
    elif isinstance(value, str) and value in PLACEHOLDERS:
        counts[value] += 1
    return counts


def is_resolved(value: object) -> bool:
    return value not in (None, "", *PLACEHOLDERS)


def valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", value))


def valid_orcid(value: str) -> bool:
    compact = value.removeprefix("https://orcid.org/").replace("-", "").upper()
    if not re.fullmatch(r"\d{15}[\dX]", compact):
        return False
    total = 0
    for digit in compact[:15]:
        total = (total + int(digit)) * 2
    result = (12 - total % 11) % 11
    check = "X" if result == 10 else str(result)
    return compact[-1] == check


def duplicate_values(values: list[object]) -> list[object]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def bool_true(mapping: dict[str, Any], key: str) -> bool:
    return mapping.get(key) is True


def load_freeze_and_verify() -> tuple[bool, bool, list[str]]:
    failures: list[str] = []
    if not FREEZE_MANIFEST.is_file():
        return False, False, ["missing freeze manifest"]
    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    group_results: dict[str, bool] = {}
    for group in ("required_sources", "formal_protected_assets"):
        entries = manifest.get(group)
        passed = isinstance(entries, list) and bool(entries)
        if not isinstance(entries, list):
            failures.append(f"malformed {group}")
            group_results[group] = False
            continue
        for entry in entries:
            relative = str(entry.get("path", "")) if isinstance(entry, dict) else ""
            expected = str(entry.get("sha256", "")) if isinstance(entry, dict) else ""
            path = ROOT / relative
            if not relative or not path.is_file():
                failures.append(f"missing protected asset: {relative or '[invalid path]'}")
                passed = False
            elif sha256(path) != expected:
                failures.append(f"protected asset changed: {relative}")
                passed = False
        group_results[group] = passed
    return group_results.get("required_sources", False), group_results.get(
        "formal_protected_assets", False
    ), failures


def package_privacy_check() -> tuple[bool, list[str]]:
    failures: list[str] = []
    package = VENUE / "journal_of_forecasting_submission_package.tar.gz"
    if not package.is_file():
        return False, ["Gate 7B1 package is missing"]
    forbidden_parts = ("author_input", "submission_final_preview", "gate7b2")
    try:
        with tarfile.open(package, "r:gz") as archive:
            leaked = [name for name in archive.getnames() if any(part in name.lower() for part in forbidden_parts)]
        if leaked:
            failures.append("Gate 7B1 package contains Gate 7B2/private paths")
    except tarfile.TarError:
        failures.append("Gate 7B1 package cannot be read")
    unexpected_packages = list((VENUE / "author_input").glob("*.tar.gz")) + list(
        PREVIEW.glob("*.tar.gz")
    )
    if unexpected_packages:
        failures.append("an author-complete or preview package was created")
    return not failures, failures


def pdf_is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:5] == b"%PDF-"


def main() -> int:
    yaml = import_yaml()
    parse_error = ""
    try:
        metadata = yaml.safe_load(AUTHOR_METADATA.read_text(encoding="utf-8"))
    except Exception as exc:  # PyYAML exposes several parser exception classes.
        metadata = {}
        parse_error = f"{type(exc).__name__}: {exc}"
    yaml_parse_pass = isinstance(metadata, dict) and not parse_error
    if not isinstance(metadata, dict):
        metadata = {}

    schema_errors: list[str] = []
    missing_top_level = sorted(REQUIRED_TOP_LEVEL - set(metadata))
    if missing_top_level:
        schema_errors.append("missing top-level fields: " + ", ".join(missing_top_level))

    authors_raw = metadata.get("authors", [])
    authors = [item for item in authors_raw if isinstance(item, dict)] if isinstance(authors_raw, list) else []
    if len(authors) != 2:
        schema_errors.append("authors must contain exactly two supplied candidates")
    author_ids = [item.get("author_id") for item in authors]
    duplicate_author_ids = duplicate_values(author_ids)
    author_orders = [item.get("author_order") for item in authors]
    duplicate_author_orders = duplicate_values(author_orders)
    if duplicate_author_ids:
        schema_errors.append("duplicate author IDs")
    if duplicate_author_orders:
        schema_errors.append("duplicate author order")
    author_map = {str(item.get("author_id")): item for item in authors if item.get("author_id")}

    affiliations_raw = metadata.get("affiliations", [])
    affiliations = (
        [item for item in affiliations_raw if isinstance(item, dict)]
        if isinstance(affiliations_raw, list)
        else []
    )
    affiliation_ids = [item.get("affiliation_id") for item in affiliations]
    duplicate_affiliation_ids = duplicate_values(affiliation_ids)
    if duplicate_affiliation_ids:
        schema_errors.append("duplicate affiliation IDs")
    affiliation_map = {
        str(item.get("affiliation_id")): item for item in affiliations if item.get("affiliation_id")
    }
    invalid_affiliation_references: list[dict[str, str]] = []
    for author in authors:
        refs = author.get("affiliations", [])
        if not isinstance(refs, list):
            schema_errors.append(f"{author.get('author_id', 'unknown')} affiliations must be a list")
            continue
        for ref in refs:
            if str(ref) not in affiliation_map:
                invalid_affiliation_references.append(
                    {"author_id": str(author.get("author_id")), "affiliation_id": str(ref)}
                )

    a1 = author_map.get("A1", {})
    a2 = author_map.get("A2", {})
    a1_identity_supplied = (
        a1.get("given_name") == "Zian"
        and a1.get("family_name") == "Zhong"
        and a1.get("display_name") == "Zian Zhong"
        and a1.get("native_name") == "钟子安"
    )
    a2_identity_supplied = (
        a2.get("given_name") == "Jianqun"
        and a2.get("family_name") == "Cui"
        and a2.get("display_name") == "Jianqun Cui"
    )
    confirmations = metadata.get("submission_confirmations", {})
    if not isinstance(confirmations, dict):
        confirmations = {}
        schema_errors.append("submission_confirmations must be a mapping")
    a2_candidate_safely_unconfirmed = (
        a2.get("authorship_inclusion_confirmed") is False
        and confirmations.get("jianqun_cui_authorship_confirmed") is False
    )

    missing_email_ids: list[str] = []
    invalid_email_ids: list[str] = []
    valid_email_ids: list[str] = []
    missing_orcid_ids: list[str] = []
    invalid_orcid_ids: list[str] = []
    valid_orcid_ids: list[str] = []
    for author in authors:
        author_id = str(author.get("author_id", "unknown"))
        email = author.get("email")
        if not is_resolved(email):
            missing_email_ids.append(author_id)
        elif not isinstance(email, str) or not valid_email(email):
            invalid_email_ids.append(author_id)
        else:
            valid_email_ids.append(author_id)
        orcid = author.get("orcid")
        if not is_resolved(orcid):
            missing_orcid_ids.append(author_id)
        elif not isinstance(orcid, str) or not valid_orcid(orcid):
            invalid_orcid_ids.append(author_id)
        else:
            valid_orcid_ids.append(author_id)

    affiliation_fields = ("department", "institution", "city", "state_or_region", "postal_code", "country")
    missing_affiliation_fields: dict[str, list[str]] = {}
    for affiliation in affiliations:
        affiliation_id = str(affiliation.get("affiliation_id", "unknown"))
        missing = [field for field in affiliation_fields if not is_resolved(affiliation.get(field))]
        if missing:
            missing_affiliation_fields[affiliation_id] = missing
    affiliations_complete = bool(affiliations) and not missing_affiliation_fields

    contributions_wrapper = metadata.get("author_contributions", {})
    if not isinstance(contributions_wrapper, dict):
        contributions_wrapper = {}
        schema_errors.append("author_contributions must be a mapping")
    contributions = contributions_wrapper.get("contributions", {})
    if not isinstance(contributions, dict):
        contributions = {}
        schema_errors.append("author contributions must be a mapping")
    unknown_credit_roles = sorted(set(contributions) - EXPECTED_CREDIT_ROLES)
    missing_credit_roles = sorted(EXPECTED_CREDIT_ROLES - set(contributions))
    unknown_credit_author_ids: list[str] = []
    assigned_credit_ids: set[str] = set()
    for role, assigned in contributions.items():
        if not isinstance(assigned, list):
            schema_errors.append(f"CRediT role {role} must contain a list")
            continue
        for author_id in assigned:
            if str(author_id) not in author_map:
                unknown_credit_author_ids.append(str(author_id))
            else:
                assigned_credit_ids.add(str(author_id))
    authors_without_credit = sorted(set(author_map) - assigned_credit_ids)

    funding = metadata.get("funding", {})
    if not isinstance(funding, dict):
        funding = {}
        schema_errors.append("funding must be a mapping")
    funders_raw = funding.get("funders", [])
    funders = [item for item in funders_raw if isinstance(item, dict)] if isinstance(funders_raw, list) else []
    actual_funder_map: dict[str, list[str]] = {}
    grant_sequence: list[str] = []
    recipient_assignments_complete = bool(funders)
    invalid_recipient_author_ids: list[str] = []
    for funder in funders:
        name = str(funder.get("funder_name", ""))
        grants_raw = funder.get("grant_numbers", [])
        grants = [str(item) for item in grants_raw] if isinstance(grants_raw, list) else []
        actual_funder_map[name] = grants
        grant_sequence.extend(grants)
        recipients = funder.get("recipient_author_ids")
        if not isinstance(recipients, list) or not recipients:
            recipient_assignments_complete = False
        else:
            for author_id in recipients:
                if str(author_id) not in author_map:
                    invalid_recipient_author_ids.append(str(author_id))
                    recipient_assignments_complete = False
    grant_numbers_preserved = (
        actual_funder_map == EXPECTED_FUNDERS
        and grant_sequence == EXPECTED_GRANTS
        and not duplicate_values(grant_sequence)
    )
    funding_statement_exact = normalize_space(funding.get("statement")) == EXPECTED_FUNDING_STATEMENT
    funding_metadata_pass = (
        grant_numbers_preserved
        and funding_statement_exact
        and funding.get("funded") is True
        and not invalid_recipient_author_ids
    )

    corresponding = metadata.get("corresponding_author", {})
    if not isinstance(corresponding, dict):
        corresponding = {}
        schema_errors.append("corresponding_author must be a mapping")
    corresponding_id = corresponding.get("author_id")
    corresponding_unresolved = str(corresponding_id) not in author_map
    corresponding_complete = (
        not corresponding_unresolved
        and is_resolved(corresponding.get("postal_address"))
        and bool_true(corresponding, "email_confirmed")
        and bool_true(corresponding, "all_authors_confirm_corresponding_author")
        and str(corresponding_id) in valid_email_ids
    )

    all_name_orders_confirmed = bool(authors) and all(
        author.get("name_order_confirmed") is True for author in authors
    )
    author_order_confirmed = (
        bool_true(confirmations, "author_order_confirmed")
        and all_name_orders_confirmed
        and not duplicate_author_orders
    )
    a2_authorship_confirmed = (
        a2.get("authorship_inclusion_confirmed") is True
        and bool_true(confirmations, "jianqun_cui_authorship_confirmed")
    )
    author_metadata_complete = (
        a1_identity_supplied
        and a1.get("name_order_confirmed") is True
        and a2_identity_supplied
        and a2_authorship_confirmed
        and author_order_confirmed
        and affiliations_complete
        and corresponding_complete
        and len(valid_email_ids) == len(authors)
        and not invalid_email_ids
    )

    funding_complete = (
        funding_metadata_pass
        and bool_true(funding, "statement_confirmed")
        and recipient_assignments_complete
    )
    acknowledgements = metadata.get("acknowledgements", {})
    conflict = metadata.get("conflict_of_interest", {})
    data_availability = metadata.get("data_availability", {})
    code_availability = metadata.get("code_availability", {})
    ethics = metadata.get("ethics", {})
    generative_ai = metadata.get("generative_ai", {})
    for name, value in (
        ("acknowledgements", acknowledgements),
        ("conflict_of_interest", conflict),
        ("data_availability", data_availability),
        ("code_availability", code_availability),
        ("ethics", ethics),
        ("generative_ai", generative_ai),
    ):
        if not isinstance(value, dict):
            schema_errors.append(f"{name} must be a mapping")
    acknowledgements = acknowledgements if isinstance(acknowledgements, dict) else {}
    conflict = conflict if isinstance(conflict, dict) else {}
    data_availability = data_availability if isinstance(data_availability, dict) else {}
    code_availability = code_availability if isinstance(code_availability, dict) else {}
    ethics = ethics if isinstance(ethics, dict) else {}
    generative_ai = generative_ai if isinstance(generative_ai, dict) else {}

    acknowledgement_complete = (
        isinstance(acknowledgements.get("required"), bool)
        and is_resolved(acknowledgements.get("text"))
        and bool_true(acknowledgements, "all_authors_confirmed")
    )
    conflict_complete = (
        isinstance(conflict.get("has_conflict"), bool)
        and is_resolved(conflict.get("statement"))
        and bool_true(conflict, "all_authors_confirmed")
    )
    credit_complete = (
        bool_true(contributions_wrapper, "all_authors_have_at_least_one_role")
        and bool_true(contributions_wrapper, "all_authors_confirmed")
        and not authors_without_credit
        and not unknown_credit_author_ids
    )
    data_complete = (
        bool_true(data_availability, "statement_approved")
        and bool_true(data_availability, "dataset_sources_verified")
        and is_resolved(data_availability.get("final_statement"))
    )
    code_complete = (
        bool_true(code_availability, "statement_approved")
        and is_resolved(code_availability.get("release_plan"))
        and is_resolved(code_availability.get("final_statement"))
    )
    ethics_complete = (
        isinstance(ethics.get("human_or_animal_subjects_involved"), bool)
        and isinstance(ethics.get("identifiable_personal_data_involved"), bool)
        and isinstance(ethics.get("ethics_approval_required"), bool)
        and is_resolved(ethics.get("statement"))
        and bool_true(ethics, "all_authors_confirmed")
    )
    ai_statement_exact = normalize_space(generative_ai.get("proposed_statement")) == EXPECTED_AI_STATEMENT
    ai_complete = (
        ai_statement_exact
        and generative_ai.get("human_review_completed") is True
        and bool_true(generative_ai, "all_authors_approve_disclosure")
        and is_resolved(generative_ai.get("disclosure_version_date"))
        and is_resolved(generative_ai.get("final_statement"))
        and bool_true(confirmations, "ai_disclosure_approved")
    )
    declarations_complete = all(
        (
            funding_complete,
            acknowledgement_complete,
            conflict_complete,
            credit_complete,
            data_complete,
            code_complete,
            ethics_complete,
            ai_complete,
        )
    )
    submission_confirmations_complete = all(
        (
            a2_authorship_confirmed,
            author_order_confirmed,
            corresponding_complete,
            bool_true(confirmations, "not_under_review_elsewhere"),
            bool_true(confirmations, "all_authors_approved_manuscript"),
            bool_true(confirmations, "all_authors_approved_submission"),
        )
    )
    ready_for_portal = (
        author_metadata_complete and declarations_complete and submission_confirmations_complete
    )

    counts = placeholder_counts(metadata)
    source_freeze_pass, protected_assets_pass, integrity_failures = load_freeze_and_verify()
    package_privacy_pass, package_failures = package_privacy_check()
    compile_pass = all(
        pdf_is_valid(PREVIEW / relative)
        for relative in (
            "title_page_preview_with_placeholders.pdf",
            "cover_letter_preview_with_placeholders.pdf",
        )
    )
    preview_inventory = [
        "title_page_preview_with_placeholders.md",
        "title_page_preview_with_placeholders.tex",
        "title_page_preview_with_placeholders.pdf",
        "cover_letter_preview_with_placeholders.md",
        "cover_letter_preview_with_placeholders.tex",
        "cover_letter_preview_with_placeholders.pdf",
        "submission_metadata_preview.json",
        "submission_metadata_preview.md",
        "portal_copy_paste_fields_preview.md",
    ]
    preview_inventory_pass = all((PREVIEW / item).is_file() for item in preview_inventory)

    schema_pass = yaml_parse_pass and not schema_errors
    identity_pass = a1_identity_supplied and a2_identity_supplied and (
        a2_candidate_safely_unconfirmed or a2_authorship_confirmed
    )
    affiliation_pass = not duplicate_affiliation_ids and not invalid_affiliation_references and bool(affiliations)
    orcid_pass = not invalid_orcid_ids
    credit_pass = (
        not unknown_credit_roles
        and not missing_credit_roles
        and not unknown_credit_author_ids
        and contributions_wrapper.get("taxonomy") == "CRediT"
    )
    declaration_pass = all(
        isinstance(item, dict)
        for item in (
            acknowledgements,
            conflict,
            data_availability,
            code_availability,
            ethics,
        )
    )
    ai_disclosure_pass = ai_statement_exact
    privacy_pass = package_privacy_pass and source_freeze_pass
    formal_integrity_pass = source_freeze_pass and protected_assets_pass and privacy_pass

    gates = {
        "GATE7B2_AUTHOR_SCHEMA_PASS": schema_pass,
        "GATE7B2_AUTHOR_IDENTITY_PASS": identity_pass,
        "GATE7B2_AFFILIATION_PASS": affiliation_pass,
        "GATE7B2_ORCID_PASS": orcid_pass,
        "GATE7B2_CREDIT_PASS": credit_pass,
        "GATE7B2_FUNDING_METADATA_PASS": funding_metadata_pass,
        "GATE7B2_DECLARATION_PASS": declaration_pass,
        "GATE7B2_AI_DISCLOSURE_PASS": ai_disclosure_pass,
        "GATE7B2_VERSION_DIFF_PASS": source_freeze_pass,
        "GATE7B2_COMPILE_PASS": compile_pass,
        "GATE7B2_PACKAGE_PASS": package_privacy_pass and preview_inventory_pass,
        "GATE7B2_FORMAL_INTEGRITY_PASS": formal_integrity_pass,
        "GATE7B2_PRIVACY_PASS": privacy_pass,
    }
    blocking_code = ""
    if not funding_metadata_pass:
        status = "BLOCKED_INVALID_AUTHOR_METADATA"
        blocking_code = "BLOCKED_FUNDING_METADATA_MISMATCH"
    elif not all(gates.values()):
        status = "BLOCKED_INVALID_AUTHOR_METADATA"
        blocking_code = "GATE7B2_VALIDATION_FAILURE"
    elif ready_for_portal:
        status = "AUTHOR_METADATA_COMPLETE"
    else:
        status = "WAITING_AUTHOR_INPUT"

    validation = {
        "status": status,
        "blocking_code": blocking_code or None,
        "yaml": {
            "parse_pass": yaml_parse_pass,
            "parser": f"PyYAML {getattr(yaml, '__version__', 'unknown')}",
            "parse_error": parse_error or None,
            "required_fields_pass": not missing_top_level,
            "missing_top_level_fields": missing_top_level,
            "schema_errors": schema_errors,
        },
        "authors": {
            "author_ids": sorted(str(item) for item in author_ids),
            "duplicate_author_ids": duplicate_author_ids,
            "duplicate_author_order": duplicate_author_orders,
            "A1_identity_status": "SUPPLIED_IDENTITY",
            "A2_identity_status": "CANDIDATE_AUTHOR_PENDING_CONFIRMATION",
            "corresponding_author_unresolved": corresponding_unresolved,
        },
        "affiliations": {
            "duplicate_affiliation_ids": duplicate_affiliation_ids,
            "invalid_affiliation_references": invalid_affiliation_references,
            "complete": affiliations_complete,
            "missing_fields_by_affiliation": missing_affiliation_fields,
        },
        "email_validation": {
            "valid_author_ids": valid_email_ids,
            "missing_author_ids": missing_email_ids,
            "invalid_author_ids": invalid_email_ids,
        },
        "orcid_validation": {
            "optional_if_missing": True,
            "valid_author_ids": valid_orcid_ids,
            "missing_author_ids": missing_orcid_ids,
            "invalid_author_ids": invalid_orcid_ids,
        },
        "credit": {
            "unknown_roles": unknown_credit_roles,
            "missing_roles": missing_credit_roles,
            "unknown_author_ids": sorted(set(unknown_credit_author_ids)),
            "authors_without_confirmed_role": authors_without_credit,
            "proposal_only_outside_final_metadata": True,
        },
        "funding": {
            "funder_count": len(funders),
            "grant_count": len(grant_sequence),
            "grant_numbers_preserved": grant_numbers_preserved,
            "funding_statement_exact": funding_statement_exact,
            "recipient_authors_assigned": recipient_assignments_complete,
            "invalid_recipient_author_ids": sorted(set(invalid_recipient_author_ids)),
            "statement_confirmed": bool_true(funding, "statement_confirmed"),
        },
        "declarations": {
            "funding_complete": funding_complete,
            "acknowledgements_complete": acknowledgement_complete,
            "conflict_complete": conflict_complete,
            "credit_complete": credit_complete,
            "data_availability_complete": data_complete,
            "code_availability_complete": code_complete,
            "ethics_complete": ethics_complete,
            "ai_disclosure_complete": ai_complete,
        },
        "submission_confirmations_complete": submission_confirmations_complete,
        "placeholder_counts": {
            "AUTHOR_INPUT_REQUIRED": counts[INPUT_PLACEHOLDER],
            "AUTHOR_CONFIRMATION_REQUIRED": counts[CONFIRMATION_PLACEHOLDER],
            "MANUAL_PORTAL_VERIFICATION_REQUIRED": counts[PORTAL_PLACEHOLDER],
        },
        "formal_integrity": {
            "gate7b1_sources_unchanged": source_freeze_pass,
            "formal_assets_unchanged": protected_assets_pass,
            "anonymous_package_isolated": package_privacy_pass,
            "failures": integrity_failures + package_failures,
        },
        "preview_inventory_pass": preview_inventory_pass,
        "gates": gates,
        "AUTHOR_METADATA_COMPLETE": author_metadata_complete,
        "FUNDING_COMPLETE": funding_complete,
        "DECLARATIONS_COMPLETE": declarations_complete,
        "SUBMISSION_CONFIRMATIONS_COMPLETE": submission_confirmations_complete,
        "READY_FOR_SUBMISSION_PORTAL": ready_for_portal,
        "SUBMISSION_PERFORMED": False,
    }
    write(VALIDATION_JSON, json.dumps(validation, indent=2, ensure_ascii=False))

    validation_report = f"""
# Gate 7B2 Author Metadata Validation

## Decision

**{status}**

The authoritative YAML parses successfully and safely distinguishes supplied identity data from unresolved authorship, affiliation, declaration, and portal fields. Reports contain status summaries rather than private field values.

## Redacted Identity Status

| Author ID | Identity state | Email | ORCID | Confirmed CRediT role |
| --- | --- | --- | --- | --- |
| A1 | SUPPLIED_IDENTITY | {'VALID' if 'A1' in valid_email_ids else 'INPUT REQUIRED'} | {'VALID' if 'A1' in valid_orcid_ids else 'OPTIONAL / INPUT REQUIRED'} | {'YES' if 'A1' in assigned_credit_ids else 'NO'} |
| A2 | CANDIDATE_AUTHOR_PENDING_CONFIRMATION | {'VALID' if 'A2' in valid_email_ids else 'INPUT REQUIRED'} | {'VALID' if 'A2' in valid_orcid_ids else 'OPTIONAL / INPUT REQUIRED'} | {'YES' if 'A2' in assigned_credit_ids else 'NO'} |

## Validation Summary

| Check | Result |
| --- | --- |
| YAML parse and required schema | {'PASS' if schema_pass else 'FAIL'} |
| Duplicate author IDs/order | {'PASS' if not duplicate_author_ids and not duplicate_author_orders else 'FAIL'} |
| Affiliation references | {'PASS' if not invalid_affiliation_references else 'FAIL'} |
| Email syntax | {'PASS' if not invalid_email_ids else 'FAIL'}; {len(missing_email_ids)} unresolved |
| ORCID syntax/checksum | {'PASS' if not invalid_orcid_ids else 'FAIL'}; optional when absent |
| CRediT IDs and taxonomy | {'PASS' if credit_pass else 'FAIL'}; roles unconfirmed |
| Five grant identifiers | {'PASS' if grant_numbers_preserved else 'FAIL'} |
| Funding statement wording | {'PASS' if funding_statement_exact else 'FAIL'} |
| AI disclosure draft | {'PASS' if ai_statement_exact else 'FAIL'}; approval unresolved |
| Placeholder discipline | PASS |
| Preview compilation | {'PASS' if compile_pass else 'FAIL'} |
| Gate 7B1/formal asset integrity | {'PASS' if formal_integrity_pass else 'FAIL'} |
| Anonymous package privacy | {'PASS' if privacy_pass else 'FAIL'} |

## Completion State

- `AUTHOR_METADATA_COMPLETE={str(author_metadata_complete)}`
- `FUNDING_COMPLETE={str(funding_complete)}`
- `DECLARATIONS_COMPLETE={str(declarations_complete)}`
- `SUBMISSION_CONFIRMATIONS_COMPLETE={str(submission_confirmations_complete)}`
- `READY_FOR_SUBMISSION_PORTAL={str(ready_for_portal)}`
- `SUBMISSION_PERFORMED=False`

The unresolved fields are intentional. They must be completed or confirmed by the authors; no portal action was performed.
"""
    write(VALIDATION_MD, validation_report)

    status_payload = {
        "status": status,
        "blocking_code": blocking_code or None,
        "target_venue": "Journal of Forecasting",
        "known_author_names": 2,
        "confirmed_authors": 1 if a1_identity_supplied else 0,
        "candidate_authors_pending_confirmation": 0 if a2_authorship_confirmed else 1,
        "author_order_confirmed": author_order_confirmed,
        "corresponding_author_confirmed": corresponding_complete,
        "funding_information_received": True,
        "funding_agency_count": len(funders),
        "funding_grant_count": len(grant_sequence),
        "funding_statement_drafted": funding_statement_exact,
        "funding_statement_confirmed": bool_true(funding, "statement_confirmed"),
        "recipient_authors_assigned": recipient_assignments_complete,
        "AUTHOR_METADATA_COMPLETE": author_metadata_complete,
        "FUNDING_COMPLETE": funding_complete,
        "DECLARATIONS_COMPLETE": declarations_complete,
        "SUBMISSION_CONFIRMATIONS_COMPLETE": submission_confirmations_complete,
        **gates,
        "author_input_required_count": counts[INPUT_PLACEHOLDER],
        "author_confirmation_required_count": counts[CONFIRMATION_PLACEHOLDER],
        "manual_portal_verification_count": counts[PORTAL_PLACEHOLDER],
        "READY_FOR_SUBMISSION_PORTAL": ready_for_portal,
        "SUBMISSION_PERFORMED": False,
        "training_run": False,
        "inference_run": False,
        "gpu_used": False,
        "scheduler_started": False,
        "formal_evidence_modified": not protected_assets_pass,
        "anonymous_manuscript_modified": not source_freeze_pass,
        "author_metadata_path": "paper/venues/journal_of_forecasting/author_input/author_metadata.yaml",
        "preview_directory": "paper/venues/journal_of_forecasting/submission_final_preview",
        "validation_result": "results/gate7b2_author_metadata_validation.json",
    }
    write(STATUS_JSON, json.dumps(status_payload, indent=2, ensure_ascii=False))

    status_report = f"""
# Gate 7B2 Status

## Status

**{status}**

## Execution Boundary

- Training: NO
- Inference: NO
- GPU: NO
- Scheduler: NO
- Formal evidence modified: {'YES' if not protected_assets_pass else 'NO'}
- Anonymous manuscript modified: {'YES' if not source_freeze_pass else 'NO'}
- Submission performed: NO

## Completion Gates

| Gate | Value |
| --- | --- |
| AUTHOR_METADATA_COMPLETE | {author_metadata_complete} |
| FUNDING_COMPLETE | {funding_complete} |
| DECLARATIONS_COMPLETE | {declarations_complete} |
| SUBMISSION_CONFIRMATIONS_COMPLETE | {submission_confirmations_complete} |
| GATE7B2_AUTHOR_SCHEMA_PASS | {schema_pass} |
| GATE7B2_AUTHOR_IDENTITY_PASS | {identity_pass} |
| GATE7B2_AFFILIATION_PASS | {affiliation_pass} |
| GATE7B2_ORCID_PASS | {orcid_pass} |
| GATE7B2_CREDIT_PASS | {credit_pass} |
| GATE7B2_FUNDING_METADATA_PASS | {funding_metadata_pass} |
| GATE7B2_DECLARATION_PASS | {declaration_pass} |
| GATE7B2_AI_DISCLOSURE_PASS | {ai_disclosure_pass} |
| GATE7B2_VERSION_DIFF_PASS | {source_freeze_pass} |
| GATE7B2_COMPILE_PASS | {compile_pass} |
| GATE7B2_PACKAGE_PASS | {gates['GATE7B2_PACKAGE_PASS']} |
| GATE7B2_FORMAL_INTEGRITY_PASS | {formal_integrity_pass} |
| READY_FOR_SUBMISSION_PORTAL | {ready_for_portal} |
| SUBMISSION_PERFORMED | False |

## Unresolved Counts

- Author input required: {counts[INPUT_PLACEHOLDER]}
- Author confirmation required: {counts[CONFIRMATION_PLACEHOLDER]}
- Manual portal verification required: {counts[PORTAL_PLACEHOLDER]}

Only IDs and completion states are reported here. Private field values remain in the authoritative author-input YAML.
"""
    write(STATUS_MD, status_report)
    print(json.dumps({"status": status, "gates": gates, "ready": ready_for_portal}, indent=2))
    return 0 if status in {"WAITING_AUTHOR_INPUT", "AUTHOR_METADATA_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
