#!/usr/bin/env python3
"""Generate Gate 7B2 instructions and placeholder previews from the sole YAML source."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
AUTHOR_DIR = VENUE / "author_input"
PREVIEW_DIR = VENUE / "submission_final_preview"
METADATA_PATH = AUTHOR_DIR / "author_metadata.yaml"


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


def load_metadata() -> dict[str, object]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Author metadata is missing: {METADATA_PATH}")
    yaml = import_yaml()
    parsed = yaml.safe_load(METADATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("author_metadata.yaml must contain a YAML mapping")
    return parsed


def author_by_id(metadata: dict[str, object], author_id: str) -> dict[str, object]:
    authors = metadata.get("authors")
    if not isinstance(authors, list):
        raise TypeError("authors must be a list")
    for author in authors:
        if isinstance(author, dict) and author.get("author_id") == author_id:
            return author
    raise KeyError(f"Missing author {author_id}")


def make_instructions(metadata: dict[str, object]) -> None:
    a1 = author_by_id(metadata, "A1")
    a2 = author_by_id(metadata, "A2")
    funding = metadata.get("funding")
    if not isinstance(funding, dict):
        raise TypeError("funding must be a mapping")
    statement = str(funding.get("statement", ""))
    write(
        AUTHOR_DIR / "author_metadata_instructions.md",
        f"""
# Gate 7B2 Author Metadata Instructions

## Source Of Truth

`author_metadata.yaml` is the only authoritative structured author-metadata file. Files in `submission_final_preview/` are generated previews and must not be edited as competing metadata sources.

The current state is **WAITING_AUTHOR_INPUT**. Replace a placeholder only after the relevant author supplies or confirms the value. Keep these exact markers while a field remains unresolved:

- `AUTHOR_INPUT_REQUIRED`
- `AUTHOR_CONFIRMATION_REQUIRED`
- `MANUAL_PORTAL_VERIFICATION_REQUIRED`

Do not place this YAML file in the anonymous manuscript or anonymous source archive. Do not print full email addresses or postal addresses in public audit reports.

## Identity Boundary

- A1 has the supplied publication name `{a1['display_name']}`. Name order, affiliation, email, corresponding-author status, and submission approvals remain unconfirmed.
- A2 has the supplied candidate name `{a2['display_name']}`. `authorship_inclusion_confirmed=false` must remain until that person explicitly agrees to authorship, order, final manuscript, submission, contributions, conflicts, and AI disclosure.
- The supervisor relationship does not establish authorship eligibility or contribution roles.

## Funding Draft

The five supplied grant identifiers are preserved exactly. Authors must verify official funder names, assign recipients if the live portal requires them, and confirm this draft before submission:

> {statement}

## Proposed CRediT Mapping

**PROPOSAL_ONLY**  
**AUTHOR_CONFIRMATION_REQUIRED**

```yaml
proposed_credit_mapping:
  A1:
    display_name: "{a1['display_name']}"
    possible_roles:
      - methodology
      - software
      - validation
      - formal_analysis
      - investigation
      - data_curation
      - writing_original_draft
      - visualization

  A2:
    display_name: "{a2['display_name']}"
    possible_roles:
      - conceptualization
      - methodology
      - supervision
      - project_administration
      - funding_acquisition
      - writing_review_editing
```

These roles are suggestions for discussion only. They are intentionally absent from the final metadata contributions map until every author confirms them.

## Affiliation Warning

Possible institution suggested by funding text:

`Central China Normal University`

This is not treated as confirmed affiliation. The `institution` field must remain `AUTHOR_INPUT_REQUIRED` until explicitly supplied by the author.

## Completion Workflow

1. Resolve author identity, inclusion, order, affiliations, emails, and corresponding author.
2. Confirm CRediT roles, funding recipients, acknowledgements, conflicts, availability statements, ethics, and AI disclosure.
3. Confirm exclusivity and unanimous manuscript/submission approval.
4. Validate the YAML with `scripts/gate7b2_validate_author_metadata.py`.
5. Verify every `MANUAL_PORTAL_VERIFICATION_REQUIRED` field in the live Wiley portal.

ORCID is optional in this project file. If supplied, it must pass the ISO 7064 MOD 11-2 checksum.
""",
    )


ACTIONS = [
    "Confirm whether Jianqun Cui is a manuscript author.",
    "Confirm final author order.",
    "Select corresponding author.",
    "Provide department and institution for each author.",
    "Provide city, postal code and country details.",
    "Provide author emails.",
    "Provide ORCID identifiers if available.",
    "Confirm CRediT roles.",
    "Confirm which authors received each listed grant.",
    "Confirm exact funding statement.",
    "Confirm acknowledgements.",
    "Confirm conflict-of-interest statement.",
    "Confirm data-availability statement.",
    "Select code-release plan and provide URL when available.",
    "Confirm ethics statement.",
    "Approve AI-use disclosure and version date.",
    "Confirm manuscript is not under review elsewhere.",
    "Confirm all authors approved the manuscript and submission.",
    "Decide whether reviewer suggestions are needed.",
    "Confirm open-access choice in the live portal.",
]


def make_remaining_actions() -> None:
    body = "# Gate 7B2 Remaining Author Actions\n\n" + "\n".join(
        f"{index}. {action}" for index, action in enumerate(ACTIONS, start=1)
    )
    body += "\n\nNo private values are reproduced in this checklist.\n"
    write(ROOT / "reports/gate7b2_remaining_author_actions.md", body)
    write(AUTHOR_DIR / "remaining_author_actions.md", body)


def make_title_page_preview(metadata: dict[str, object]) -> None:
    manuscript = metadata["manuscript"]
    funding = metadata["funding"]
    if not isinstance(manuscript, dict) or not isinstance(funding, dict):
        raise TypeError("manuscript and funding must be mappings")
    title = str(manuscript["title"])
    short_title = str(manuscript["short_title"])
    statement = str(funding["statement"])
    write(
        PREVIEW_DIR / "title_page_preview_with_placeholders.md",
        f"""
# {title}

**Running title:** {short_title}

## Authors

- Zian Zhong
- Jianqun Cui **[AUTHORSHIP CONFIRMATION REQUIRED]**

## Affiliations

**[AUTHOR INPUT REQUIRED]**

Possible institution text appearing in the funding record is not treated as an affiliation.

## Corresponding Author

**[CORRESPONDING AUTHOR REQUIRED]**

Email, ORCID, and postal address: **[AUTHOR INPUT REQUIRED]**

## Funding

{statement}

**[AUTHOR CONFIRMATION REQUIRED]**

## Acknowledgements

**[AUTHOR INPUT REQUIRED]**

## Conflict Of Interest

**[AUTHOR INPUT REQUIRED]**

## Author Contributions

**[AUTHOR CONFIRMATION REQUIRED]**

## Data And Code Availability

**[AUTHOR CONFIRMATION REQUIRED]**

## Generative AI Disclosure

**[AUTHOR CONFIRMATION REQUIRED]**

This is a placeholder preview, not an author-complete title page.
""",
    )
    write(
        PREVIEW_DIR / "title_page_preview_with_placeholders.tex",
        rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[hidelinks]{{hyperref}}
\setlength{{\emergencystretch}}{{4em}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries {title}\par}}
\vspace{{0.5em}}
{{\small Running title: {short_title}\par}}
\end{{center}}
\begin{{description}}
\item[Authors] Zian Zhong; Jianqun Cui \textbf{{[AUTHORSHIP CONFIRMATION REQUIRED]}}
\item[Affiliations] \textbf{{[AUTHOR INPUT REQUIRED]}}
\item[Corresponding author] \textbf{{[CORRESPONDING AUTHOR REQUIRED]}}
\item[Email, ORCID, and address] \textbf{{[AUTHOR INPUT REQUIRED]}}
\item[Funding] {statement} \textbf{{[AUTHOR CONFIRMATION REQUIRED]}}
\item[Acknowledgements] \textbf{{[AUTHOR INPUT REQUIRED]}}
\item[Conflict of interest] \textbf{{[AUTHOR INPUT REQUIRED]}}
\item[Author contributions] \textbf{{[AUTHOR CONFIRMATION REQUIRED]}}
\item[Data and code availability] \textbf{{[AUTHOR CONFIRMATION REQUIRED]}}
\item[Generative AI disclosure] \textbf{{[AUTHOR CONFIRMATION REQUIRED]}}
\end{{description}}
\vfill
\noindent\textit{{Placeholder preview only; not an author-complete title page.}}
\end{{document}}
""",
    )


def make_cover_letter_preview(metadata: dict[str, object]) -> None:
    manuscript = metadata["manuscript"]
    if not isinstance(manuscript, dict):
        raise TypeError("manuscript must be a mapping")
    title = str(manuscript["title"])
    write(
        PREVIEW_DIR / "cover_letter_preview_with_placeholders.md",
        f"""
# Cover Letter Preview With Placeholders

Editor  
*Journal of Forecasting*

Dear Editor,

Please consider the Research Article "{title}" for the *Journal of Forecasting*. The manuscript examines probabilistic time-series forecasting when observations are missing from the historical input and evaluates fixed heavy-tailed forecast distributions under a frozen, paired protocol.

Submitting institution: **[AUTHOR INPUT REQUIRED]**

Corresponding author: **[CORRESPONDING AUTHOR REQUIRED]**

The manuscript is not under consideration elsewhere: **[AUTHOR CONFIRMATION REQUIRED]**

All authors have approved the manuscript and submission: **[AUTHOR CONFIRMATION REQUIRED]**

The conflict-of-interest declaration is complete and approved: **[AUTHOR CONFIRMATION REQUIRED]**

Sincerely,

Zian Zhong  
on behalf of all authors  
**[AUTHOR CONFIRMATION REQUIRED]**

This is a placeholder preview, not a submission-ready cover letter.
""",
    )
    write(
        PREVIEW_DIR / "cover_letter_preview_with_placeholders.tex",
        rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\setlength{{\emergencystretch}}{{4em}}
\setlength{{\parskip}}{{0.8em}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\thispagestyle{{empty}}
\noindent Editor\\
\emph{{Journal of Forecasting}}

\vspace{{1em}}
\noindent Dear Editor,

Please consider the Research Article ``{title}'' for the \emph{{Journal of Forecasting}}. The manuscript examines probabilistic time-series forecasting when observations are missing from the historical input and evaluates fixed heavy-tailed forecast distributions under a frozen, paired protocol.

Submitting institution: \textbf{{[AUTHOR INPUT REQUIRED]}}

Corresponding author: \textbf{{[CORRESPONDING AUTHOR REQUIRED]}}

The manuscript is not under consideration elsewhere:\\
\textbf{{[AUTHOR CONFIRMATION REQUIRED]}}

All authors have approved the manuscript and submission:\\
\textbf{{[AUTHOR CONFIRMATION REQUIRED]}}

The conflict-of-interest declaration is complete and approved:\\
\textbf{{[AUTHOR CONFIRMATION REQUIRED]}}

\noindent Sincerely,\\
Zian Zhong\\
on behalf of all authors\\
\textbf{{[AUTHOR CONFIRMATION REQUIRED]}}

\vfill
\noindent\textit{{Placeholder preview only; not a submission-ready cover letter.}}
\end{{document}}
""",
    )


def make_portal_previews(metadata: dict[str, object]) -> None:
    manuscript = metadata["manuscript"]
    funding = metadata["funding"]
    if not isinstance(manuscript, dict) or not isinstance(funding, dict):
        raise TypeError("manuscript and funding must be mappings")
    preview = {
        "document_type": "PREVIEW_WITH_PLACEHOLDERS",
        "authoritative_source": "paper/venues/journal_of_forecasting/author_input/author_metadata.yaml",
        "target_venue": "Journal of Forecasting",
        "publisher": "Wiley",
        "article_type": manuscript["article_type"],
        "title": manuscript["title"],
        "short_title": manuscript["short_title"],
        "authors": [
            {"author_id": "A1", "display_name": "Zian Zhong", "status": "SUPPLIED_NAME"},
            {"author_id": "A2", "display_name": "Jianqun Cui", "status": "AUTHORSHIP_CONFIRMATION_REQUIRED"},
        ],
        "author_order": "AUTHOR_CONFIRMATION_REQUIRED",
        "affiliations": "AUTHOR_INPUT_REQUIRED",
        "corresponding_author": "AUTHOR_INPUT_REQUIRED",
        "emails": "AUTHOR_INPUT_REQUIRED",
        "orcid": "AUTHOR_INPUT_REQUIRED",
        "funding_statement": funding["statement"],
        "funding_statement_confirmed": False,
        "funding_recipients": "AUTHOR_INPUT_REQUIRED",
        "credit_roles": "AUTHOR_CONFIRMATION_REQUIRED",
        "acknowledgements": "AUTHOR_INPUT_REQUIRED",
        "conflict_of_interest": "AUTHOR_INPUT_REQUIRED",
        "data_availability": "AUTHOR_INPUT_REQUIRED",
        "code_availability": "AUTHOR_INPUT_REQUIRED",
        "ethics": "AUTHOR_INPUT_REQUIRED",
        "generative_ai_disclosure": "AUTHOR_CONFIRMATION_REQUIRED",
        "submission_confirmations": "AUTHOR_CONFIRMATION_REQUIRED",
        "manual_portal_verification": "MANUAL_PORTAL_VERIFICATION_REQUIRED",
        "ready_for_submission_portal": False,
        "submission_performed": False,
    }
    write(
        PREVIEW_DIR / "submission_metadata_preview.json",
        json.dumps(preview, indent=2, ensure_ascii=False),
    )
    write(
        PREVIEW_DIR / "submission_metadata_preview.md",
        f"""
# Submission Metadata Preview

This file is derived from `author_input/author_metadata.yaml`; it is not a second source of truth.

| Field | Preview value |
| --- | --- |
| Venue | Journal of Forecasting |
| Publisher | Wiley |
| Article type | {manuscript['article_type']} |
| Title | {manuscript['title']} |
| Authors | A1 supplied name; A2 candidate author pending confirmation |
| Author order | [AUTHOR CONFIRMATION REQUIRED] |
| Affiliations | [AUTHOR INPUT REQUIRED] |
| Corresponding author | [AUTHOR INPUT REQUIRED] |
| Emails and ORCIDs | [AUTHOR INPUT REQUIRED] |
| Funding | Drafted; [AUTHOR CONFIRMATION REQUIRED] |
| CRediT | [AUTHOR CONFIRMATION REQUIRED] |
| Declarations | [AUTHOR INPUT REQUIRED] and [AUTHOR CONFIRMATION REQUIRED] |
| Portal-only fields | [MANUAL PORTAL VERIFICATION REQUIRED] |
| Ready for submission portal | NO |
| Submission performed | NO |
""",
    )
    write(
        PREVIEW_DIR / "portal_copy_paste_fields_preview.md",
        f"""
# Journal Of Forecasting Portal Fields Preview

**PREVIEW WITH PLACEHOLDERS -- DO NOT SUBMIT**

| Portal field | Copy/paste candidate or required action |
| --- | --- |
| Article type | Research Article [MANUAL PORTAL VERIFICATION REQUIRED] |
| Title | {manuscript['title']} |
| Short title | {manuscript['short_title']} |
| Author A1 | Supplied publication name; remaining identity fields [AUTHOR INPUT REQUIRED] |
| Author A2 | Candidate name; [AUTHORSHIP CONFIRMATION REQUIRED] |
| Author order | [AUTHOR CONFIRMATION REQUIRED] |
| Affiliations | [AUTHOR INPUT REQUIRED] |
| Corresponding author | [CORRESPONDING AUTHOR REQUIRED] |
| Funding organizations and grants | Draft preserved in YAML; [AUTHOR CONFIRMATION REQUIRED] |
| Funding recipients | [AUTHOR INPUT REQUIRED] |
| Acknowledgements | [AUTHOR INPUT REQUIRED] |
| Conflict of interest | [AUTHOR INPUT REQUIRED] |
| CRediT roles | [AUTHOR CONFIRMATION REQUIRED] |
| Data availability | [AUTHOR INPUT REQUIRED] |
| Code availability | [AUTHOR INPUT REQUIRED] |
| Ethics | [AUTHOR INPUT REQUIRED] |
| AI disclosure | Draft available; [AUTHOR CONFIRMATION REQUIRED] |
| Reviewer fields | [MANUAL PORTAL VERIFICATION REQUIRED] |
| Subject categories | [MANUAL PORTAL VERIFICATION REQUIRED] |
| Open access | [AUTHOR INPUT REQUIRED] |
| Upload categories | [MANUAL PORTAL VERIFICATION REQUIRED] |
| Exclusivity and approvals | [AUTHOR CONFIRMATION REQUIRED] |

No upload, account creation, email, or submission has been performed.
""",
    )


def main() -> int:
    metadata = load_metadata()
    make_instructions(metadata)
    make_remaining_actions()
    make_title_page_preview(metadata)
    make_cover_letter_preview(metadata)
    make_portal_previews(metadata)
    print("Generated Gate 7B2 author instructions, action lists, and placeholder previews.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
