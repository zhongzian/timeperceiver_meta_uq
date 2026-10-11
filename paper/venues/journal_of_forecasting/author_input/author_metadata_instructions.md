# Gate 7B2 Author Metadata Instructions

## Source Of Truth

`author_metadata.yaml` is the only authoritative structured author-metadata file. Files in `submission_final_preview/` are generated previews and must not be edited as competing metadata sources.

The current state is **WAITING_AUTHOR_INPUT**. Replace a placeholder only after the relevant author supplies or confirms the value. Keep these exact markers while a field remains unresolved:

- `AUTHOR_INPUT_REQUIRED`
- `AUTHOR_CONFIRMATION_REQUIRED`
- `MANUAL_PORTAL_VERIFICATION_REQUIRED`

Do not place this YAML file in the anonymous manuscript or anonymous source archive. Do not print full email addresses or postal addresses in public audit reports.

## Identity Boundary

- A1 has the supplied publication name `Zian Zhong`. Name order, affiliation, email, corresponding-author status, and submission approvals remain unconfirmed.
- A2 has the supplied candidate name `Jianqun Cui`. `authorship_inclusion_confirmed=false` must remain until that person explicitly agrees to authorship, order, final manuscript, submission, contributions, conflicts, and AI disclosure.
- The supervisor relationship does not establish authorship eligibility or contribution roles.

## Funding Draft

The five supplied grant identifiers are preserved exactly. Authors must verify official funder names, assign recipients if the live portal requires them, and confirm this draft before submission:

> This work was supported by the National Natural Science Foundation of China (Nos. 62272189, 62372206, and 62402193), the Hubei Provincial Natural Science Foundation of China (No. 2024AFB421), and the Fundamental Research Funds for the Central Universities, Central China Normal University (No. XJ2026001501).

## Proposed CRediT Mapping

**PROPOSAL_ONLY**  
**AUTHOR_CONFIRMATION_REQUIRED**

```yaml
proposed_credit_mapping:
  A1:
    display_name: "Zian Zhong"
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
    display_name: "Jianqun Cui"
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
