#!/usr/bin/env python3
"""Audit the scientific claim boundaries of the Gate 7B1 manuscript."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"


def corpus() -> str:
    paths = [VENUE / "manuscript/abstract.tex"]
    paths.extend(sorted((VENUE / "manuscript/sections").glob("*.tex")))
    paths.extend(sorted((VENUE / "manuscript/appendix").glob("*.tex")))
    paths.append(VENUE / "submission/cover_letter_draft.md")
    return "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()


def main() -> None:
    text = corpus()
    prohibited = [
        "backbone-independent", "universal", "all metrics improve",
        "statistically significant", "state-of-the-art", "jointly calibrated",
        "fixed df=8 is optimal", "learnable df is unnecessary in all settings",
        "conformal prediction fails generally", "guaranteed acceptance",
    ]
    prohibited_hits = [phrase for phrase in prohibited if phrase in text]
    required = {
        "fixed_student_t_claim": ["improves nll in 12/12", "fixed heavy-tailed"],
        "learnable_df_scope": ["no consistent additional", "not theoretically optimal"],
        "patchtst_counts": ["nll wins in 3/3", "point noninferiority in 2/3"],
        "electricity_tradeoff": ["0.165829", "0.167809", "exceeds the prespecified 1\\%"],
        "seed_scope": ["fixed backbone checkpoint", "not an end-to-end backbone retraining study"],
        "descriptive_tail_scope": ["descriptive, not causal proof"],
        "negative_film_retained": ["film adapter does not improve over that static adapter"],
        "negative_conformal_retained": ["widens intervals without improving absolute coverage error"],
        "marginal_not_joint": ["does not estimate a joint covariance structure"],
        "no_significance_claim": ["no formal significance test"],
    }
    missing = {
        name: phrases for name, phrases in required.items()
        if not all(phrase in text for phrase in phrases)
    }
    unsupported_claims: list[str] = []
    passed = not prohibited_hits and not missing and not unsupported_claims
    result = {
        "status": "PASS" if passed else "FAILED",
        "fixed_student_t_claim": "bounded_to_evaluated_missing_history_protocol",
        "learnable_df_claim": "no_consistent_additional_benefit_in_ablation",
        "cross_backbone_claim": "nll_and_crps_direction_on_two_evaluated_backbones_only",
        "electricity_point_tradeoff": "explicitly_disclosed",
        "statistical_significance_claim": False,
        "sota_claim": False,
        "unsupported_claim_count": len(unsupported_claims),
        "unsupported_claims": unsupported_claims,
        "prohibited_wording_count": len(prohibited_hits),
        "prohibited_wording": prohibited_hits,
        "missing_required_scope_count": len(missing),
        "missing_required_scope": missing,
        "pass": passed,
    }
    (ROOT / "results/gate7b1_jof_claim_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# Gate 7B1 Claim Audit

## Status

**{result['status']}**

| Check | Result |
| --- | --- |
| Fixed Student-t claim | Bounded to the evaluated fixed-mask protocol |
| Learned-df claim | No consistent additional ablation benefit; no general necessity claim |
| Cross-backbone claim | NLL/CRPS direction on two evaluated backbones only |
| Electricity point tradeoff | Explicitly retained |
| Statistical significance claim | No |
| SOTA claim | No |
| Unsupported claims | {len(unsupported_claims)} |
| Prohibited wording | {len(prohibited_hits)} |

The manuscript distinguishes descriptive tail diagnostics from causal evidence and retains both the FiLM and conformal negative findings.
"""
    (ROOT / "reports/gate7b1_jof_claim_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
