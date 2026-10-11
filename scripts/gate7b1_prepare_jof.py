#!/usr/bin/env python3
"""Prepare the separate Journal of Forecasting Free Format manuscript tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENUE = ROOT / "paper/venues/journal_of_forecasting"
MANUSCRIPT = VENUE / "manuscript"
SUBMISSION = VENUE / "submission"
SUPPLEMENT = VENUE / "supplementary"
DECLARATIONS = VENUE / "declarations"
METADATA = VENUE / "metadata"
CHECKS = VENUE / "checks"
ARCHIVE = VENUE / "archive"
FREEZE = ROOT / "artifacts/gate7b1_jof_pre_adaptation/source_hashes.sha256"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_sources() -> list[dict[str, str]]:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    records: list[dict[str, str]] = []
    for line in FREEZE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.lstrip("* ")
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {relative}")
        records.append({"path": relative, "sha256": actual})
    if len(records) != 15:
        raise RuntimeError(f"Expected 15 frozen inputs, found {len(records)}")
    return records


def copy_manuscript_assets() -> None:
    source = ROOT / "paper/manuscript"
    for kind in ("tables", "figures"):
        target = MANUSCRIPT / kind
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((source / kind).glob("*")):
            if path.is_file():
                shutil.copy2(path, target / path.name)
    extra = ROOT / "paper/figures/fig4_student_t_df.pdf"
    if not extra.is_file():
        raise FileNotFoundError(extra)
    shutil.copy2(extra, MANUSCRIPT / "figures/fig4_student_t_df.pdf")


def make_guideline_audit() -> None:
    retrieved = "2026-07-22"
    audit = {
        "status": "PASS",
        "journal": "Journal of Forecasting",
        "publisher": "Wiley",
        "retrieved_at": retrieved,
        "official_scope_source": "https://onlinelibrary.wiley.com/journal/1099131x",
        "official_author_guidelines_source": "https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/forauthors.html",
        "official_submission_source": "https://onlinelibrary.wiley.com/page/journal/1099131x/homepage/forauthors.html",
        "official_data_sharing_source": "https://authorservices.wiley.com/author-resources/Journal-Authors/open-access/data-sharing-citation/data-sharing-policy.html",
        "official_ai_policy_source": "https://www.wiley.com/en-us/publish/article/ai-guidelines/",
        "official_ethics_source": "https://authorservices.wiley.com/ethics-guidelines/index.html",
        "article_type": "Research Article",
        "free_format_submission": True,
        "latex_accepted": True,
        "review_model": "single-blind",
        "abstract_limit": "NOT_STATED",
        "keyword_limit": "up to 6 keywords",
        "word_limit": "NOT_STATED",
        "page_limit": "NOT_STATED",
        "figure_limit": "NOT_STATED",
        "table_limit": "NOT_STATED",
        "reference_style": "APA at final formatting; any consistent style at initial Free Format submission",
        "title_page_requirements": [
            "full title", "short running title under 40 characters", "author names",
            "affiliations", "email addresses", "acknowledgements", "conflict of interest",
        ],
        "author_information_requirements": "All coauthor details, affiliations, and email addresses; ORCID required during submission.",
        "corresponding_author_requirements": "Corresponding author identity and email required.",
        "data_availability_statement": "Expected for research articles under Wiley data sharing policy.",
        "code_availability_statement": "NOT_STATED",
        "conflict_of_interest_statement": "Required on title page.",
        "funding_statement": "Funding information must be supplied when applicable.",
        "author_contribution_statement": "MANUAL_VERIFICATION_REQUIRED",
        "AI_use_disclosure": "Required when AI substantially assisted drafting, restructuring, literature review, or code.",
        "ethics_statement": "Required when applicable; this computational study reports no human or animal participants.",
        "cover_letter": "NOT_STATED",
        "highlights": "NOT_STATED",
        "graphical_abstract": "NOT_STATED",
        "supplementary_material": "Upload supporting information as separate files.",
        "source_file_requirements": "Upload editable source; for LaTeX provide the .tex source and a peer-review PDF, with supporting files separately.",
        "unknown_fields": [
            "abstract_limit", "word_limit", "page_limit", "figure_limit", "table_limit",
            "cover_letter_requirement", "highlights_requirement", "graphical_abstract_requirement",
        ],
        "manual_verification_fields": [
            "exact article type in the live portal", "cover-letter requirement",
            "data/code statement portal fields", "AI disclosure portal field",
            "supplement upload category", "LaTeX source upload timing",
            "current APC and open-access options",
        ],
        "free_format_primary_artifact": "single-column manuscript PDF",
        "third_party_template_used": False,
        "manual_portal_verification_count": 7,
        "pass": True,
    }
    write(ROOT / "results/gate7b1_jof_guideline_audit.json", json.dumps(audit, indent=2))
    unknown = "\n".join(f"- `{item}`" for item in audit["unknown_fields"])
    manual = "\n".join(f"- {item}" for item in audit["manual_verification_fields"])
    report = f"""
# Gate 7B1 Journal Guideline Audit

## Decision

**PASS** -- the primary artifact is a single-column Free Format PDF with editable LaTeX source. No third-party journal class is used.

## Verified Requirements

| Field | Verified value |
| --- | --- |
| Journal | Journal of Forecasting |
| Publisher | Wiley |
| Article type | Research Article |
| Submission mode | Free Format |
| Review model | Single-blind |
| LaTeX | Accepted; source and peer-review PDF supported |
| Keywords | Up to 6 |
| Reference style | APA at final formatting; consistent style accepted initially |
| Title page | Full title, short title, author/affiliation/email details, acknowledgements, conflicts |
| Supplement | Separate supporting-information files |
| Retrieved | {retrieved} |

## Official Sources

- [Journal homepage]({audit['official_scope_source']})
- [Author guidelines]({audit['official_author_guidelines_source']})
- [Wiley data sharing policy]({audit['official_data_sharing_source']})
- [Wiley AI guidance]({audit['official_ai_policy_source']})
- [Wiley publication ethics]({audit['official_ethics_source']})

## Not Stated

{unknown}

## Manual Portal Verification

{manual}

The unstated limits are preserved as `NOT_STATED`; no limits were inferred.
"""
    write(ROOT / "reports/gate7b1_jof_guideline_audit.md", report)


def make_metadata_and_declarations(abstract_plain: str) -> None:
    title = "Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations"
    short_title = "Heavy-Tailed Forecasts with Missingness"
    keywords = [
        "probabilistic forecasting", "time-series forecasting", "missing observations",
        "Student-t distribution", "proper scoring rules", "heavy-tailed errors",
    ]
    write(METADATA / "title_candidates.md", """
# Title Candidates

1. **Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations** (default)
2. Probabilistic Time-Series Forecasting with Fixed Heavy-Tailed Likelihoods under Missing Historical Data
3. Evaluating Fixed Student-t Forecast Distributions under Missing Historical Observations
""")
    write(METADATA / "keywords.txt", "\n".join(keywords))
    metadata = {
        "journal": "Journal of Forecasting",
        "publisher": "Wiley",
        "article_type": "Research Article",
        "submission_mode": "Free Format",
        "review_model": "single-blind",
        "title": title,
        "short_title": short_title,
        "abstract": abstract_plain,
        "keywords": keywords,
        "authors": "AUTHOR_INPUT_REQUIRED",
        "affiliations": "AUTHOR_INPUT_REQUIRED",
        "corresponding_author": "AUTHOR_INPUT_REQUIRED",
        "emails": "AUTHOR_INPUT_REQUIRED",
        "orcid": "AUTHOR_INPUT_REQUIRED",
        "suggested_classifications": ["probabilistic forecasting", "forecast evaluation", "missing data"],
        "data_availability": "The datasets used in this study are publicly available from their original repositories. Dataset sources and preprocessing details are documented in the supplementary material.",
        "code_availability": "AUTHOR_ACTION_REQUIRED: Code and experiment configurations will be made available upon publication or through an anonymous review repository, subject to the journal's review policy.",
        "conflict_of_interest": "AUTHOR_INPUT_REQUIRED",
        "funding": "AUTHOR_INPUT_REQUIRED",
        "author_contributions": "AUTHOR_INPUT_REQUIRED",
        "AI_use_disclosure": "AUTHOR_CONFIRMATION_REQUIRED",
        "reviewer_suggestions": "AUTHOR_INPUT_REQUIRED_IF_REQUESTED",
        "opposed_reviewers": "AUTHOR_INPUT_REQUIRED_IF_REQUESTED",
        "cover_letter": "submission/cover_letter_draft.pdf",
        "ready_for_submission_portal": False,
    }
    write(METADATA / "submission_metadata.json", json.dumps(metadata, indent=2))
    write(METADATA / "submission_metadata.md", f"""
# Submission Metadata

| Field | Value |
| --- | --- |
| Journal | Journal of Forecasting |
| Article type | Research Article |
| Submission mode | Free Format |
| Review model | Single-blind |
| Title | {title} |
| Short title | {short_title} |
| Authors | [AUTHOR INPUT REQUIRED] |
| Affiliations | [AUTHOR INPUT REQUIRED] |
| Corresponding author | [AUTHOR INPUT REQUIRED] |
| Email and ORCID | [AUTHOR INPUT REQUIRED] |
| Suggested classifications | Probabilistic forecasting; forecast evaluation; missing data |
| Reviewer suggestions | [AUTHOR INPUT REQUIRED IF REQUESTED] |
| Opposed reviewers | [AUTHOR INPUT REQUIRED IF REQUESTED] |

## Abstract

{abstract_plain}

## Keywords

{'; '.join(keywords)}

## Statements

- Data availability: The datasets are publicly available from their original repositories; sources and preprocessing are documented in the supplement.
- Code availability: [AUTHOR ACTION REQUIRED -- repository URL and release timing]
- Conflict of interest: [AUTHOR INPUT REQUIRED]
- Funding: [FUNDING INFORMATION REQUIRED]
- Author contributions: [AUTHOR INPUT REQUIRED]
- AI-use disclosure: [AUTHOR CONFIRMATION REQUIRED]
""")

    declarations = {
        "data_availability_statement.md": """
# Data Availability Statement

The datasets used in this study are publicly available from their original repositories. Dataset sources and preprocessing details are documented in the supplementary material.
""",
        "code_availability_statement.md": """
# Code Availability Statement

Code and experiment configurations will be made available upon publication or through an anonymous review repository, subject to the journal's review policy.

**AUTHOR ACTION REQUIRED:** provide and verify the repository URL and release timing before submission. `[REPOSITORY URL REQUIRED]`
""",
        "conflict_of_interest_statement.md": """
# Conflict of Interest Statement

[AUTHOR INPUT REQUIRED]

Every author must review and confirm the final conflict-of-interest statement before submission.
""",
        "funding_statement.md": """
# Funding Statement

[FUNDING INFORMATION REQUIRED]

Do not replace this placeholder until every author has confirmed the relevant funders and grant identifiers, or confirmed that no specific funding applies.
""",
        "author_contributions.md": """
# Author Contributions

[AUTHOR INPUT REQUIRED]

Complete this statement using confirmed contributor roles. No contribution is assigned in this draft.
""",
        "ethics_statement.md": """
# Ethics Statement

This computational study used public benchmark time-series datasets and did not involve human participants, identifiable personal data, animals, or prospective interventions. [AUTHOR CONFIRMATION REQUIRED]
""",
        "generative_ai_disclosure.md": """
# Generative AI Disclosure

OpenAI ChatGPT and Codex were used under author direction for language editing, manuscript organization and restructuring, and code assistance. The authors reviewed AI-assisted text and code against the formal experiment records and official sources, retained responsibility for all scientific decisions, and will verify the final manuscript. AI tools are not authors.

**AUTHOR CONFIRMATION REQUIRED:** confirm this wording, tool names, relevant versions, and dates before submission.
""",
    }
    for name, text in declarations.items():
        write(DECLARATIONS / name, text)


def make_manuscript() -> None:
    abstract_tex = r"""
Forecasting systems often require predictive uncertainty as well as point forecasts, yet incomplete historical observations change the error distribution used to estimate that uncertainty. Comparisons are additionally weakened when competing methods see different missingness realizations. We study probabilistic time-series forecasting under fixed masks applied only to historical inputs, with train-only normalization and paired evaluation assets. A lightweight distribution head represents each horizon--channel marginal with a fixed Student-$t$ distribution with degrees of freedom $8$, using the same trainable location and scale projections as Gaussian and Laplace alternatives. Proper probabilistic scores are central: negative log-likelihood and a Monte Carlo estimate of the continuous ranked probability score are reported alongside point-prediction noninferiority. On fixed TimePerceiver checkpoints, Student-$t$ improves negative log-likelihood in 12/12 dataset--seed pairs and is noninferior for both continuous ranked probability score and point prediction in 12/12. A distribution-head ablation supports fixed heavy tails over learned degrees of freedom in the evaluated protocol. On PatchTST, probabilistic-score gains appear in 3/3 datasets, while Electricity exceeds the prespecified point-error tolerance. The results support fixed heavy-tailed predictive distributions as a lightweight method for improving probabilistic forecast quality under missing historical observations, while the PatchTST Electricity result cautions against assuming metric-wise invariance across backbones.
"""
    abstract_plain = " ".join(abstract_tex.replace("$", "").replace("--", "-").split())
    write(MANUSCRIPT / "abstract.tex", abstract_tex)
    make_metadata_and_declarations(abstract_plain)

    write(MANUSCRIPT / "macros.tex", r"""
\newcommand{\PaperTitle}{Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations}
\newcommand{\ShortTitle}{Heavy-Tailed Forecasts with Missingness}
\newcommand{\StudentT}{Student-$t$}
\newcommand{\df}{\mathrm{df}}
""")
    write(MANUSCRIPT / "title_page.tex", r"""
\title{\PaperTitle}
\author{\textbf{[AUTHOR INPUT REQUIRED]}\\
\small Affiliations: [AUTHOR INPUT REQUIRED]\\
\small Corresponding author and email: [AUTHOR INPUT REQUIRED]\\
\small ORCID identifiers: [AUTHOR INPUT REQUIRED]}
\date{}
\maketitle
\begin{center}
\small Running title: \ShortTitle
\end{center}
""")

    preamble = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,array,booktabs,graphicx,longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\newcommand{\citep}[1]{\cite{#1}}
\makeatletter
\renewcommand{\@cite}[2]{(#1\if@tempswa, #2\fi)}
\makeatother
\input{macros}
\setlength{\parindent}{1.2em}
\setlength{\parskip}{0.25em}
\setlength{\emergencystretch}{4em}
\graphicspath{{figures/}}
"""
    body = r"""
\begin{document}
\input{title_page}
\begin{abstract}
\input{abstract}
\end{abstract}
\noindent\textbf{Keywords:} probabilistic forecasting; time-series forecasting; missing observations; Student-$t$ distribution; proper scoring rules; heavy-tailed errors

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_forecasting_problem}
\input{sections/04_probabilistic_method}
\input{sections/05_evaluation_design}
\input{sections/06_empirical_results}
\input{sections/07_diagnostics_and_negative_results}
\input{sections/08_discussion}
\input{sections/09_limitations}
\input{sections/10_conclusion}

\section*{Statements and Declarations}
\textbf{Data availability.} The datasets are publicly available from their original repositories; sources and preprocessing details are documented in the supplementary material.\\
\textbf{Code availability.} [AUTHOR ACTION REQUIRED: provide the verified repository URL and release timing.]\\
\textbf{Funding, conflicts, and contributions.} [AUTHOR INPUT REQUIRED.]\\
\textbf{Generative AI.} [AUTHOR CONFIRMATION REQUIRED: see the separate disclosure draft.]

\input{references_list}
\end{document}
"""
    write(MANUSCRIPT / "main.tex", preamble + body)
    anonymous = body.replace("\\input{title_page}", r"""\title{\PaperTitle}
\author{Anonymous review copy}
\date{}
\maketitle
\begin{center}\small Running title: \ShortTitle\end{center}""")
    write(MANUSCRIPT / "anonymous_main.tex", preamble + anonymous)

    sections: dict[str, str] = {
        "01_introduction.tex": r"""
\section{Introduction}
\label{sec:introduction}

Forecasting decisions require more than a point estimate when the costs of under- and over-prediction differ. Predictive distributions express uncertainty across the forecast horizon and support decisions that depend on tail risk, quantiles, or intervals. Modern probabilistic forecasting therefore evaluates the entire distribution, not only its center \citep{gneiting2007proper,gneiting2014probabilistic,salinas2020deepar}.

Historical observations are often incomplete. Missing values can remove recent evidence, entire channels, or contiguous blocks, changing both point errors and the residual shape available to a probability model. Recurrent imputation methods, deep generative imputers, and masked forecasting architectures address different parts of this problem \citep{cao2018brits,che2018grud,fortuin2020gpvae,tashiro2021csdi,du2023saits,lee2025timeperceiver}, but Gaussian residual assumptions can remain fragile when errors become heavy-tailed.

Probabilistic forecasts should be compared with proper scores that reward honest distributions. Negative log-likelihood (NLL) is distribution-sensitive, while the continuous ranked probability score (CRPS) evaluates calibration and sharpness together \citep{matheson1976crps,gneiting2007proper,jordan2019scoringrules}. Mean squared error (MSE) and mean absolute error (MAE) remain useful point criteria, but neither replaces distributional evaluation. Coverage and width are likewise diagnostics rather than complete quality measures in isolation.

Paired comparisons also require identical evaluation inputs. If Gaussian and heavy-tailed heads are evaluated under independently sampled masks, mask variation can be mistaken for a distribution-head effect. We therefore fix train, validation, and test splits; fit normalization on training data only; and reuse the same validation and test masks within every comparison.

The proposed final specification is deliberately small: a fixed \StudentT{} predictive marginal with $\df=8$. Gaussian, Laplace, and fixed \StudentT{} variants have identical trainable location and scale projections; the fixed head adds no trainable degrees-of-freedom parameter. The value $8$ is a reference setting that is empirically sufficient in the distribution-head ablation, not a theoretically optimal or dataset-specific choice.

The evidence combines missing-history degradation, four probability-head seeds on fixed TimePerceiver checkpoints, a distribution-family ablation, a PatchTST replication, efficiency measurements, and retained negative results. Our contributions are: (i) a fixed-mask protocol for paired probabilistic forecasting under missing historical observations; (ii) controlled evidence that a matched-capacity fixed heavy-tailed head improves proper scores on the evaluated TimePerceiver conditions; and (iii) a transparent second-backbone assessment showing that probabilistic-score gains transfer while a small Electricity point-error tradeoff prevents a complete multi-metric conclusion.
""",
        "02_related_work.tex": r"""
\section{Related Work}
\label{sec:related-work}

\subsection{Probabilistic forecasting and proper scores}
DeepAR established autoregressive probabilistic forecasting at scale \citep{salinas2020deepar}; spline quantile functions and diffusion-based TimeGrad broadened the available forecast distributions \citep{gasthaus2019spline,rasul2021timegrad}. Proper-score methodology distinguishes distributional quality from point accuracy \citep{matheson1976crps,gneiting2007proper,gneiting2007calibration,gneiting2014probabilistic}. We report NLL and a sampled CRPS implementation, with interval scores and coverage--width diagnostics as complementary summaries \citep{winkler1972interval,jordan2019scoringrules}.

\subsection{Heavy-tailed predictive distributions}
The Student distribution originates from small-sample inference \citep{student1908} and has long been used for robust modeling because heavier tails reduce the penalty assigned to extreme residuals relative to a Gaussian model \citep{lange1989robust}. Our use is narrower: we replace only the marginal forecast likelihood while keeping trainable location and scale capacity matched. The work neither introduces a new distribution nor learns a separate tail parameter for the final method.

\subsection{Forecasting with incomplete histories}
GRU-D, BRITS, GP-VAE, CSDI, and SAITS represent influential approaches to irregular or missing time-series data \citep{che2018grud,cao2018brits,fortuin2020gpvae,tashiro2021csdi,du2023saits}. TimePerceiver directly motivates the backbone and missing-history setting studied here \citep{lee2025timeperceiver}. Our focus differs from imputation-only evaluation: future targets remain intact, and the forecast distribution is scored after fixed historical masking.

\subsection{Forecasting backbones, calibration, and comparison}
Informer, Autoformer, FEDformer, the Temporal Fusion Transformer, and PatchTST illustrate diverse sequence-modeling choices for long-horizon forecasting \citep{zhou2021informer,wu2021autoformer,zhou2022fedformer,lim2021tft,nie2023patchtst}. Post-hoc probabilistic calibration and conformal intervals offer complementary tools \citep{kuleshov2018calibrated,romano2019cqr,stankeviciute2021conformal}, although dependent time series require careful temporal splits. Finally, paired forecast comparisons should separate descriptive consistency from formal inferential testing \citep{diebold1995comparison,giacomini2006conditional}; the present study reports no formal significance test.
""",
        "03_forecasting_problem.tex": r"""
\section{Forecasting Problem}
\label{sec:forecasting-problem}

Let $x_{1:t}\in\mathbb{R}^{t\times C}$ denote a $C$-channel historical series observed up to forecast origin $t$, and let $y_{t+1:t+H}$ be the future target over horizon $H$. A binary historical observation mask $m_{1:t}$ marks available values. The model receives the masked historical input $\widetilde{x}_{1:t}$ together with the time covariates required by the backbone. Only historical inputs are masked; future targets are never masked or exposed to the model.

For every horizon $h$ and channel $c$, the probabilistic head estimates a predictive marginal distribution
\begin{equation}
p(y_{t+h,c}\mid\widetilde{x}_{1:t},m_{1:t}).
\label{eq:marginal-forecast}
\end{equation}
The point forecast is the predicted location. Central marginal prediction intervals are obtained from the corresponding distribution quantiles. These are horizon--channel marginals: the method does not estimate a joint covariance structure, a multivariate predictive density, or joint coverage.

The official chronological train, validation, and test splits are retained. Normalization statistics are fitted on training data only and reused for validation and test. Fixed mask assets define random, block, channel, and mixed missingness at rates $0.1$, $0.3$, $0.5$, and $0.7$ for the deterministic audit. Probability-head comparisons use paired fixed masks, preventing mask resampling from confounding distribution-family differences.
""",
        "04_probabilistic_method.tex": r"""
\section{Probabilistic Forecasting Method}
\label{sec:method}

\subsection{Matched-capacity marginal heads}
Given a frozen backbone representation $z_{h,c}$, two linear projections produce location $\mu_{h,c}$ and a positive scale $\sigma_{h,c}=\operatorname{softplus}(s_{h,c})+\epsilon$. Gaussian, Laplace, and fixed \StudentT{} heads use exactly these trainable projections. The fixed head is
\begin{equation}
y_{t+h,c}\sim \operatorname{StudentT}(\df=8,\mu_{h,c},\sigma_{h,c}).
\label{eq:student-head}
\end{equation}
Thus it adds no trainable degrees-of-freedom parameters. A learned-$\df$ version is included only as an ablation and is not the proposed final method.

The fixed value $\df=8$ is a reference degrees-of-freedom setting. It is empirically sufficient under the evaluated distribution-head protocol, not theoretically optimal, and was not selected separately for each dataset. One goal of the ablation is to ask whether extra tail flexibility is supported once location and scale capacity are controlled.

\subsection{Training objective and forecast summaries}
Probability heads minimize validation-selected NLL while the official backbone checkpoint remains fixed. No test observation or target is used for head selection. NLL is a distribution-sensitive proper score. CRPS is another proper score that combines calibration and sharpness; here it is estimated with 64 Monte Carlo samples using the fixed evaluation seed 52025, rather than reported as an exact analytic value. MSE and MAE evaluate the location forecast.

For marginal intervals, prediction interval coverage probability (PICP) and mean prediction interval width (MPIW) describe coverage and sharpness separately. Neither PICP alone nor width alone is sufficient. The interval score combines coverage penalties with width for each marginal interval \citep{winkler1972interval}. These diagnostics do not imply joint calibration.
""",
        "05_evaluation_design.tex": r"""
\section{Evaluation Design}
\label{sec:evaluation}

\subsection{Datasets and fixed missing-history protocol}
Experiments use ETTh1, Weather, and Electricity with input length 384 and prediction length 96. The official splits and preprocessing rules are preserved, normalization is fitted on training data only, and fixed validation and test masks are reused within paired comparisons. Historical inputs alone are masked. The test set is never used for training, hyperparameter selection, early stopping, or conformal calibration.

The deterministic audit covers four mask types and four rates. The probabilistic TimePerceiver and PatchTST comparisons use the corrected fixed-mask pipeline; invalid historical runs in which masks were not applied are excluded from every result. Head selection uses validation NLL. Gaussian and \StudentT{} heads within a pair use the same backbone checkpoint, mask, split, target, training budget, and evaluation code.

\subsection{Seed scope and decision criteria}
The TimePerceiver comparison uses four probability-head training seeds (2021, 2022, 2023, and 2025) on a fixed official backbone checkpoint per dataset. It is therefore a probability-head seed study, not an end-to-end backbone retraining study. The PatchTST comparison uses one backbone seed and one probability-head seed (2025) per dataset under random missingness at rate $0.3$.

The paired criteria require lower NLL, CRPS noninferiority, and point-prediction noninferiority. Point noninferiority permits a relative MSE increase of at most $1\%$:
\begin{equation}
\frac{\operatorname{MSE}_{t}-\operatorname{MSE}_{g}}{\operatorname{MSE}_{g}}\leq 0.01,
\label{eq:point-gate}
\end{equation}
where subscripts $t$ and $g$ denote fixed \StudentT{} and Gaussian heads. This tolerance was fixed before the second-backbone decision. Test results do not alter it. All reported comparisons are descriptive; no formal significance test is performed.
""",
        "06_empirical_results.tex": r"""
\section{Empirical Results}
\label{sec:results}

\subsection{Forecast degradation under missing history}
Across 48 dataset--mask--rate cells, deterministic error rises as more historical values are removed. Table~\ref{tab:fixed-mask-degradation} gives the representative matrix and Figure~\ref{fig:missingness-degradation} shows the complete rate trends; full tables appear in the supplement.

\begin{table}[t]
\centering
\caption{Deterministic forecast degradation under fixed historical masks. MSE is lower-is-better; relative degradation uses the corresponding clean TimePerceiver forecast as reference.}
\label{tab:fixed-mask-degradation}
\scriptsize
\resizebox{\linewidth}{!}{\input{tables/table2_fixed_mask_degradation}}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.96\linewidth]{fig1_missingness_degradation.pdf}
\caption{MSE under fixed historical-mask assets. Line styles and markers distinguish mask types in addition to color, and larger missing rates generally increase forecast error.}
\label{fig:missingness-degradation}
\end{figure}

\subsection{TimePerceiver paired proper-score results}
Across 12 dataset--seed pairs, the \StudentT{} variant improves NLL in 12/12 cases, is CRPS-noninferior in 12/12 cases, and satisfies point noninferiority in 12/12 cases. These are four probability-head seeds on a fixed backbone checkpoint for each dataset. Table~\ref{tab:timeperceiver-multiseed} reports dataset means and variability; Figure~\ref{fig:timeperceiver-nll} retains every paired NLL delta.

\begin{table}[t]
\centering
\caption{TimePerceiver results across four probability-head seeds on fixed backbone checkpoints. Deltas are fixed \StudentT{} minus Gaussian; negative NLL and CRPS deltas indicate improvement.}
\label{tab:timeperceiver-multiseed}
\scriptsize
\resizebox{\linewidth}{!}{\input{tables/table4_timeperceiver_multiseed}}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.96\linewidth]{fig2_multiseed_nll_deltas.pdf}
\caption{Paired TimePerceiver NLL deltas for all dataset--head-seed pairs. Negative values favor the fixed \StudentT{} forecast distribution.}
\label{fig:timeperceiver-nll}
\end{figure}

\subsection{Distribution-family ablation}
The matched-head ablation compares Gaussian, Laplace, fixed \StudentT{}, and learned-$\df$ \StudentT{} forecasts. Table~\ref{tab:distribution-ablation} and Figure~\ref{fig:distribution-ablation} support fixed heavy-tailed forecasting rather than learned degrees of freedom. Fixed $\df=8$ is sufficient under the evaluated protocol, whereas learning per-channel $\df$ provides no consistent additional benefit. The earlier learned-$\df$ results remain visible in the ablation.

\begin{table}[t]
\centering
\caption{Distribution-head ablation on frozen TimePerceiver checkpoints. Gaussian, Laplace, and fixed \StudentT{} share identical trainable location and scale capacity; lower NLL, CRPS, MSE, and MAE are better.}
\label{tab:distribution-ablation}
\scriptsize
\resizebox{\linewidth}{!}{\input{tables/table8_distribution_ablation}}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.88\linewidth]{fig4_student_t_df.pdf}
\caption{Distribution-head comparison under random historical missingness at rate $0.3$. The fixed $\df=8$ setting is a common reference value, not a dataset-specific selection.}
\label{fig:distribution-ablation}
\end{figure}

\subsection{PatchTST replication and point tradeoff}
PatchTST records NLL wins in 3/3 datasets, CRPS noninferiority in 3/3, point noninferiority in 2/3, and a full paired pass in 2/3. ETTh1 and Weather pass all paired criteria. For Electricity, Gaussian MSE is 0.165829 and fixed \StudentT{} MSE is 0.167809, a change of +0.001980 or +1.194\%. This exceeds the prespecified 1\% point-noninferiority threshold. Thus probabilistic-score gains transfer to the second evaluated backbone, but complete multi-metric consistency does not.

\begin{table}[t]
\centering
\caption{PatchTST fixed \StudentT{} minus Gaussian deltas. Negative NLL and CRPS values indicate improvement. Electricity's +1.194\% relative MSE change exceeds the 1\% point-noninferiority threshold.}
\label{tab:patchtst}
\scriptsize
\resizebox{\linewidth}{!}{\input{tables/table10_second_backbone}}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.96\linewidth]{fig_gate5b_cross_backbone_deltas.pdf}
\caption{Within-backbone metric deltas for TimePerceiver and PatchTST. Negative values favor fixed \StudentT{}; the positive PatchTST Electricity MSE delta is retained.}
\label{fig:cross-backbone}
\end{figure}

\subsection{Efficiency}
Fixed \StudentT{} has the same number of trainable head parameters as Gaussian and Laplace because $\df$ is not learned. Table~\ref{tab:efficiency} reports controlled timing and memory measurements. Comparisons are interpreted within each benchmark environment, not as hardware-independent throughput claims.

\begin{table}[t]
\centering
\caption{Controlled probability-head parameter, runtime, and memory measurements. Values are compared within dataset and benchmark environment.}
\label{tab:efficiency}
\scriptsize
\resizebox{\linewidth}{!}{\input{tables/table9_efficiency}}
\end{table}

\subsection{Tail-concentrated gains}
Residual-percentile decompositions place much of the NLL improvement in the most extreme Gaussian-residual bins. Figure~\ref{fig:tail-gains} presents both evaluated backbones. These diagnostics are descriptive and do not establish a causal mechanism or treat forecast elements as independent inferential samples.

\begin{figure}[t]
\centering
\includegraphics[width=0.48\linewidth]{fig_gate5a_tail_nll_gain.pdf}
\includegraphics[width=0.48\linewidth]{fig_gate5b_patchtst_tail_nll_gain.pdf}
\caption{Descriptive NLL differences across Gaussian-residual percentile bins for TimePerceiver (left) and PatchTST (right). Negative values favor fixed \StudentT{}.}
\label{fig:tail-gains}
\end{figure}
""",
        "07_diagnostics_and_negative_results.tex": r"""
\section{Diagnostics and Negative Results}
\label{sec:negative-results}

The retained negative results constrain the interpretation. On ETTh1 with random missingness at rate $0.3$, a matched-capacity static adapter improves the official forecast, but the tested missingness-aware FiLM adapter does not improve over that static adapter. Mechanism checks show nonconstant conditioning, nonzero FiLM modulation, finite gradients, and parameter updates. The result therefore concerns this global-statistics conditioning design and fixed condition; it does not establish that missingness conditioning is generally unhelpful.

Under a strict raw-time selection and calibration split, the evaluated additive split-conformal correction widens intervals without improving absolute coverage error. Native intervals are already near the nominal targets, while the calibration segment is descriptively harder. This is a result for one additive score, split, dataset, and missingness condition; it does not generalize to other conformal designs. Detailed tables and figures for both negative findings are in the supplement.
""",
        "08_discussion.tex": r"""
\section{Discussion}
\label{sec:discussion}

\subsection{Why heavy tails help}
The residual-bin analyses show that NLL gains are concentrated in extreme Gaussian-residual regions on both evaluated backbones. This pattern is consistent with a heavier-tailed forecast assigning less implausibly small density to large errors. It is descriptive, not causal proof: the experiments do not manipulate tail behavior independently of the likelihood family.

\subsection{Why a fixed degree of freedom can suffice}
A fixed $\df$ provides robust tails without adding learned shape parameters. One possible explanation for the learned-$\df$ result is that per-channel flexibility may add estimation variance or be weakly identified under the available head-training data. This remains a hypothesis. The evidence only shows that learned $\df$ did not yield a consistent additional gain in this ablation.

\subsection{Proper-score versus point tradeoff}
The PatchTST Electricity case illustrates why distributional and point objectives must be reported together. Better NLL and CRPS can coexist with a small MSE deterioration because the proper scores evaluate the predictive distribution while MSE evaluates its center. The +1.194\% deterioration exceeds the fixed 1\% tolerance, so it is a genuine protocol-level tradeoff rather than a rounding artifact.

\subsection{Practical interpretation}
The supported use case is a forecasting system that requires marginal predictive uncertainty under incomplete historical inputs and can retain paired fixed-mask evaluation. Fixed \StudentT{} is attractive here because it changes the likelihood without increasing trainable head capacity. This interpretation does not imply a particular business deployment, joint uncertainty, or invariant point performance across architectures.
""",
        "09_limitations.tex": r"""
\section{Limitations}
\label{sec:limitations}

The evidence covers three datasets and two backbones with fixed historical masks. Some conditions are single-seed. TimePerceiver varies four probability-head seeds while holding the backbone checkpoint fixed, and PatchTST uses one backbone seed and one head seed under random missingness at rate $0.3$ only. PatchTST Electricity has a point-prediction tradeoff. No formal significance test is reported.

The predictive distributions are marginal by horizon and channel. We do not model joint covariance, assess joint coverage, or report a multivariate energy score. The fixed $\df=8$ value is not theoretically optimal, and the study does not perform dataset-specific tail selection. The conformal study covers a limited additive split procedure, and the FiLM study covers a limited global conditioning design.

The experiments do not evaluate decision-specific economic loss, domain-specific forecast value, density forecast combination, or operational consequences. These are untested extensions rather than implied benefits. Broader missingness shifts, more independently trained backbones, and alternative tail models are needed before stronger generalization claims are warranted.
""",
        "10_conclusion.tex": r"""
\section{Conclusion}
\label{sec:conclusion}

This study evaluates a matched-capacity fixed \StudentT{} forecast head under fixed masks applied only to historical observations. Proper probabilistic scores improve consistently across the evaluated TimePerceiver probability-head seeds, and a distribution ablation supports fixed heavy tails over learned degrees of freedom in this setting. PatchTST reproduces the NLL and CRPS direction on all three datasets but retains a small Electricity MSE tradeoff that exceeds the prespecified tolerance. The resulting conclusion is deliberately bounded: fixed heavy-tailed predictive marginals are a lightweight way to improve probabilistic forecast quality under the evaluated missing-history protocol, but point behavior and broader generalization must be checked for each backbone and condition.
""",
    }
    for name, text in sections.items():
        write(MANUSCRIPT / "sections" / name, text)


def make_supplement() -> None:
    write(MANUSCRIPT / "appendix/appendix.tex", r"""
\input{../manuscript/appendix/full_missingness_results}
\input{../manuscript/appendix/probabilistic_diagnostics}
\input{../manuscript/appendix/additional_negative_results}
\input{../manuscript/appendix/protocol_and_reproducibility}
""")
    write(MANUSCRIPT / "appendix/full_missingness_results.tex", r"""
\section{Full Fixed-Mask Matrix}
\label{supp:fixed-mask}

The following tables retain all fixed-mask deterministic evaluations by dataset. Only historical inputs are masked; targets are unchanged.

\begin{table}[ht]\centering\caption{Full ETTh1 fixed-mask matrix.}\label{tab:supp-etth1}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/tableA1_etth1}}\end{table}
\begin{table}[ht]\centering\caption{Full Weather fixed-mask matrix.}\label{tab:supp-weather}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/tableA1_weather}}\end{table}
\begin{table}[ht]\centering\caption{Full Electricity fixed-mask matrix.}\label{tab:supp-electricity}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/tableA1_electricity}}\end{table}

\section{TimePerceiver Seed-Level Summaries}
\label{supp:timeperceiver-seeds}
The four probability-head seeds are 2021, 2022, 2023, and 2025 on fixed official backbone checkpoints. The panels summarize the dataset-level means and variability; the main manuscript preserves all paired NLL deltas.
\begin{figure}[ht]\centering
\includegraphics[width=0.32\linewidth]{fig3a_multiseed_nll.pdf}
\includegraphics[width=0.32\linewidth]{fig3b_multiseed_crps.pdf}
\includegraphics[width=0.32\linewidth]{fig3c_multiseed_mse.pdf}
\caption{TimePerceiver probability-head seed summaries for NLL, CRPS, and MSE.}\label{fig:supp-multiseed}
\end{figure}
\begin{table}[ht]\centering\caption{ETTh1 paired results across additional missing-history conditions. Negative deltas favor fixed \StudentT{}.}\label{tab:supp-etth1-conditions}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/table5_etth1_cross_condition}}\end{table}
""")
    write(MANUSCRIPT / "appendix/probabilistic_diagnostics.tex", r"""
\section{Distribution-Head Ablation Details}
\label{supp:distribution}
The ablation retains Gaussian, Laplace, fixed-$\df$ \StudentT{}, and learned-$\df$ \StudentT{} results. The learned variant is visible even though it is not the final method.
\begin{figure}[ht]\centering
\includegraphics[width=0.48\linewidth]{fig_gate5a_kurtosis_df.pdf}
\includegraphics[width=0.48\linewidth]{fig4_student_t_df.pdf}
\caption{Descriptive channel-tail diagnostics and distribution-head comparison.}\label{fig:supp-df}
\end{figure}

\section{PIT and QQ Diagnostics}
\label{supp:pit-qq}
These probability integral transform and quantile plots are descriptive diagnostics; they are not formal evidence of joint calibration.
\begin{figure}[ht]\centering
\includegraphics[width=0.48\linewidth]{fig_gate5a_pit_histograms.pdf}
\includegraphics[width=0.48\linewidth]{fig_gate5a_qq.pdf}
\caption{PIT histograms and quantile diagnostics for the distribution ablation.}\label{fig:supp-pit-qq}
\end{figure}

\section{PatchTST Detailed Results}
\label{supp:patchtst}
The second-backbone study uses one backbone seed and one probability-head seed per dataset. ETTh1 and Weather pass the full paired criteria; Electricity preserves NLL and CRPS improvements but exceeds the 1\% point-error tolerance.
\begin{figure}[ht]\centering\includegraphics[width=0.82\linewidth]{fig_gate5b_patchtst_tail_nll_gain.pdf}\caption{Descriptive PatchTST tail-bin NLL decomposition.}\label{fig:supp-patchtst-tail}\end{figure}
""")
    write(MANUSCRIPT / "appendix/additional_negative_results.tex", r"""
\section{Adapter Ablation}
\label{supp:adapter}
The static adapter improves on the official reference in the evaluated ETTh1 condition, while global-statistics FiLM does not improve over matched static capacity.
\begin{table}[ht]\centering\caption{Static and missingness-aware adapter comparison on ETTh1 random-0.3.}\label{tab:supp-adapter}\scriptsize\input{../manuscript/tables/table3_adapter_ablation}\end{table}
\begin{figure}[ht]\centering\includegraphics[width=0.78\linewidth]{figA1_gate4a_adapter.pdf}\caption{Adapter MSE comparison.}\label{fig:supp-adapter}\end{figure}

\section{Strict Conformal Negative Result}
\label{supp:conformal}
The additive correction uses a temporally separated calibration set and cannot shrink intervals. It widens the already near-nominal native intervals and does not improve absolute coverage error under this protocol.
\begin{table}[ht]\centering\caption{Formal additive conformal results.}\label{tab:supp-conformal}\scriptsize\input{../manuscript/tables/table6_conformal}\end{table}
\begin{table}[ht]\centering\caption{Conformal mechanism audit; counterfactual rows are diagnostic only.}\label{tab:supp-conformal-mechanism}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/table7_conformal_mechanism}}\end{table}
\begin{table}[ht]\centering\caption{All retained probability heads and interval procedures.}\label{tab:supp-conformal-all}\scriptsize\resizebox{\linewidth}{!}{\input{../manuscript/tables/tableA2_conformal_all_heads}}\end{table}
\begin{figure}[ht]\centering
\includegraphics[width=0.48\linewidth]{fig5_conformal_tradeoff.pdf}
\includegraphics[width=0.48\linewidth]{fig6_calibration_test_shift.pdf}
\caption{Interval-width tradeoff and calibration-to-test coverage shift.}\label{fig:supp-conformal}
\end{figure}
\begin{figure}[ht]\centering
\includegraphics[width=0.48\linewidth]{figA2_gate4c_channel_coverage.pdf}
\includegraphics[width=0.48\linewidth]{figA3_gate4c_horizon_coverage.pdf}
\caption{Channel- and horizon-level marginal coverage diagnostics.}\label{fig:supp-coverage}
\end{figure}
""")
    write(MANUSCRIPT / "appendix/protocol_and_reproducibility.tex", r"""
\section{Protocol and Reproducibility}
\label{supp:protocol}
The chronological splits, train-only normalization statistics, fixed validation masks, fixed test masks, metric implementations, evaluation seeds, configuration files, and checkpoint hashes are retained in the project evidence registry. Input length is 384, prediction length is 96, and CRPS uses 64 Monte Carlo samples with fixed evaluation seed 52025. Test masks are never used during head training, and test labels are never used for model selection.

For traceability, the project stages map to the evidence as follows: the fixed missing-history assets and deterministic evaluation precede the adapter study; probabilistic-head validation and multi-seed fixed-backbone comparisons establish the heavy-tail signal; the distribution ablation refines the final fixed-$\df$ specification; the conformal study records a negative result; and PatchTST provides the second-backbone assessment. Historical probability-head runs that did not actually apply the input mask were marked invalid and excluded without reporting their metrics.

\section{Source and Mask Integrity}
\label{supp:integrity}
Every formal result is linked to a unique experiment record, resolved configuration, checkpoint hash, fixed-mask hash, and metric-protocol hash. The venue adaptation is a document-only transformation of the frozen evidence: it performs no training, inference, metric recomputation, mask generation, or checkpoint modification. Exact repository URLs and anonymous-review release details remain an author action.
""")

    write(SUPPLEMENT / "supplementary.tex", r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,array,booktabs,graphicx,longtable}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\newcommand{\citep}[1]{\cite{#1}}
\makeatletter
\renewcommand{\@cite}[2]{(#1\if@tempswa, #2\fi)}
\makeatother
\input{../manuscript/macros}
\setlength{\emergencystretch}{4em}
\graphicspath{{../manuscript/figures/}}
\begin{document}
\title{Supplementary Material for ``\PaperTitle''}
\author{[AUTHOR INPUT REQUIRED]}
\date{}
\maketitle
\input{../manuscript/appendix/appendix}
\input{../manuscript/references_list}
\end{document}
""")


def make_administrative_files() -> None:
    title = "Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations"
    short = "Heavy-Tailed Forecasts with Missingness"
    write(SUBMISSION / "title_page.md", f"""
# {title}

**Short title:** {short}

| Field | Required entry |
| --- | --- |
| Authors | [AUTHOR INPUT REQUIRED] |
| Affiliations | [AUTHOR INPUT REQUIRED] |
| Corresponding author | [AUTHOR INPUT REQUIRED] |
| Email | [AUTHOR INPUT REQUIRED] |
| ORCID | [AUTHOR INPUT REQUIRED] |
| Funding | [FUNDING INFORMATION REQUIRED] |
| Acknowledgements | [AUTHOR INPUT REQUIRED] |
| Conflict of interest | [AUTHOR INPUT REQUIRED] |
| Data availability | Public benchmark datasets; sources and preprocessing are documented in the supplement. |
| Code availability | [REPOSITORY URL REQUIRED; AUTHOR ACTION REQUIRED] |
| Author contributions | [AUTHOR INPUT REQUIRED] |
""")
    write(SUBMISSION / "title_page.tex", r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{4em}
\begin{document}
\begin{center}
{\Large\bfseries Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations\par}
\vspace{0.5em}
{\small Running title: Heavy-Tailed Forecasts with Missingness\par}
\end{center}
\begin{description}
\item[Authors] [AUTHOR INPUT REQUIRED]
\item[Affiliations] [AUTHOR INPUT REQUIRED]
\item[Corresponding author] [AUTHOR INPUT REQUIRED]
\item[Email and ORCID] [AUTHOR INPUT REQUIRED]
\item[Funding] [FUNDING INFORMATION REQUIRED]
\item[Acknowledgements] [AUTHOR INPUT REQUIRED]
\item[Conflict of interest] [AUTHOR INPUT REQUIRED]
\item[Data availability] Public benchmark datasets; sources and preprocessing are documented in the supplement.
\item[Code availability] [REPOSITORY URL REQUIRED; AUTHOR ACTION REQUIRED]
\item[Author contributions] [AUTHOR INPUT REQUIRED]
\end{description}
\end{document}
""")
    write(SUBMISSION / "cover_letter_draft.md", f"""
# Cover Letter Draft

Dear Editor,

Please consider the Research Article "{title}" for Journal of Forecasting. The manuscript studies probabilistic time-series forecasting when observations are missing from the historical input. It contributes a fixed-mask paired evaluation protocol and a matched-capacity fixed Student-t forecast head, with negative log-likelihood and continuous ranked probability score as the central proper-score evidence.

Across four probability-head seeds on fixed TimePerceiver backbones, the fixed Student-t head improves NLL in 12/12 dataset-seed pairs while satisfying CRPS and point-prediction noninferiority in 12/12. A controlled distribution-family ablation supports fixed heavy tails without learned degrees-of-freedom parameters. A PatchTST assessment preserves NLL and CRPS improvements in 3/3 datasets but transparently reports that Electricity exceeds the prespecified 1% point-error tolerance. This distinction between distributional quality and point accuracy, together with the paper's calibration-and-sharpness analysis, fits the journal's focus on forecasting methodology and evaluation.

[AUTHOR CONFIRMATION REQUIRED: This manuscript is not under consideration elsewhere.]

All authors have approved the manuscript and the declarations: [AUTHOR CONFIRMATION REQUIRED].

Sincerely,

[CORRESPONDING AUTHOR INPUT REQUIRED]
""")
    write(SUBMISSION / "cover_letter_draft.tex", r"""
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\setlength{\emergencystretch}{4em}
\setlength{\parskip}{0.8em}
\setlength{\parindent}{0pt}
\begin{document}
\thispagestyle{empty}
\noindent Editor\\
\emph{Journal of Forecasting}

\vspace{1em}
\noindent Dear Editor,

Please consider the Research Article ``Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations'' for the \emph{Journal of Forecasting}. The manuscript studies probabilistic time-series forecasting when observations are missing from the historical input. It contributes a fixed-mask paired evaluation protocol and a matched-capacity fixed Student-$t$ forecast head, with negative log-likelihood and continuous ranked probability score as the central proper-score evidence.

Across four probability-head seeds on fixed TimePerceiver backbones, the fixed Student-$t$ head improves NLL in 12/12 dataset--seed pairs while satisfying CRPS and point-prediction noninferiority in 12/12. A controlled distribution-family ablation supports fixed heavy tails without learned degrees-of-freedom parameters. A PatchTST assessment preserves NLL and CRPS improvements in 3/3 datasets but transparently reports that Electricity exceeds the prespecified 1\% point-error tolerance. This distinction between distributional quality and point accuracy, together with the calibration-and-sharpness analysis, fits the journal's focus on forecasting methodology and evaluation.

\textbf{[AUTHOR CONFIRMATION REQUIRED: This manuscript is not under consideration elsewhere.]}

All authors have approved the manuscript and declarations: \textbf{[AUTHOR CONFIRMATION REQUIRED].}

\noindent Sincerely,\\
\mbox{[CORRESPONDING AUTHOR INPUT REQUIRED]}
\end{document}
""")
    write(SUBMISSION / "submission_checklist.md", """
# Journal of Forecasting Submission Checklist

## Ready

- [x] Manuscript PDF
- [x] Anonymous manuscript PDF
- [x] LaTeX source archive
- [x] Vector figures
- [x] Tables
- [x] Supplementary PDF and source
- [x] Complete verified bibliography

## Author Input Required

- [ ] Author names
- [ ] Affiliations
- [ ] Corresponding author
- [ ] Email addresses
- [ ] ORCID identifiers
- [ ] Funding information
- [ ] Author contributions
- [ ] Acknowledgements
- [ ] Conflict-of-interest confirmation
- [ ] Code repository URL and release timing
- [ ] AI-use disclosure approval, including tool versions and dates
- [ ] Not-under-review confirmation
- [ ] Reviewer suggestions or opposed reviewers if requested

## Manual Portal Verification

- [ ] Exact Research Article option
- [ ] Cover-letter requirement
- [ ] Data/code statement fields
- [ ] AI disclosure field
- [ ] Supplement upload category
- [ ] LaTeX source upload timing
- [ ] Current APC and open-access options
""")


def main() -> None:
    frozen = verify_frozen_sources()
    for directory in (MANUSCRIPT, SUBMISSION, SUPPLEMENT, DECLARATIONS, METADATA, CHECKS, ARCHIVE):
        directory.mkdir(parents=True, exist_ok=True)
    copy_manuscript_assets()
    make_guideline_audit()
    make_manuscript()
    make_supplement()
    make_administrative_files()
    payload = {
        "status": "PREPARED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_venue": "Journal of Forecasting",
        "frozen_source_count": len(frozen),
        "venue_root": "paper/venues/journal_of_forecasting",
        "mother_manuscript_modified": False,
        "gate7a_package_modified": False,
        "training": False,
        "inference": False,
        "gpu": False,
    }
    write(CHECKS / "preparation.json", json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
