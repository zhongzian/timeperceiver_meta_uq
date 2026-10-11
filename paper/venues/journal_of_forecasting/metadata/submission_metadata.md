# Submission Metadata

| Field | Value |
| --- | --- |
| Journal | Journal of Forecasting |
| Article type | Research Article |
| Submission mode | Free Format |
| Review model | Single-blind |
| Title | Fixed Heavy-Tailed Probabilistic Forecasting under Missing Historical Observations |
| Short title | Heavy-Tailed Forecasts with Missingness |
| Authors | [AUTHOR INPUT REQUIRED] |
| Affiliations | [AUTHOR INPUT REQUIRED] |
| Corresponding author | [AUTHOR INPUT REQUIRED] |
| Email and ORCID | [AUTHOR INPUT REQUIRED] |
| Suggested classifications | Probabilistic forecasting; forecast evaluation; missing data |
| Reviewer suggestions | [AUTHOR INPUT REQUIRED IF REQUESTED] |
| Opposed reviewers | [AUTHOR INPUT REQUIRED IF REQUESTED] |

## Abstract

Forecasting systems often require predictive uncertainty as well as point forecasts, yet incomplete historical observations change the error distribution used to estimate that uncertainty. Comparisons are additionally weakened when competing methods see different missingness realizations. We study probabilistic time-series forecasting under fixed masks applied only to historical inputs, with train-only normalization and paired evaluation assets. A lightweight distribution head represents each horizon-channel marginal with a fixed Student-t distribution with degrees of freedom 8, using the same trainable location and scale projections as Gaussian and Laplace alternatives. Proper probabilistic scores are central: negative log-likelihood and a Monte Carlo estimate of the continuous ranked probability score are reported alongside point-prediction noninferiority. On fixed TimePerceiver checkpoints, Student-t improves negative log-likelihood in 12/12 dataset-seed pairs and is noninferior for both continuous ranked probability score and point prediction in 12/12. A distribution-head ablation supports fixed heavy tails over learned degrees of freedom in the evaluated protocol. On PatchTST, probabilistic-score gains appear in 3/3 datasets, while Electricity exceeds the prespecified point-error tolerance. The results support fixed heavy-tailed predictive distributions as a lightweight method for improving probabilistic forecast quality under missing historical observations, while the PatchTST Electricity result cautions against assuming metric-wise invariance across backbones.

## Keywords

probabilistic forecasting; time-series forecasting; missing observations; Student-t distribution; proper scoring rules; heavy-tailed errors

## Statements

- Data availability: The datasets are publicly available from their original repositories; sources and preprocessing are documented in the supplement.
- Code availability: [AUTHOR ACTION REQUIRED -- repository URL and release timing]
- Conflict of interest: [AUTHOR INPUT REQUIRED]
- Funding: [FUNDING INFORMATION REQUIRED]
- Author contributions: [AUTHOR INPUT REQUIRED]
- AI-use disclosure: [AUTHOR CONFIRMATION REQUIRED]
