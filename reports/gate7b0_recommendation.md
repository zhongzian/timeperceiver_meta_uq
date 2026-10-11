# Gate 7B0 Recommendation

## Recommended primary venue

- Venue: **Journal of Forecasting**
- Total score: **87/100**
- Risk level: **MODERATE**
- Why it fits: A multidisciplinary forecasting journal covering theoretical, practical, computational, and methodological forecasting work across statistics, economics, psychology, systems engineering, and the social sciences.
- Main weaknesses: MODERATE: a fixed Student-t head needs a stronger statistical and forecasting rationale because it offers no new distribution or theory. MODERATE: three datasets and two backbones are useful, but the forecasting audience may ask for broader series diversity, stronger statistical baselines, or clearer heavy-tail diagnostics.
- Required manuscript changes:
  - Lead with the forecasting problem, proper-score evidence, heavy-tail rationale, and consequences of missing histories.
  - Strengthen statistical interpretation of fixed df=8 and make the learnable-df negative ablation central to the design argument.
  - Discuss heavy-tail diagnostics and the limits of three datasets and two backbones without adding unsupported claims.
  - Prepare the author title page and Wiley declarations after the venue is confirmed; preserve the venue-neutral source until Gate 7B1.
- Manual checks remaining:
  - JCR, Scopus, and CAS categories and quartiles
  - Current optional open-access APC
  - Exact submission-portal article type and data/code statement wording
  - Whether the current figure and table inventory should move partly to supporting information

## Backup venues

### Backup 1: Neural Computing and Applications

Choose when: the author prefers explicit neural-computing and benchmark scope over a forecasting-specialist readership.
Main advantage: Covers practical neural computing and related techniques and explicitly lists benchmarks, intelligent forecasting, machine learning, performance measures, and applied statistics. Original Articles should present new, significant research, development, or applications with practical value.
Main risk: MODERATE: the paper must distinguish a principled heavy-tail robustness study from a routine output-layer substitution.
Adaptation effort: MODERATE

### Backup 2: Applied Intelligence

Choose when: the author wants a broad applied-intelligence audience and can foreground the protocol as an intelligent-systems contribution.
Main advantage: GOOD: recent official articles use predictive distributions, CRPS, calibration, and interval metrics.
Main risk: HIGH: fixed Student-t df=8 is an empirical design choice, not a new distribution, theoretical result, or backbone.
Adaptation effort: MODERATE_TO_HIGH

## Venues not recommended

- **Engineering Applications of Artificial Intelligence**: VERY_HIGH: the current benchmark manuscript does not identify a real-world engineering application, which is an explicit journal requirement.

## Submission cascade

Journal of Forecasting -> Neural Computing and Applications -> Applied Intelligence

Submit sequentially, never simultaneously.

## Decision boundary

Gate 7B0 recommends a venue. It does not submit or adapt the manuscript.
