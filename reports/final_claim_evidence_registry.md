# Final Claim-Evidence Registry

Claims: 9; supported: 9; unsupported: 0.

## C1

The official TimePerceiver baseline was reproduced on ETTh1, Weather, and Electricity under a unified project protocol.

- Supported: True
- Strength: descriptive
- Scope: ETTh1, Weather, and Electricity; official project protocol; one official training seed per dataset.
- Sources: reports/latest_results.csv;reports/gate2_paper_comparison.csv
- Required qualifier: Project reproduction; agreement with every paper value is not claimed.
- Prohibited wording: Exact reproduction of every published result.
- Limitations: One official training seed per dataset.

## C2

Fixed historical-input missingness degrades deterministic forecasting, with stronger degradation at higher missing rates.

- Supported: True
- Strength: descriptive
- Scope: Three datasets, four mask types, four missing rates, and one fixed test-mask seed per condition.
- Sources: reports/gate3b_full_results.csv;reports/gate3b_degradation_summary.csv
- Required qualifier: Single fixed test-mask seed; historical inputs only.
- Prohibited wording: All missingness processes always cause the same degradation.
- Limitations: Three datasets, four tested mask types, and four rates.

## C3

A small frozen-backbone adapter improves ETTh1 random-0.3 forecasting, but the tested global-statistics FiLM conditioning does not improve over a same-capacity static adapter.

- Supported: True
- Strength: scoped comparative
- Scope: ETTh1 random missing rate 0.3, seed 2025, and the tested global-statistics single-FiLM design.
- Sources: reports/gate4a_minimal_results.csv;results/gate4a_formal_decision.json;results/gate4a_mechanism_audit.json
- Required qualifier: ETTh1 random 0.3, seed 2025, current global-statistics single-FiLM design.
- Prohibited wording: Missingness-aware adapters are generally ineffective.
- Limitations: Single condition and seed.

## C4

The Student-t head consistently improves NLL and CRPS over a Gaussian head across three datasets and four probability-head training seeds, while satisfying the predefined point-prediction noninferiority criterion.

- Supported: True
- Strength: primary comparative
- Scope: Three datasets, four probability-head training seeds, fixed official backbones, and 12 paired cells.
- Sources: reports/gate4b_multiseed_results.csv;reports/gate4b_multiseed_paired_deltas.csv;results/gate4b_multiseed_formal_decision.json
- Required qualifier: Fixed official backbone; probability-head training seeds; 12 paired cells.
- Prohibited wording: Full end-to-end backbone-seed evidence or formal significance inference.
- Limitations: Four head-training seeds and fixed validation/test masks.

## C5

The fixed-df Student-t head uses the same trainable parameter count as the matched Gaussian and Laplace heads.

- Supported: True
- Strength: implementation fact
- Scope: Matched frozen-backbone probability heads in the Gate 5a and Gate 5b comparisons.
- Sources: reports/gate5a_distribution_results.csv;reports/gate5a_efficiency.csv;reports/gate5b_patchtst_probability_results.csv
- Required qualifier: Fixed df=8 introduces no trainable df parameter; all compared heads retain matched location-scale capacity.
- Prohibited wording: No computational overhead.
- Limitations: Runtime and distributional assumptions still differ even when trainable counts match.

## C6

The Gate 5a ablation supports fixed df=8 as sufficient for the evaluated three-dataset comparison; learned per-channel df does not provide consistent additional value.

- Supported: True
- Strength: scoped comparative
- Scope: TimePerceiver, ETTh1/Weather/Electricity, random missing rate 0.3, and one probability-head seed.
- Sources: reports/gate5a_pairwise_deltas.csv;results/gate5a_formal_decision.json
- Required qualifier: Empirical sufficiency in the evaluated scope, not a theoretical optimum.
- Prohibited wording: A data-independent optimum or a necessity for learned degrees of freedom.
- Limitations: Single probability-head seed with fixed official backbones.

## C7

Under the evaluated strict raw-time split, additive split conformal calibration widened intervals but did not improve absolute coverage error over native intervals.

- Supported: True
- Strength: scoped negative result
- Scope: ETTh1 random missing rate 0.3 under the strict raw-time split and additive score.
- Sources: reports/gate4c_interval_metrics.csv;results/gate4c_formal_decision.json
- Required qualifier: Additive score, current split, ETTh1 random 0.3, and time-series dependence.
- Prohibited wording: Conformal prediction does not work.
- Limitations: No exact exchangeability guarantee is claimed.

## C8

The negative conformal result is associated with a harder calibration segment, already well-calibrated native test intervals, and an additive-only correction that cannot shrink over-covered intervals.

- Supported: True
- Strength: mechanism diagnosis
- Scope: Gate 4c diagnostic split and counterfactual analyses; descriptive only.
- Sources: results/gate4c_mechanism_audit.json;reports/gate4c_mechanism_native_shift.csv;reports/gate4c_mechanism_counterfactual_shrinkage.csv
- Required qualifier: Descriptive distribution shift; counterfactual shrinkage is not formal calibration.
- Prohibited wording: The test-label oracle is a valid calibrator.
- Limitations: Mechanism audit is diagnostic and does not alter the formal result.

## C9

Across the two evaluated backbones, the probabilistic-score direction is consistent: on PatchTST, fixed-t8 improves NLL and CRPS on all three datasets, while the complete multi-metric gate remains negative because Electricity misses the predefined 1% point-prediction tolerance.

- Supported: True
- Strength: scoped mixed replication result
- Scope: PatchTST random missing rate 0.3, seed 2025, one clean backbone checkpoint, and three datasets.
- Sources: reports/gate5b_patchtst_probability_results.csv;reports/gate5b_patchtst_paired_deltas.csv;results/gate5b_formal_decision.json
- Required qualifier: PatchTST, three datasets, random missing rate 0.3, single backbone/head seed 2025, no significance claim.
- Prohibited wording: Metric consistency across every criterion or beyond the two evaluated architectures.
- Limitations: The pre-registered second-backbone signal gate is negative because Electricity exceeds the 1% MSE tolerance.
