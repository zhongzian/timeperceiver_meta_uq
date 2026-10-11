# Final Project Synthesis

## Evidence boundary

Gate 0 through Gate 3b pass. Gate 4a establishes a positive static-adapter capacity signal but a negative additional signal for the tested missingness-aware FiLM design. Gate 4b establishes the primary positive result: corrected fixed-mask Student-t probabilistic forecasting passes expanded-condition, cross-dataset, and four-seed probability-head confirmation. Gate 4c passes implementation and leakage guards but produces a negative strict additive conformal result; its mechanism audit is complete. Gate 5b completes a PatchTST replication with a mixed point-tradeoff decision.

## Main result

The primary contribution is `STUDENT_T_PROBABILISTIC_FORECASTING`. In the formal multi-seed matrix, Student-t improves NLL in 12/12 paired cells, meets CRPS noninferiority in 12/12, meets point noninferiority in 12/12, and passes dataset-level mean gates in 3/3 datasets. The comparison uses fixed official backbone checkpoints and four probability-head training seeds, not full end-to-end multi-seed training.

On PatchTST, fixed-t8 improves NLL and CRPS in 3/3 datasets. Point noninferiority passes in ETTh1 and Weather but fails on Electricity because MSE rises by 1.194%, slightly beyond the pre-registered 1% tolerance. Consequently, `GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE=False`; the probabilistic-score direction is supported, while the complete multi-metric gate is not.

## Negative results retained

The tested global-statistics FiLM conditioning does not improve over the static adapter. Under the strict Gate 4c raw-time split, additive marginal and mask-severity-conditioned conformal corrections widen intervals and worsen absolute coverage error relative to native intervals. The PatchTST Electricity point tradeoff is also retained. None of these results is removed or reinterpreted as fully positive.

## Scope

The evidence covers ETTh1, Weather, and Electricity. The second-backbone experiment covers only PatchTST, random missing rate 0.3, and seed 2025. The project models independent marginal predictive distributions, does not establish joint coverage, and makes no formal significance or broader-generalization claim. The historical unmasked Gate 4b minimal runs remain auditable but contribute no paper metric, table, figure, claim, or reproduction checkpoint.

## Artifact status

Canonical registry, source manifest, metric consistency, tables, figures, claims, and narrative sections passed final paper-artifact verification. The synthesis status is `PAPER_READY_EVIDENCE_SYNTHESIS_COMPLETE`.
