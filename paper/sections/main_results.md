# Main Results

## 1. Clean Official TimePerceiver Baselines

The unified clean-input reproduction yielded ETTh1 MSE 0.374462, MAE 0.398943; Weather MSE 0.144059, MAE 0.199637; Electricity MSE 0.125364, MAE 0.220123. These are project reproduction values, not a claim of exact agreement with every published table.

## 2. Fixed-Mask Degradation

All 48 Gate 3b evaluations use immutable test masks and the same official deterministic checkpoints. Across the 12 dataset-by-mask-type curves, MSE increases monotonically with the evaluated missing rate. Table 2 presents representative conditions and Table A1 retains the full matrix. This evidence uses one fixed test-mask seed per condition.

## 3. TimePerceiver Student-t Multi-Seed Results

The primary matrix contains 12 paired dataset-by-seed cells and 24 probability-head runs. Student-t wins NLL in 12/12 cells, meets CRPS noninferiority in 12/12, meets point-prediction noninferiority in 12/12, and passes all paired gates in 12/12. All three dataset-level means pass:

- ETTh1: NLL 0.915014 to 0.871113, CRPS 0.342011 to 0.337656, and MSE 0.416862 to 0.415611.
- Weather: NLL 0.577226 to 0.103670, CRPS 0.244788 to 0.207544, and MSE 0.210306 to 0.209818.
- Electricity: NLL 0.439644 to 0.365348, CRPS 0.209365 to 0.207395, and MSE 0.167419 to 0.167703.

These are four probability-head training seeds with fixed official backbones and fixed validation/test masks. They support repeatability of the probability head, not full end-to-end backbone retraining or formal significance inference.

## 4. Distribution-Head Ablation

The matched Gaussian, Laplace, fixed-t8, and learned-df Student-t comparison uses one probability-head seed on each of the three TimePerceiver checkpoints. Fixed-t8 records 3/3 NLL wins, 3/3 CRPS-noninferior outcomes, and 3/3 point-noninferior outcomes relative to Gaussian. Learned per-channel df does not provide consistent additional value over the fixed form. The formal ablation decision is `FIXED_HEAVY_TAIL_SUFFICIENT`: fixed df=8 is empirically sufficient in this scope and avoids trainable df parameters.

## 5. PatchTST Second-Backbone Replication

The matched PatchTST study records 3/3 NLL wins, 3/3 CRPS improvements/noninferiority outcomes, 2/3 point-noninferiority outcomes, and 2/3 full-pair passes. Electricity fixed-t8 MSE is 0.167809 versus 0.165829 for Gaussian, a 1.194% increase that exceeds the pre-registered 1% tolerance. The decision is `SECOND_BACKBONE_POINT_TRADEOFF`. The probabilistic-score direction transfers, but the complete multi-metric gate does not.

## 6. Final Synthesis

The strongest evidence supports fixed heavy-tailed probabilistic forecasting under missing historical observations. TimePerceiver provides the complete three-dataset, four-head-seed result, and the fixed-df ablation shows that df=8 captures the useful effect without extra trainable parameters. PatchTST confirms the NLL and CRPS direction on all three datasets while exposing a small Electricity point-prediction tradeoff. Thus, the probabilistic-score result extends across the two evaluated backbones, whereas complete metric-wise consistency does not.
