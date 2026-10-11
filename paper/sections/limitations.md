# Limitations

1. TimePerceiver multi-seed experiments vary probability-head seeds, not full end-to-end backbone seeds.
2. PatchTST uses one clean-backbone seed and one probability-head seed.
3. PatchTST evaluates only random historical missingness at rate 0.3.
4. The complete PatchTST gate is negative because Electricity MSE increases by approximately 1.194%, above the pre-registered 1% tolerance.
5. Only two backbones and three datasets are evaluated.
6. The fixed Student-t head models marginal predictive distributions, not joint horizon/channel covariance.
7. Native marginal interval calibration does not imply joint coverage.
8. No formal significance claim is made.
9. Fixed df=8 is supported empirically but is not established as a theoretical optimum.
10. Conformal findings are specific to the evaluated additive split-conformal protocol and ETTh1 random missing rate 0.3.
11. Fixed validation and test masks improve pairing but do not sample the full range of possible missingness realizations.
12. Gate 4a evaluates only the tested global-statistics, single-point FiLM design and does not rule out other conditioning structures.
