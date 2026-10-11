# Second-Backbone Replication

## Protocol

We evaluate supervised PatchTST with input length 384, prediction length 96, random historical missingness at rate 0.3, and seed 2025. Capacity-matched Gaussian and fixed Student-t (df=8) heads use the same frozen clean PatchTST backbone and the same fixed validation and test masks. Future targets remain unmasked and are not used for model selection.

## Results

- ETTh1: Gaussian versus fixed-t8 NLL is 0.892433 versus 0.857116 (delta -0.035317), CRPS is 0.338364 versus 0.335825 (delta -0.002538), and MSE is 0.426829 versus 0.421389 (delta -0.005439).
- Weather: Gaussian versus fixed-t8 NLL is 2.096352 versus 0.277655 (delta -1.818698), CRPS is 0.274037 versus 0.197832 (delta -0.076205), and MSE is 0.232359 versus 0.182180 (delta -0.050179).
- Electricity: Gaussian versus fixed-t8 NLL is 0.413581 versus 0.330148 (delta -0.083433), CRPS is 0.206058 versus 0.204371 (delta -0.001688), and MSE is 0.165829 versus 0.167809 (delta +0.001980).

## Gate Interpretation

Fixed-t8 records 3/3 NLL wins and 3/3 CRPS improvements/noninferiority outcomes. Point-prediction noninferiority and the full-pair gate each pass in 2/3 datasets. The formal decision is `SECOND_BACKBONE_POINT_TRADEOFF`.

## Electricity Tradeoff

The Electricity MSE increases from 0.165829 to 0.167809, corresponding to approximately 1.194%, which is slightly above the pre-registered 1% tolerance. This threshold miss remains visible despite improvements in NLL, CRPS, MAE, and coverage error.

## Tail Diagnostic

The residual-percentile diagnostic covers approximately 1.82e8 forecast elements. Most NLL gains are concentrated in the most extreme Gaussian-residual regions. This is descriptive analysis only, is not a formal gate, and is not a causal mechanism proof; the element count is not treated as an independent sample size for inference.

## Conclusion

The result supports transfer of the probabilistic-score direction across the two evaluated backbones, but not complete cross-backbone multi-metric consistency. Its scope is three datasets, random missing rate 0.3, and one PatchTST backbone/head seed.
