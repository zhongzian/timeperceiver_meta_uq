# Ablation and Negative Results

## Gate 4a: adapter capacity versus missingness conditioning

The frozen-backbone static adapter improves ETTh1 random-0.3 MSE from 0.425394 to 0.361377. The missingness-aware FiLM adapter also improves on the official result, reaching 0.363156, but is 0.001778 MSE worse than the static adapter. The mechanism audit shows that the condition encoder is not constant, FiLM modulation is nonzero, parameters update, and optimization is functional. The evidence therefore does not support a simple gradient or optimization failure. It instead shows that the tested global-statistics, single-point FiLM design provides no additional generalization benefit under this fixed condition. This conclusion does not extend to all missingness-aware adapters.

## Gate 4c: strict additive conformal calibration

Gate 4c passes implementation, split, leakage, and replay-parity checks. It uses a strict raw-time selection/calibration split with a 479-row embargo and independently retrained probability heads. Native test intervals are already close to the 90% and 95% targets, while the calibration segment is descriptively harder. Positive additive conformal quantiles can only widen intervals; in this experiment they move test intervals toward overcoverage and worsen absolute coverage error. Mask-severity conditioning changes bin-specific quantiles but has negligible aggregate effect and does not repair the worst-bin tradeoff. The counterfactual test-label shrinkage analysis is diagnostic only and is not formal calibration. The expanded Gate 4c route is stopped, and the negative result is retained.

## Gate 5b: second-backbone point tradeoff

PatchTST fixed-t8 wins NLL and meets CRPS noninferiority in 3/3 datasets, but point noninferiority holds in only 2/3. The Electricity miss is retained even though its NLL, CRPS, MAE, and coverage error improve. The probabilistic-score direction is supported across the two evaluated backbones; complete metric-wise consistency is not.
