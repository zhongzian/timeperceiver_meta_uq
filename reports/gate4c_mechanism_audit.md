# Gate 4c Conformal Calibration Mechanism Audit

## Status

**COMPLETE**

This is a read-only diagnostic audit. It does not change the formal Gate 4c decision.

## Execution Safety

- training: `false`
- optimizer_created: `false`
- backward_called: `false`
- checkpoint_updated: `false`
- formal_calibrator_updated: `false`
- scheduler_started: `false`
- eval_only_replay: `true`
- replay_parity: `true`

## Native Calibration Shift

| Head | Level | Calibration PICP | Test PICP | Shift | Calibration MPIW | Test MPIW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 90% | 0.869114 | 0.904533 | +0.035418 | 2.037646 | 2.040512 |
| Gaussian | 95% | 0.912017 | 0.937263 | +0.025246 | 2.428005 | 2.431420 |
| Student-t | 90% | 0.861667 | 0.897581 | +0.035913 | 1.951690 | 1.953188 |
| Student-t | 95% | 0.913178 | 0.937700 | +0.024522 | 2.422368 | 2.424187 |

## Score Distributions

| Head | Level | Zero ratio | p90 | p95 | p99 | Max | Formal q-hat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 90% | 0.869114 | 0.126820 | 0.480288 | 1.380655 | 2.673472 | 0.126842 |
| Gaussian | 95% | 0.912017 | 0.000000 | 0.279860 | 1.168460 | 2.482032 | 0.279881 |
| Student-t | 90% | 0.861667 | 0.157364 | 0.528535 | 1.487427 | 2.794663 | 0.157400 |
| Student-t | 95% | 0.913178 | 0.000000 | 0.287256 | 1.238920 | 2.563394 | 0.287271 |

## Expansion Cost

| Head | Method | Level | Delta PICP | Delta MPIW | Delta abs error | Delta score |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | marginal | 90% | +0.023728 | +0.253684 | +0.023728 | +0.043504 |
| Gaussian | conditioned | 90% | +0.023264 | +0.248879 | +0.023264 | +0.042719 |
| Gaussian | marginal | 95% | +0.027448 | +0.559761 | +0.001974 | +0.026919 |
| Gaussian | conditioned | 95% | +0.027267 | +0.555086 | +0.001793 | +0.026947 |
| Student-t | marginal | 90% | +0.030055 | +0.314800 | +0.025216 | +0.043630 |
| Student-t | conditioned | 90% | +0.029556 | +0.309425 | +0.024717 | +0.042878 |
| Student-t | marginal | 95% | +0.026805 | +0.574543 | +0.002205 | +0.028387 |
| Student-t | conditioned | 95% | +0.026562 | +0.569313 | +0.001962 | +0.028369 |

## Horizon Findings

- Worst native coverage horizons: `{'gaussian_head_90': [60, 84, 65, 61, 10, 17, 86, 77, 63, 64], 'gaussian_head_95': [9, 10, 3, 8, 4, 7, 6, 11, 2, 34], 'student_t_head_90': [10, 8, 9, 4, 3, 1, 6, 7, 11, 2], 'student_t_head_95': [10, 8, 4, 9, 3, 6, 7, 11, 1, 74]}`
- Best native coverage horizons: `{'gaussian_head_90': [14, 22, 26, 70, 32, 72, 13, 25, 33, 20], 'gaussian_head_95': [60, 17, 84, 16, 29, 61, 65, 86, 53, 46], 'student_t_head_90': [76, 43, 90, 66, 48, 17, 36, 56, 69, 95], 'student_t_head_95': [60, 84, 64, 65, 63, 59, 53, 17, 47, 67]}`
- Early/late concentration: `{'gaussian_head_90': 'late_65_96', 'gaussian_head_95': 'early_1_32', 'student_t_head_90': 'early_1_32', 'student_t_head_95': 'early_1_32'}`
- Conformal overexpansion pattern: `{'gaussian_head_90': 1.0, 'gaussian_head_95': 0.6666666666666666, 'student_t_head_90': 0.9791666666666666, 'student_t_head_95': 0.7083333333333334}`

## Channel Findings

- Worst native channels: `{'gaussian_head_90': [2, 0, 5], 'gaussian_head_95': [2, 0, 5], 'student_t_head_90': [2, 0, 5], 'student_t_head_95': [2, 0, 5]}`
- Best native channels: `{'gaussian_head_90': [6, 4, 1], 'gaussian_head_95': [4, 6, 1], 'student_t_head_90': [6, 4, 1], 'student_t_head_95': [4, 6, 1]}`
- Channels dominating q-hat: `{'gaussian_head_90': {'counts_at_or_above_qhat_by_channel': [12077, 9218, 13484, 8476, 0, 15502, 2663], 'top_two_channels': [5, 2], 'top_two_share': 0.47193096711169, 'one_or_two_channels_dominate_qhat': False}, 'gaussian_head_95': {'counts_at_or_above_qhat_by_channel': [6220, 3799, 7408, 3283, 0, 9863, 137], 'top_two_channels': [5, 2], 'top_two_share': 0.5623901009443179, 'one_or_two_channels_dominate_qhat': True}, 'student_t_head_90': {'counts_at_or_above_qhat_by_channel': [12206, 8479, 13676, 7872, 1, 16728, 2458], 'top_two_channels': [5, 2], 'top_two_share': 0.4950179094757408, 'one_or_two_channels_dominate_qhat': False}, 'student_t_head_95': {'counts_at_or_above_qhat_by_channel': [6296, 3301, 7596, 2929, 0, 10473, 115], 'top_two_channels': [5, 2], 'top_two_share': 0.5883751221100618, 'one_or_two_channels_dominate_qhat': True}}`
- Global q-hat overwidth evidence: `{'gaussian_head_90': [1, 3, 4, 5], 'gaussian_head_95': [1, 3, 4, 5, 6], 'student_t_head_90': [1, 3, 4, 5], 'student_t_head_95': [1, 3, 4, 5, 6]}`

## Severity Findings

- Calibration severity range: `{'min': 0.2667410714285714, 'p10': 0.2888020833333333, 'p90': 0.31063988095238093, 'max': 0.3325892857142857}`
- Test severity range: `{'min': 0.26971726190476186, 'p10': 0.2890625, 'p90': 0.3113839285714286, 'max': 0.3381696428571429}`
- Marginal q-hat: `{'gaussian_head_90': 0.12684202194213867, 'gaussian_head_95': 0.2798805236816406, 'student_t_head_90': 0.15740013122558594, 'student_t_head_95': 0.28727149963378906}`
- Conditioned q-hat range: `{'gaussian_head_90': [0.10121715068817139, 0.14423775672912598], 'gaussian_head_95': [0.24478280544281006, 0.3018209934234619], 'student_t_head_90': [0.12894153594970703, 0.17847555875778198], 'student_t_head_95': [0.2508053779602051, 0.3087609112262726]}`
- Max conditioned-minus-marginal difference: `0.036466121673583984`
- Conditioned effect negligible: `True`

## Distribution Shift

| Head | Variable | Calibration mean | Test mean | Relative shift | KS | Wasserstein |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian | absolute_residual | 0.504375 | 0.452589 | -0.102672 | 0.053604 | 0.052122 |
| Gaussian | scale | 0.619400 | 0.620272 | +0.001407 | 0.034346 | 0.005674 |
| Gaussian | standardized_residual | 0.820685 | 0.729426 | -0.111198 | 0.046537 | 0.091275 |
| Student-t | absolute_residual | 0.497388 | 0.449636 | -0.096006 | 0.050625 | 0.048013 |
| Student-t | scale | 0.523457 | 0.523885 | +0.000818 | 0.020211 | 0.004229 |
| Student-t | standardized_residual | 0.960191 | 0.855262 | -0.109279 | 0.044715 | 0.104931 |
| Student-t | df | 7.860019 | 7.860019 | +0.000000 | 0.000000 | 0.000000 |

## Counterfactual Shrinkage

**COUNTERFACTUAL_ONLY / USES_TEST_LABELS / NOT_FORMAL_CALIBRATION**

| Head | Level | Native PICP | Formal q-hat | Oracle signed delta | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Gaussian | 90% | 0.904533 | 0.126842 | -0.019348 | test_target_would_prefer_shrinkage |
| Gaussian | 95% | 0.937263 | 0.279881 | +0.108755 | test_target_requires_expansion |
| Student-t | 90% | 0.897581 | 0.157400 | +0.009730 | test_target_requires_expansion |
| Student-t | 95% | 0.937700 | 0.287271 | +0.110282 | test_target_requires_expansion |

## Diagnostic Conclusions

- `NATIVE_ALREADY_WELL_CALIBRATED`: `true`
- `CALIBRATION_TEST_DISTRIBUTION_SHIFT`: `true`
- `ADDITIVE_ONLY_EXPANSION_LIMITATION`: `true`
- `GLOBAL_QHAT_OVERWIDENS_EASY_SUBGROUPS`: `true`
- `SEVERITY_CONDITION_LOW_INFORMATION`: `false`
- `CONDITIONED_QHAT_EFFECT_NEGLIGIBLE`: `true`
- `EXTREME_SCORE_TAIL_DRIVES_QHAT`: `true`
- `FINITE_SAMPLE_ORDER_STATISTIC_SENSITIVITY`: `false`

## Final Interpretation

Strict split conformal did not outperform the native intervals because the native Gaussian and Student-t intervals were already close to their target coverage on test, while calibration produced positive additive q-hat values under a descriptively different error distribution. The one-sided additive correction can only widen intervals, so it moved test coverage into overcoverage and increased width; severity conditioning changed bin-specific q-hat values but had negligible aggregate impact and did not repair the worst-bin trade-off. This is a mechanism diagnosis, not an exchangeability or significance claim.

The formal state remains `GATE4C_SIGNAL_POSITIVE=False` and `ENTER_GATE4C_EXPANDED_MATRIX=False`.

## Protected Formal Artifacts

- Bit-identical after audit: `true`

## Next

Stop Gate 4c method expansion. Keep Student-t probabilistic forecasting as the main innovation result, and retain the conformal negative result plus the additive-only limitation in the ablation and limitations analysis.
