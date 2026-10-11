# Gate 4a Adapter Ablation

| variant | trainable_parameters | parameter_overhead_pct | best_epoch | validation_mse | test_mse | test_mae | test_rmse | mse_change_vs_official | mse_change_vs_static | source_evidence_ids |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Official | 0 | 0.000000 | NA | NA | 0.425394 | 0.450336 | 0.652223 | 0.000000 | NA | EV-d8e55f8661b5e4b7 |
| Static adapter | 33600 | 2.956716 | 1 | 0.678666 | 0.361377 | 0.390458 | 0.601147 | -0.064017 | 0.000000 | EV-1022a9906a254ce9 |
| Missingness-aware FiLM adapter | 39136 | 3.443870 | 5 | 0.679754 | 0.363156 | 0.391077 | 0.602624 | -0.062239 | 0.001778 | EV-9c8b5cf7cd660b74 |

Note: Static capacity improves the official result; the tested global-statistics FiLM condition does not improve on the static adapter.
