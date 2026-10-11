# Gate 2 Official Baseline Report

Status: PASS
Generated: 2026-07-05T10:41:49

Official model code: not modified by Gate 2 runner.

| Dataset | Status | Experiment ID | MSE | MAE | RMSE | Best epoch | Elapsed seconds |
| ------- | ------ | ------------- | --: | --: | ---: | ---------: | --------------: |
| ETTh1 | SUCCESS | `timeperceiver__etth1__official_baseline__seed2025__20260701_203054` | 0.374462 | 0.398943 | 0.611933 | 26 | 300.602 |
| Weather | SUCCESS | `timeperceiver__weather__official_baseline__seed2025__20260701_203558` | 0.144059 | 0.199637 | 0.37955 | 38 | 3184.89 |
| Electricity | SUCCESS | `timeperceiver__electricity__official_baseline__seed2025__20260705_090044` | 0.125364 | 0.220123 | 0.354068 | 60 | NA |

## Protocol

All Gate 2 runs use the first `pred_len=96` block from the official dataset script and `run.py` defaults for omitted arguments.
No smoke configuration is counted as a formal baseline.

## Paper Values

Paper reference values are left as `NA` until extracted from a reliable paper/table source.
