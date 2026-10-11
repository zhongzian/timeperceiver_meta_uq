# Gate 3b-smoke Missingness Evaluation Report

- Status: **PASS**
- Run ID: `timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__20260709_153942`
- Run dir: `/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__20260709_153942`
- Dataset: `ETTh1`
- Mask: `random`, rate `0.3`, seed `2025`
- Smoke only: `true`
- Full Gate 3b result: `false`

## Metrics

| MSE | MAE | RMSE | Evaluated batches | Evaluated samples |
|---:|---:|---:|---:|---:|
| 0.4120240495318458 | 0.4251030683517456 | 0.6418909950543361 | 8 | 256 |

## Mask

- mask path: `/mnt/data/zza/code/timeperceiver_meta_uq/artifacts/masks/etth1/random/rate_0.3/seed_2025.npz`
- metadata actual missing rate: `0.30014400059844404`
- evaluated missing rate mean: `0.3015354722738266`

## Baseline Checkpoint

- baseline run: `timeperceiver__etth1__official_baseline__seed2025__20260701_203054`
- checkpoint: `/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_baseline__seed2025__20260701_203054/checkpoints/ETTh1_384_96_TimePerceiver_sl384_pl96_dm256_nl32_ld64_ul1_gf1_nh4_df512_pl12_qs1_sr1.0/checkpoint.pth`

## Safety Checks

| Check | Status |
|---|---|
| Training started | NO |
| Official model structure modified | NO |
| Mask applied only to seq_x/history | YES |
| Future prediction target masked | NO |
| Evaluation-time resampling | NO |
| Entered Gate 4 | NO |
