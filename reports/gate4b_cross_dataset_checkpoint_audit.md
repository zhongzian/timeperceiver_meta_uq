# Gate 4b Cross-Dataset Checkpoint Audit

Status: **PASS**

```json
{
  "status": "PASS",
  "expected": 2,
  "verified": 2,
  "rows": [
    {
      "dataset": "weather",
      "status": "PASS",
      "official_experiment_id": "timeperceiver__weather__official_baseline__seed2025__20260701_203558",
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official_baseline__seed2025__20260701_203558",
      "checkpoint_path": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official_baseline__seed2025__20260701_203558/checkpoints/weather_384_96_TimePerceiver_sl384_pl96_dm512_nl32_ld128_ul1_gf1_nh8_df1024_pl24_qs1_sr1.0/checkpoint.pth",
      "checkpoint_sha256": "a99ec027aba651569e188a6c1ecd07d512ed6a077e3ea38f9f4db81aa8da635c",
      "strict_load": true,
      "tensor_count": 115,
      "clean_mse": 0.14405854046344757,
      "clean_mae": 0.19963745772838593,
      "clean_rmse": 0.3795504570007324,
      "gate4a_adapter": false,
      "probability_head_weights": false
    },
    {
      "dataset": "electricity",
      "status": "PASS",
      "official_experiment_id": "timeperceiver__electricity__official_baseline__seed2025__20260705_090044",
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official_baseline__seed2025__20260705_090044",
      "checkpoint_path": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official_baseline__seed2025__20260705_090044/checkpoints/electricity_384_96_TimePerceiver_sl384_pl96_dm512_nl32_ld128_ul1_gf1_nh8_df1024_pl24_qs1_sr0.0/checkpoint.pth",
      "checkpoint_sha256": "93fd70e275cc3f8764e0fc404d05bb32e0431c78aa09ecc3fa83ff111b5c5f1a",
      "strict_load": true,
      "tensor_count": 115,
      "clean_mse": 0.12536390125751495,
      "clean_mae": 0.2201225608587265,
      "clean_rmse": 0.3540676534175873,
      "gate4a_adapter": false,
      "probability_head_weights": false
    }
  ]
}
```
