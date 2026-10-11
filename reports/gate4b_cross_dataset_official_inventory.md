# Gate 4b Cross-Dataset Official Inventory

Status: **PASS**

```json
{
  "status": "PASS",
  "expected": 2,
  "verified": 2,
  "rows": [
    {
      "dataset": "weather",
      "variant": "official",
      "status": "SUCCESS",
      "experiment_id": "timeperceiver__weather__official_missing_eval_full__random_rate0.3__seed2025__20260711_185320",
      "mse": 0.24080970118857523,
      "mae": 0.30962911078027683,
      "rmse": 0.49072365052906836,
      "actual_missing_rate": 0.30006058041642353,
      "sample_count": 10444,
      "batch_count": 82,
      "checkpoint_sha256": "a99ec027aba651569e188a6c1ecd07d512ed6a077e3ea38f9f4db81aa8da635c",
      "test_mask_sha256": "09e9cb9bab52a60da5e9784287a920cbb9ef2f169bb00d54ac8fed684a4cb1e5",
      "input_mask_applied": true,
      "NLL": "NA",
      "CRPS": "NA",
      "PICP": "NA",
      "MPIW": "NA",
      "coverage_error": "NA",
      "interval_score": "NA",
      "scale": "NA",
      "df": "NA",
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__weather__official_missing_eval_full__random_rate0.3__seed2025__20260711_185320"
    },
    {
      "dataset": "electricity",
      "variant": "official",
      "status": "SUCCESS",
      "experiment_id": "timeperceiver__electricity__official_missing_eval_full__random_rate0.3__seed2025__20260711_185505",
      "mse": 0.20057218964824725,
      "mae": 0.31889133399855646,
      "rmse": 0.4478528660712659,
      "actual_missing_rate": 0.29999999426029345,
      "sample_count": 5165,
      "batch_count": 81,
      "checkpoint_sha256": "93fd70e275cc3f8764e0fc404d05bb32e0431c78aa09ecc3fa83ff111b5c5f1a",
      "test_mask_sha256": "4008073202719c7cf1c1e868536e9f1f8fa976af557c5bf267b199c32a2c9521",
      "input_mask_applied": true,
      "NLL": "NA",
      "CRPS": "NA",
      "PICP": "NA",
      "MPIW": "NA",
      "coverage_error": "NA",
      "interval_score": "NA",
      "scale": "NA",
      "df": "NA",
      "run_dir": "/mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__electricity__official_missing_eval_full__random_rate0.3__seed2025__20260711_185505"
    }
  ]
}
```
