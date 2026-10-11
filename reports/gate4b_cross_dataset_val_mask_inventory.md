# Gate 4b Cross-Dataset Val Mask Inventory

Status: **PASS**

```json
{
  "status": "PASS",
  "expected": 2,
  "created_or_reused": 2,
  "verified": 2,
  "rows": [
    {
      "dataset": "weather",
      "status": "PASS",
      "created": true,
      "reused": false,
      "verified": true,
      "path": "artifacts/gate4b_val_masks/weather/random/rate_0.3/seed_43025.npz",
      "metadata_path": "artifacts/gate4b_val_masks/weather/random/rate_0.3/seed_43025.metadata.json",
      "shape": "5175x384x21",
      "dtype": "uint8",
      "target_missing_rate": 0.3,
      "actual_missing_rate": 0.3000480935127674,
      "sha256": "88a4135d84603bd79a8bec066dfa71f0022d3abd71ad2979d0bacf2e2ccd9306",
      "split": "validation",
      "future_target_used": false
    },
    {
      "dataset": "electricity",
      "status": "PASS",
      "created": true,
      "reused": false,
      "verified": true,
      "path": "artifacts/gate4b_val_masks/electricity/random/rate_0.3/seed_43025.npz",
      "metadata_path": "artifacts/gate4b_val_masks/electricity/random/rate_0.3/seed_43025.metadata.json",
      "shape": "2537x384x321",
      "dtype": "uint8",
      "target_missing_rate": 0.3,
      "actual_missing_rate": 0.29999740855074897,
      "sha256": "27a60818d8888557ba5fd21a7cd04db9f3401c9e9faaee6f4196c7b47e71a6a9",
      "split": "validation",
      "future_target_used": false
    }
  ]
}
```
