# Gate 4b Cross-Dataset Test Mask Inventory

Status: **PASS**

```json
{
  "status": "PASS",
  "expected": 2,
  "verified": 2,
  "hash_mismatch": 0,
  "metadata_mismatch": 0,
  "rows": [
    {
      "dataset": "weather",
      "status": "PASS",
      "verified": true,
      "path": "artifacts/masks/weather/random/rate_0.3/seed_2025.npz",
      "metadata_path": "artifacts/masks/weather/random/rate_0.3/seed_2025.json",
      "shape": "10444x384x21",
      "dtype": "uint8",
      "target_missing_rate": 0.3,
      "actual_missing_rate": 0.3000566038524436,
      "sha256": "09e9cb9bab52a60da5e9784287a920cbb9ef2f169bb00d54ac8fed684a4cb1e5",
      "split": "test",
      "future_target_used": false,
      "hash_mismatch": false,
      "metadata_mismatch": false
    },
    {
      "dataset": "electricity",
      "status": "PASS",
      "verified": true,
      "path": "artifacts/masks/electricity/random/rate_0.3/seed_2025.npz",
      "metadata_path": "artifacts/masks/electricity/random/rate_0.3/seed_2025.json",
      "shape": "5165x384x321",
      "dtype": "uint8",
      "target_missing_rate": 0.3,
      "actual_missing_rate": 0.3000004130942652,
      "sha256": "4008073202719c7cf1c1e868536e9f1f8fa976af557c5bf267b199c32a2c9521",
      "split": "test",
      "future_target_used": false,
      "hash_mismatch": false,
      "metadata_mismatch": false
    }
  ]
}
```
