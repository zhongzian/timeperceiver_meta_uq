# Gate 4b Cross-Dataset Reuse Inventory

Status: **PASS**

```json
{
  "status": "PASS",
  "expected_probability_tasks": 4,
  "reused_probability_tasks": 0,
  "rows": [
    {
      "dataset": "weather",
      "variant": "gaussian_head",
      "reusable": false,
      "run_dir": "",
      "reason": "no_matching_corrected_formal_run",
      "input_mask_applied": false,
      "mask_application_verified": false,
      "test_mask_sha256": "09e9cb9bab52a60da5e9784287a920cbb9ef2f169bb00d54ac8fed684a4cb1e5",
      "actual_test_missing_rate": 0.3000566038524436,
      "legacy_unmasked_invalid": false
    },
    {
      "dataset": "weather",
      "variant": "student_t_head",
      "reusable": false,
      "run_dir": "",
      "reason": "no_matching_corrected_formal_run",
      "input_mask_applied": false,
      "mask_application_verified": false,
      "test_mask_sha256": "09e9cb9bab52a60da5e9784287a920cbb9ef2f169bb00d54ac8fed684a4cb1e5",
      "actual_test_missing_rate": 0.3000566038524436,
      "legacy_unmasked_invalid": false
    },
    {
      "dataset": "electricity",
      "variant": "gaussian_head",
      "reusable": false,
      "run_dir": "",
      "reason": "no_matching_corrected_formal_run",
      "input_mask_applied": false,
      "mask_application_verified": false,
      "test_mask_sha256": "4008073202719c7cf1c1e868536e9f1f8fa976af557c5bf267b199c32a2c9521",
      "actual_test_missing_rate": 0.3000004130942652,
      "legacy_unmasked_invalid": false
    },
    {
      "dataset": "electricity",
      "variant": "student_t_head",
      "reusable": false,
      "run_dir": "",
      "reason": "no_matching_corrected_formal_run",
      "input_mask_applied": false,
      "mask_application_verified": false,
      "test_mask_sha256": "4008073202719c7cf1c1e868536e9f1f8fa976af557c5bf267b199c32a2c9521",
      "actual_test_missing_rate": 0.3000004130942652,
      "legacy_unmasked_invalid": false
    }
  ]
}
```
