# Gate 4b Multi-Seed Pairing Validation

Status: **PASS**

```json
{
  "status": "PASS",
  "expected_pair_audits": 9,
  "passed_pair_audits": 9,
  "different_seed_sequences_distinct_by_dataset": {
    "etth1": true,
    "weather": true,
    "electricity": true
  },
  "rows": [
    {
      "dataset": "etth1",
      "experiment_seed": 2021,
      "status": "PASS",
      "first_batch_indices_sha256": "3e643a1f87e534f60cd77c9115b1c9c0f73153b81a45923c32d4d9764f3744f8",
      "first_batch_mask_sha256": "f0c19eaa420a8fbd487cbc78bf94deeb22d8ad7f3d35a5506adcac6b91f89174",
      "actual_first_batch_missing_rate": 0.30035400390625,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "etth1",
      "experiment_seed": 2022,
      "status": "PASS",
      "first_batch_indices_sha256": "1c3f1a8e33e0216fb52eb57d6751ffee298051af03ad94a0fa30db2c5abaadfa",
      "first_batch_mask_sha256": "f80c8677f36c6872a74663893589a407824de73fd89b14e3f6f05f2ddeac5b1f",
      "actual_first_batch_missing_rate": 0.2999994158744812,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "etth1",
      "experiment_seed": 2023,
      "status": "PASS",
      "first_batch_indices_sha256": "f5e27144f1740c8b4772d70f40b634e97844c402ffcee28f12976bcd8b9d4488",
      "first_batch_mask_sha256": "3794c24230fb1486cf8a2145f50adc6d760f28485692bbdd74e88eb1fac9352b",
      "actual_first_batch_missing_rate": 0.3004964292049408,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "weather",
      "experiment_seed": 2021,
      "status": "PASS",
      "first_batch_indices_sha256": "72ba9c7815faf0f58c76d79ddb2c2410998b7aaeb99d1a21fe2f18d8dce2bbff",
      "first_batch_mask_sha256": "25b43485777de10473304b79433c100185af32a1f97f7cfd279b1c35b98ff04a",
      "actual_first_batch_missing_rate": 0.3003501296043396,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "weather",
      "experiment_seed": 2022,
      "status": "PASS",
      "first_batch_indices_sha256": "f9b9f639bf830f7b42827d049b19e434fd0df06ba193651f871cc7efdf3e45a6",
      "first_batch_mask_sha256": "07fd8163fbea6c9cd0889bacea6939faad58b197a113b723f27daa0162916974",
      "actual_first_batch_missing_rate": 0.2997523844242096,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "weather",
      "experiment_seed": 2023,
      "status": "PASS",
      "first_batch_indices_sha256": "a9991516521694b2ef81f855a5948cf246ce526253ac23f3c14139b94d31779e",
      "first_batch_mask_sha256": "ef48204b97cf98382f3d5b4e705c1cf3e03b0b8b7de79b760d7284cbf424df14",
      "actual_first_batch_missing_rate": 0.3002668023109436,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "electricity",
      "experiment_seed": 2021,
      "status": "PASS",
      "first_batch_indices_sha256": "69cea5e007622b3bffbde2e2478ce9c24ddb30775e8b70ba312056fd98f2b03a",
      "first_batch_mask_sha256": "aea746c809fe64b78a9c061ca2a5258db1ca6d2f2b9f25bf573ab0c23124eb80",
      "actual_first_batch_missing_rate": 0.2999400794506073,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "electricity",
      "experiment_seed": 2022,
      "status": "PASS",
      "first_batch_indices_sha256": "8be8d8c9986dc7d3ed3c18afeb1a70ac2adf4007bab7a5fac9ddf577e7ebcf77",
      "first_batch_mask_sha256": "88815b9eb2691bffbc3a5e166ccbbe390537568e662d8dd7f7cf5c5991e0b023",
      "actual_first_batch_missing_rate": 0.2997051775455475,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    },
    {
      "dataset": "electricity",
      "experiment_seed": 2023,
      "status": "PASS",
      "first_batch_indices_sha256": "0b3e87da30084e46b730f2169422a51601a79e5d31b2951f29d8f143a303934e",
      "first_batch_mask_sha256": "ba859acfc8e2a7b2f72ebbe338c20b6e20df8520c492997dd81bb2eb4b8200ea",
      "actual_first_batch_missing_rate": 0.30009928345680237,
      "checks": {
        "mask_equal": true,
        "first_batch_indices_equal": true,
        "validation_mask_hash_equal": true,
        "test_mask_hash_equal": true,
        "source_checkpoint_hash_equal": true,
        "training_protocol_hash_equal": true,
        "metric_protocol_hash_equal": true
      },
      "different_seed_masks_distinct": true
    }
  ]
}
```
