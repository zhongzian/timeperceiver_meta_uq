# Gate 4b Multi-Seed Seed Protocol Audit

Status: **PASS**

```json
{
  "status": "PASS",
  "scope": "PROBABILISTIC_HEAD_TRAINING_SEEDS",
  "FULL_END_TO_END_MULTI_SEED": false,
  "trainer_checks": {
    "random.seed(experiment_seed)": true,
    "np.random.seed(experiment_seed)": true,
    "torch.manual_seed(experiment_seed)": true,
    "torch.cuda.manual_seed_all(experiment_seed)": true,
    "torch.Generator().manual_seed(experiment_seed)": true,
    "worker_init_fn=seed_worker": true,
    "train_mask_base_seed, experiment_seed, train_mask_seed_rule": true
  },
  "runner_checks": {
    "allowed_seeds = {2021, 2022, 2023, 2025}": true,
    "40000 + experiment_seed": true,
    "fixed_official_checkpoint": true
  },
  "scheduler_checks": {
    "experiment_seed": true,
    "reuse_metric_protocol_hash": true,
    "reuse_schema_audit_pass": true
  },
  "new_gpu_smoke_required": true,
  "new_gpu_smoke_reason": "shared trainer seed and train-mask path was modified",
  "seed_rows": [
    {
      "experiment_seed": 2021,
      "train_mask_base_seed": 42021,
      "torch_seed": 2021,
      "numpy_seed": 2021,
      "python_seed": 2021,
      "dataloader_seed": 2021,
      "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
      "train_mask_seed_rule": "sha256(dataset|experiment_seed|base_seed|epoch|sample_index|missing_type|target_rate)",
      "validation_mask_seed": 43025,
      "test_mask_seed": 2025,
      "variant_in_mask_seed": false
    },
    {
      "experiment_seed": 2022,
      "train_mask_base_seed": 42022,
      "torch_seed": 2022,
      "numpy_seed": 2022,
      "python_seed": 2022,
      "dataloader_seed": 2022,
      "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
      "train_mask_seed_rule": "sha256(dataset|experiment_seed|base_seed|epoch|sample_index|missing_type|target_rate)",
      "validation_mask_seed": 43025,
      "test_mask_seed": 2025,
      "variant_in_mask_seed": false
    },
    {
      "experiment_seed": 2023,
      "train_mask_base_seed": 42023,
      "torch_seed": 2023,
      "numpy_seed": 2023,
      "python_seed": 2023,
      "dataloader_seed": 2023,
      "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
      "train_mask_seed_rule": "sha256(dataset|experiment_seed|base_seed|epoch|sample_index|missing_type|target_rate)",
      "validation_mask_seed": 43025,
      "test_mask_seed": 2025,
      "variant_in_mask_seed": false
    },
    {
      "experiment_seed": 2025,
      "train_mask_base_seed": 42025,
      "torch_seed": 2025,
      "numpy_seed": 2025,
      "python_seed": 2025,
      "dataloader_seed": 2025,
      "worker_seed_rule": "torch.initial_seed()%2**32 -> numpy/python",
      "train_mask_seed_rule": "historical corrected rule retained for legal reuse",
      "validation_mask_seed": 43025,
      "test_mask_seed": 2025,
      "variant_in_mask_seed": false
    }
  ],
  "train_window_counts": {
    "etth1": 8161,
    "weather": 36408,
    "electricity": 17933
  }
}
```
