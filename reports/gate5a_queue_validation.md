# Gate 5a Queue Validation

Status: **PASS**

```json
{
  "status": "PASS",
  "queue": "configs/queues/gate5a_distribution_ablation.yaml",
  "logical_tasks": 12,
  "reused_tasks": 6,
  "active_tasks": 6,
  "required_free_mib": {
    "etth1": 4096,
    "weather": 7000,
    "electricity": 18000
  },
  "expected_required_free_mib": {
    "etth1": 4096,
    "weather": 7000,
    "electricity": 18000
  },
  "checks": {
    "datasets_exact": true,
    "heads_exact": true,
    "each_dataset_head_once": true,
    "logical_tasks_12": true,
    "reused_6": true,
    "active_6": true,
    "random_rate_0_3_only": true,
    "seed_2025_only": true,
    "train_mask_base_seed_42025": true,
    "frozen_backbone": true,
    "full_training": true,
    "optimization_protocol": true,
    "distribution_initialization": true,
    "fixed_assets_present": true,
    "no_forbidden_modules": true,
    "resource_formula_exact": true,
    "resource_floors_respected": true,
    "one_task_per_gpu": true,
    "bounded_concurrency": true,
    "old_unmasked_reuse_count_zero": true,
    "all_reuse_valid": true,
    "all_active_validate_only_pass": true
  },
  "reused_validation": [
    {
      "task_id": "etth1__random_rate0.3__gaussian_head",
      "run_dir": "runs/timeperceiver__etth1__gate4b_corrected_minimal__gaussian_head__random_rate0.3__seed2025__20260713_130557",
      "status": "PASS"
    },
    {
      "task_id": "etth1__random_rate0.3__student_t_head",
      "run_dir": "runs/timeperceiver__etth1__gate4b_corrected_minimal__student_t_head__random_rate0.3__seed2025__20260713_130713",
      "status": "PASS"
    },
    {
      "task_id": "weather__random_rate0.3__gaussian_head",
      "run_dir": "runs/timeperceiver__weather__gate4b_cross_dataset__gaussian_head__random_rate0.3__seed2025__20260715_233846",
      "status": "PASS"
    },
    {
      "task_id": "weather__random_rate0.3__student_t_head",
      "run_dir": "runs/timeperceiver__weather__gate4b_cross_dataset__student_t_head__random_rate0.3__seed2025__20260715_234156",
      "status": "PASS"
    },
    {
      "task_id": "electricity__random_rate0.3__gaussian_head",
      "run_dir": "runs/timeperceiver__electricity__gate4b_cross_dataset__gaussian_head__random_rate0.3__seed2025__20260716_062020",
      "status": "PASS"
    },
    {
      "task_id": "electricity__random_rate0.3__student_t_head",
      "run_dir": "runs/timeperceiver__electricity__gate4b_cross_dataset__student_t_head__random_rate0.3__seed2025__20260716_184923",
      "status": "PASS"
    }
  ],
  "active_validate_only": [
    {
      "task_id": "etth1__random_rate0.3__laplace_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    },
    {
      "task_id": "etth1__random_rate0.3__student_t_fixed_df8_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    },
    {
      "task_id": "weather__random_rate0.3__laplace_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    },
    {
      "task_id": "weather__random_rate0.3__student_t_fixed_df8_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    },
    {
      "task_id": "electricity__random_rate0.3__laplace_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    },
    {
      "task_id": "electricity__random_rate0.3__student_t_fixed_df8_head",
      "status": "PASS",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}",
      "stderr": ""
    }
  ]
}
```
