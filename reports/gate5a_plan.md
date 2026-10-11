# Gate 5a Plan

Status: **PLANNING_PASS**

```json
{
  "status": "PLANNING_PASS",
  "stage": "Gate 5a distribution-head ablation",
  "datasets": [
    "etth1",
    "weather",
    "electricity"
  ],
  "missing_type": "random",
  "missing_rate": 0.3,
  "experiment_seed": 2025,
  "logical_tasks": 12,
  "reused_tasks": 6,
  "new_active_tasks": 6,
  "formal_configs": [
    "configs/gate5a/etth1_random0.3_laplace_seed2025.yaml",
    "configs/gate5a/etth1_random0.3_fixed_t8_seed2025.yaml",
    "configs/gate5a/weather_random0.3_laplace_seed2025.yaml",
    "configs/gate5a/weather_random0.3_fixed_t8_seed2025.yaml",
    "configs/gate5a/electricity_random0.3_laplace_seed2025.yaml",
    "configs/gate5a/electricity_random0.3_fixed_t8_seed2025.yaml"
  ],
  "queue_created": false,
  "scheduler_started": false,
  "old_unmasked_excluded": true,
  "second_backbone_started": false
}
```
