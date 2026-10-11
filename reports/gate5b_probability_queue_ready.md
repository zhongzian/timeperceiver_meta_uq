# Gate 5b Probability Queue Ready

Status: **PASS**

```json
{
  "status": "PASS",
  "queue": "configs/queues/gate5b_patchtst_probability.yaml",
  "logical_tasks": 6,
  "launch_enabled": true,
  "prerequisites": {
    "backbone_manifest": true,
    "distribution_validation": true,
    "pairing_validation": true,
    "pairing_3_of_3": true,
    "probability_smoke": true,
    "probability_smoke_6_of_6": true
  },
  "dataset_required_free_mib": {
    "electricity": 3937,
    "etth1": 8192,
    "weather": 10000
  },
  "max_concurrent_total": 2,
  "max_tasks_per_gpu": 1
}
```
