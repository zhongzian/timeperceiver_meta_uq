# Gate 4b Multi-Seed Queue Validation

Status: **PASS**

```json
{
  "status": "PASS",
  "queue": "configs/queues/gate4b_multiseed_confirmation.yaml",
  "logical_tasks": 24,
  "reused_tasks": 6,
  "active_tasks": 18,
  "dataset_count": 3,
  "seed_count": 4,
  "variant_count": 2,
  "pair_count": 12,
  "active_validate_only_pass": 18,
  "old_unmasked_reuse_count": 0,
  "canonical_metric_protocol_hash": "1ec40c25e66d9146850d2b31bd50e9e342b48e2f46d7081ed5f8b4603c19ddea",
  "dataset_required_free_mib": {
    "etth1": 4096,
    "weather": 7000,
    "electricity": 18000
  },
  "errors": [],
  "active_validation": [
    {
      "task_id": "etth1__seed2021__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "etth1__seed2021__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "etth1__seed2022__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "etth1__seed2022__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "etth1__seed2023__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "etth1__seed2023__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2021__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2021__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2022__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2022__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2023__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "weather__seed2023__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2021__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2021__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2022__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2022__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2023__gaussian_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    },
    {
      "task_id": "electricity__seed2023__student_t_head",
      "return_code": 0,
      "output": "{\n  \"status\": \"PASS\",\n  \"errors\": []\n}"
    }
  ],
  "scheduler_started": false
}
```
