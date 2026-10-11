# Gate 5b Checkpoint Provenance

Status: **PASS**

```json
{
  "status": "PASS",
  "manifest": "results/gate5b_patchtst_backbone_manifest.json",
  "datasets": [
    {
      "dataset": "etth1",
      "checkpoint": "runs/patchtst__etth1__gate5b0_backbone__clean__seed2025__20260720_201509/checkpoints/best_backbone.pt",
      "checkpoint_sha256": "de276de1767f3199ef759c0aba172f7915a8b8a619219c5db00038ddac69b203",
      "checks": {
        "checkpoint_exists": true,
        "sha256_match": true,
        "dataset_match": true,
        "seed_match": true,
        "patchtst_commit_match": true,
        "model_state_present": true,
        "selection_metric_validation_mse": true,
        "test_not_used_for_selection": true,
        "strict_reload": true
      },
      "status": "PASS"
    },
    {
      "dataset": "weather",
      "checkpoint": "runs/patchtst__weather__gate5b0_backbone__clean__seed2025__20260720_222732/checkpoints/best_backbone.pt",
      "checkpoint_sha256": "e581b59dbb59fadeac18a6d0f97a49d39b1e85d64ca43a1a3d58b3b793488045",
      "checks": {
        "checkpoint_exists": true,
        "sha256_match": true,
        "dataset_match": true,
        "seed_match": true,
        "patchtst_commit_match": true,
        "model_state_present": true,
        "selection_metric_validation_mse": true,
        "test_not_used_for_selection": true,
        "strict_reload": true
      },
      "status": "PASS"
    },
    {
      "dataset": "electricity",
      "checkpoint": "runs/patchtst__electricity__gate5b0_backbone__clean__seed2025__20260720_184734/checkpoints/best_backbone.pt",
      "checkpoint_sha256": "716f715eee0798b373135b81bb70165c2df698f57534b7ae18e5b6bc03b1009d",
      "checks": {
        "checkpoint_exists": true,
        "sha256_match": true,
        "dataset_match": true,
        "seed_match": true,
        "patchtst_commit_match": true,
        "model_state_present": true,
        "selection_metric_validation_mse": true,
        "test_not_used_for_selection": true,
        "strict_reload": true
      },
      "status": "PASS"
    }
  ],
  "probability_tasks_materialized": 6,
  "all_checkpoint_hashes_match": true,
  "queue_launch_enabled": false
}
```
