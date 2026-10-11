# Gate 5b Plan

Status: **PLANNING_PASS**

```json
{
  "status": "PLANNING_PASS",
  "stage": "Gate 5b Second-Backbone Minimal Replication",
  "second_backbone": "PatchTST supervised",
  "datasets": [
    "etth1",
    "weather",
    "electricity"
  ],
  "seq_len": 384,
  "pred_len": 96,
  "missing_type": "random",
  "missing_rate": 0.3,
  "experiment_seed": 2025,
  "gate5b0_backbone_tasks": 3,
  "gate5b1_probability_tasks": 6,
  "maximum_new_formal_tasks": 9,
  "extra_tasks": 0,
  "probability_heads": [
    "Gaussian",
    "Fixed Student-t df=8"
  ],
  "sequence": [
    "source/environment/config/architecture/data audits",
    "backbone smoke 3/3",
    "backbone queue validation and execution",
    "checkpoint manifest",
    "distribution/pairing validation",
    "probability smoke 6/6",
    "probability queue validation and execution",
    "formal decision and strict closure"
  ],
  "test_used_for_configuration_selection": false,
  "learnable_df": false,
  "laplace": false,
  "adapter": false,
  "conformal": false
}
```
