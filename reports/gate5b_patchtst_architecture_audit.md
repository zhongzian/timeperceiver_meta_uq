# Gate 5b PatchTST Architecture Audit

Status: **PASS**

```json
{
  "status": "PASS",
  "official_encoder_output": "[B,C,d_model,patch_count]",
  "hidden_feature_point": "official shared Flatten_Head output before linear projection",
  "deterministic_head_reuse": "loc projection initialized by strict state_dict copy",
  "scale_parameterization": "softplus(raw_scale)+1e-4; zero weight and bias for initial scale 0.5",
  "datasets_passed": 3,
  "datasets_total": 3,
  "rows": [
    {
      "dataset": "etth1",
      "channels": 7,
      "hidden_size": 768,
      "forecast_shape": [
        2,
        96,
        7
      ],
      "loc_shape": [
        2,
        96,
        7
      ],
      "scale_shape": [
        2,
        96,
        7
      ],
      "initial_scale_exact_0.5": true,
      "gaussian_fixed_t8_initialization_equal": true,
      "frozen_backbone_remains_eval_during_head_training": true,
      "gaussian_trainable_parameters": 147648,
      "fixed_t8_trainable_parameters": 147648,
      "gaussian_backbone_trainable_parameters": 0,
      "fixed_t8_backbone_trainable_parameters": 0,
      "fixed_t8_df_parameters": 0,
      "fixed_df": 8.0,
      "deterministic_official_parity": true,
      "status": "PASS"
    },
    {
      "dataset": "weather",
      "channels": 21,
      "hidden_size": 6144,
      "forecast_shape": [
        2,
        96,
        21
      ],
      "loc_shape": [
        2,
        96,
        21
      ],
      "scale_shape": [
        2,
        96,
        21
      ],
      "initial_scale_exact_0.5": true,
      "gaussian_fixed_t8_initialization_equal": true,
      "frozen_backbone_remains_eval_during_head_training": true,
      "gaussian_trainable_parameters": 1179840,
      "fixed_t8_trainable_parameters": 1179840,
      "gaussian_backbone_trainable_parameters": 0,
      "fixed_t8_backbone_trainable_parameters": 0,
      "fixed_t8_df_parameters": 0,
      "fixed_df": 8.0,
      "deterministic_official_parity": true,
      "status": "PASS"
    },
    {
      "dataset": "electricity",
      "channels": 321,
      "hidden_size": 6144,
      "forecast_shape": [
        2,
        96,
        321
      ],
      "loc_shape": [
        2,
        96,
        321
      ],
      "scale_shape": [
        2,
        96,
        321
      ],
      "initial_scale_exact_0.5": true,
      "gaussian_fixed_t8_initialization_equal": true,
      "frozen_backbone_remains_eval_during_head_training": true,
      "gaussian_trainable_parameters": 1179840,
      "fixed_t8_trainable_parameters": 1179840,
      "gaussian_backbone_trainable_parameters": 0,
      "fixed_t8_backbone_trainable_parameters": 0,
      "fixed_t8_df_parameters": 0,
      "fixed_df": 8.0,
      "deterministic_official_parity": true,
      "status": "PASS"
    }
  ]
}
```
