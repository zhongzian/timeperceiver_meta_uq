# Gate 5a Architecture Audit

Status: **PASS**

```json
{
  "status": "PASS",
  "datasets_expected": 3,
  "datasets_passed": 3,
  "rows": [
    {
      "dataset": "etth1",
      "status": "PASS",
      "target_channels": 7,
      "gaussian_trainable_params": 6168,
      "laplace_trainable_params": 6168,
      "fixed_t8_trainable_params": 6168,
      "learnable_t_trainable_params": 6175,
      "learnable_t_extra_params": 7,
      "gaussian_total_params": 1142564,
      "laplace_total_params": 1142564,
      "fixed_t8_total_params": 1142564,
      "learnable_t_total_params": 1142571,
      "learnable_t_raw_df_shape": [
        1,
        1,
        7
      ],
      "fixed_t8_raw_df_absent": true,
      "trainable_parameter_names": {
        "gaussian_head": null,
        "laplace_head": null,
        "student_t_fixed_df8_head": null,
        "student_t_head": [
          "probabilistic_head.raw_df",
          "probabilistic_head.loc_projection.weight",
          "probabilistic_head.loc_projection.bias",
          "probabilistic_head.scale_projection.weight",
          "probabilistic_head.scale_projection.bias"
        ]
      }
    },
    {
      "dataset": "weather",
      "status": "PASS",
      "target_channels": 21,
      "gaussian_trainable_params": 24624,
      "laplace_trainable_params": 24624,
      "fixed_t8_trainable_params": 24624,
      "learnable_t_trainable_params": 24645,
      "learnable_t_extra_params": 21,
      "gaussian_total_params": 4518472,
      "laplace_total_params": 4518472,
      "fixed_t8_total_params": 4518472,
      "learnable_t_total_params": 4518493,
      "learnable_t_raw_df_shape": [
        1,
        1,
        21
      ],
      "fixed_t8_raw_df_absent": true,
      "trainable_parameter_names": {
        "gaussian_head": null,
        "laplace_head": null,
        "student_t_fixed_df8_head": null,
        "student_t_head": [
          "probabilistic_head.raw_df",
          "probabilistic_head.loc_projection.weight",
          "probabilistic_head.loc_projection.bias",
          "probabilistic_head.scale_projection.weight",
          "probabilistic_head.scale_projection.bias"
        ]
      }
    },
    {
      "dataset": "electricity",
      "status": "PASS",
      "target_channels": 321,
      "gaussian_trainable_params": 24624,
      "laplace_trainable_params": 24624,
      "fixed_t8_trainable_params": 24624,
      "learnable_t_trainable_params": 24945,
      "learnable_t_extra_params": 321,
      "gaussian_total_params": 4672072,
      "laplace_total_params": 4672072,
      "fixed_t8_total_params": 4672072,
      "learnable_t_total_params": 4672393,
      "learnable_t_raw_df_shape": [
        1,
        1,
        321
      ],
      "fixed_t8_raw_df_absent": true,
      "trainable_parameter_names": {
        "gaussian_head": null,
        "laplace_head": null,
        "student_t_fixed_df8_head": null,
        "student_t_head": [
          "probabilistic_head.raw_df",
          "probabilistic_head.loc_projection.weight",
          "probabilistic_head.loc_projection.bias",
          "probabilistic_head.scale_projection.weight",
          "probabilistic_head.scale_projection.bias"
        ]
      }
    }
  ]
}
```
