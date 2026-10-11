# Gate 5a Distribution Validation

Status: **PASS**

Synthetic double-precision checks; no training or test-set selection is performed.

```json
{
  "status": "PASS",
  "thresholds": {
    "laplace_nll_max_abs_diff": 1e-07,
    "laplace_quantile_max_abs_diff": 1e-06,
    "fixed_t_nll_max_abs_diff": 1e-07,
    "fixed_t_quantile_max_abs_diff": 1e-06,
    "initial_fixed_learnable_nll_max_abs_diff": 1e-06
  },
  "observed": {
    "laplace_nll_max_abs_diff": 0.0,
    "laplace_quantile_max_abs_diff": 0.0,
    "fixed_t_nll_max_abs_diff": 0.0,
    "fixed_t_quantile_max_abs_diff": 0.0,
    "initial_fixed_learnable_nll_max_abs_diff": 1.7895178849869353e-07,
    "crps": {
      "gaussian_head": 0.8598733615357157,
      "laplace_head": 0.8514668919543095,
      "student_t_fixed_df8_head": 0.8499188898262774,
      "student_t_head": 0.8499188897055939
    },
    "nll_finite": {
      "gaussian_head": true,
      "laplace_head": true,
      "student_t_fixed_df8_head": true,
      "student_t_head": true
    },
    "interval_nesting": {
      "gaussian_head": true,
      "laplace_head": true,
      "student_t_fixed_df8_head": true,
      "student_t_head": true
    },
    "learnable_initial_df_min": 7.999999842032131,
    "learnable_initial_df_max": 7.999999842032131
  },
  "capacity": [
    {
      "channels": 7,
      "counts": {
        "gaussian_head": 1560,
        "laplace_head": 1560,
        "student_t_fixed_df8_head": 1560,
        "student_t_head": 1567
      },
      "status": "PASS"
    },
    {
      "channels": 21,
      "counts": {
        "gaussian_head": 1560,
        "laplace_head": 1560,
        "student_t_fixed_df8_head": 1560,
        "student_t_head": 1581
      },
      "status": "PASS"
    },
    {
      "channels": 321,
      "counts": {
        "gaussian_head": 1560,
        "laplace_head": 1560,
        "student_t_fixed_df8_head": 1560,
        "student_t_head": 1881
      },
      "status": "PASS"
    }
  ],
  "checks": {
    "laplace_formula": true,
    "laplace_quantile": true,
    "fixed_t_nll_parity": true,
    "fixed_t_quantile_parity": true,
    "fixed_t_sample_shape": true,
    "fixed_t_sample_values": true,
    "initial_fixed_learnable_parity": true,
    "capacity_fairness": true,
    "scale_finite_positive": true,
    "fixed_df_exactly_8": true,
    "learnable_df_above_2_01": true,
    "all_nll_finite": true,
    "all_crps_finite": true,
    "all_intervals_nested": true
  }
}
```
