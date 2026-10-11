# Gate 5b Formal Decision

Status: **PASS**

```json
{
  "status": "PASS",
  "decision": "SECOND_BACKBONE_POINT_TRADEOFF",
  "GATE5B_IMPLEMENTATION_PASS": true,
  "GATE5B_NLL_WIN_COUNT": 3,
  "GATE5B_CRPS_NONINFERIOR_COUNT": 3,
  "GATE5B_POINT_NONINFERIOR_COUNT": 2,
  "GATE5B_FULL_PAIR_PASS_COUNT": 2,
  "GATE5B_SECOND_BACKBONE_SIGNAL_POSITIVE": false,
  "CROSS_BACKBONE_DIRECTION_CONSISTENT": false,
  "READY_FOR_PAPER_FINALIZATION": true,
  "statistical_significance_claim": false,
  "scope": "two evaluated backbones; three datasets; random0.3; PatchTST seed2025",
  "implementation_checks": {
    "source_audit": true,
    "environment_audit": true,
    "data_protocol_3_of_3": true,
    "backbone_smoke_3_of_3": true,
    "backbone_formal_3_of_3": true,
    "distribution_validation": true,
    "pairing_3_of_3": true,
    "probability_smoke_6_of_6": true,
    "probability_formal_6_of_6": true,
    "probability_run_audits": true,
    "capacity_fairness_3_of_3": true,
    "paired_rows_3": true,
    "timeperceiver_upstream_tracked_diff_empty": true,
    "patchtst_upstream_tracked_diff_empty": true
  },
  "probability_run_audit": [
    {
      "experiment_id": "patchtst__electricity__gate5b1_probability__gaussian__random_rate0.3__seed2025__20260721_184438",
      "dataset": "electricity",
      "distribution": "gaussian",
      "trainable_parameter_count": 1179840,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true
      },
      "pass": true
    },
    {
      "experiment_id": "patchtst__electricity__gate5b1_probability__fixed_t8__random_rate0.3__seed2025__20260721_184438",
      "dataset": "electricity",
      "distribution": "fixed_t8",
      "trainable_parameter_count": 1179840,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true,
        "fixed_df_exact": true,
        "fixed_df_parameter_count_zero": true
      },
      "pass": true
    },
    {
      "experiment_id": "patchtst__etth1__gate5b1_probability__gaussian__random_rate0.3__seed2025__20260721_200336",
      "dataset": "etth1",
      "distribution": "gaussian",
      "trainable_parameter_count": 147648,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true
      },
      "pass": true
    },
    {
      "experiment_id": "patchtst__etth1__gate5b1_probability__fixed_t8__random_rate0.3__seed2025__20260721_201532",
      "dataset": "etth1",
      "distribution": "fixed_t8",
      "trainable_parameter_count": 147648,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true,
        "fixed_df_exact": true,
        "fixed_df_parameter_count_zero": true
      },
      "pass": true
    },
    {
      "experiment_id": "patchtst__weather__gate5b1_probability__gaussian__random_rate0.3__seed2025__20260721_202045",
      "dataset": "weather",
      "distribution": "gaussian",
      "trainable_parameter_count": 1179840,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true
      },
      "pass": true
    },
    {
      "experiment_id": "patchtst__weather__gate5b1_probability__fixed_t8__random_rate0.3__seed2025__20260721_210413",
      "dataset": "weather",
      "distribution": "fixed_t8",
      "trainable_parameter_count": 1179840,
      "checks": {
        "status_success": true,
        "metrics_finite": true,
        "input_mask_applied": true,
        "mask_application_verified": true,
        "batch_y_unchanged": true,
        "test_not_used_for_selection": true,
        "no_leakage": true,
        "backbone_frozen": true,
        "optimizer_excludes_backbone": true,
        "strict_reload": true,
        "evaluation_after_reload": true,
        "scale_finite_positive": true,
        "source_checkpoint_hash": true,
        "fixed_df_exact": true,
        "fixed_df_parameter_count_zero": true
      },
      "pass": true
    }
  ],
  "capacity_fairness": {
    "etth1": true,
    "weather": true,
    "electricity": true
  },
  "paired_results": [
    {
      "crps_noninferior": true,
      "dataset": "etth1",
      "delta_coverage_error": "-0.012426796555519093",
      "delta_crps": "-0.00253834971874467",
      "delta_interval_score": "0.0594000518321991",
      "delta_mae": "-0.005807816982269287",
      "delta_mse": "-0.005439400672912598",
      "delta_nll": "-0.0353165864944458",
      "full_pair_pass": true,
      "nll_win": true,
      "point_noninferior": true
    },
    {
      "crps_noninferior": true,
      "dataset": "weather",
      "delta_coverage_error": "-0.07148244976997375",
      "delta_crps": "-0.07620540445036286",
      "delta_interval_score": "-0.7844064235687256",
      "delta_mae": "-0.03458458185195923",
      "delta_mse": "-0.05017869174480438",
      "delta_nll": "-1.8186977803707123",
      "full_pair_pass": true,
      "nll_win": true,
      "point_noninferior": true
    },
    {
      "crps_noninferior": true,
      "dataset": "electricity",
      "delta_coverage_error": "-0.01924067139625546",
      "delta_crps": "-0.001687663580371479",
      "delta_interval_score": "0.012489423155784607",
      "delta_mae": "-0.0018184781074523926",
      "delta_mse": "0.001980006694793701",
      "delta_nll": "-0.08343291282653809",
      "full_pair_pass": false,
      "nll_win": true,
      "point_noninferior": false
    }
  ],
  "gate5a_fixed_heavy_tail_signal_positive": true
}
```
