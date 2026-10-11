# Gate 5a Channel Tail Analysis

Status: **PASS**

```json
{
  "status": "PASS",
  "descriptive_only": true,
  "statistical_significance_claim": false,
  "correlations": [
    {
      "dataset": "etth1",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_student_t_df",
      "spearman_rho": 0.4642857142857144,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "etth1",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": -0.21428571428571433,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "etth1",
      "left": "learnable_student_t_df",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": 0.5,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "weather",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_student_t_df",
      "spearman_rho": -0.7909090909090909,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "weather",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": -0.7727272727272727,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "weather",
      "left": "learnable_student_t_df",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": 0.8636363636363636,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "electricity",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_student_t_df",
      "spearman_rho": -0.19287963661693855,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "electricity",
      "left": "gaussian_residual_excess_kurtosis",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": -0.7398734544610207,
      "descriptive_only": true,
      "statistical_significance_claim": false
    },
    {
      "dataset": "electricity",
      "left": "learnable_student_t_df",
      "right": "learnable_t_minus_gaussian_mean_nll",
      "spearman_rho": 0.4488495530272248,
      "descriptive_only": true,
      "statistical_significance_claim": false
    }
  ]
}
```
