# Gate 5a Mechanism Audit

Status: **PASS**

```json
{
  "status": "PASS",
  "descriptive_only": true,
  "statistical_significance_claim": false,
  "GAUSSIAN_RESIDUAL_HEAVY_TAIL_OBSERVED": true,
  "HEAVY_TAIL_GAIN_CONCENTRATED_IN_EXTREMES": true,
  "FIXED_T_EXPLAINS_MOST_STUDENT_T_GAIN": true,
  "LEARNED_DF_ADDS_VALUE_BEYOND_FIXED_T": false,
  "LAPLACE_COMPETITIVE_WITH_STUDENT_T": false,
  "PER_CHANNEL_DF_ASSOCIATED_WITH_CHANNEL_TAILS": true,
  "STUDENT_T_PARAMETER_COST_MINIMAL": true,
  "STUDENT_T_RUNTIME_COST_MINIMAL": false,
  "top_5_percent_student_t_nll_gain": 1.1974153121312459,
  "bottom_90_percent_student_t_nll_gain": 0.17602921463549137,
  "predeclared_rules": {
    "tail_concentrated": "mean absolute learnable-t NLL gain in top 5% > bottom 90%",
    "df_association": "mean absolute descriptive Spearman rho across datasets >= 0.3"
  }
}
```
