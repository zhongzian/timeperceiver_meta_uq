# Gate 5a Pairwise Deltas

Status: **PASS**

NLL deltas are absolute differences, never percentage changes.

```json
{
  "status": "PASS",
  "rows": [
    {
      "dataset": "etth1",
      "comparison": "Laplace vs Gaussian",
      "candidate": "laplace_head",
      "reference": "gaussian_head",
      "delta_nll": -0.039418935775756836,
      "delta_crps": -0.0023412083034444753,
      "delta_mse": 0.0038170814514160156,
      "delta_coverage_error": -0.02622198164463041,
      "delta_interval_score": 0.01938501000404358,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "etth1",
      "comparison": "Fixed-t vs Gaussian",
      "candidate": "student_t_fixed_df8_head",
      "reference": "gaussian_head",
      "delta_nll": -0.05358666181564331,
      "delta_crps": -0.007578038798343012,
      "delta_mse": -0.006404846906661987,
      "delta_coverage_error": -0.01612565219402312,
      "delta_interval_score": -0.011909663677215576,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "etth1",
      "comparison": "Learnable-t vs Fixed-t",
      "candidate": "student_t_head",
      "reference": "student_t_fixed_df8_head",
      "delta_nll": 0.012205183506011963,
      "delta_crps": 0.003714138232225661,
      "delta_mse": 0.005249649286270142,
      "delta_coverage_error": -0.0017047673463821411,
      "delta_interval_score": 0.009591370820999146,
      "nll_win": false,
      "crps_noninferior": false,
      "point_noninferior": false
    },
    {
      "dataset": "etth1",
      "comparison": "Learnable-t vs Laplace",
      "candidate": "student_t_head",
      "reference": "laplace_head",
      "delta_nll": -0.0019625425338745117,
      "delta_crps": -0.0015226922626728756,
      "delta_mse": -0.004972279071807861,
      "delta_coverage_error": 0.008391562104225148,
      "delta_interval_score": -0.02170330286026001,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "weather",
      "comparison": "Laplace vs Gaussian",
      "candidate": "laplace_head",
      "reference": "gaussian_head",
      "delta_nll": -0.38192594051361084,
      "delta_crps": -0.03585291972973781,
      "delta_mse": -0.0021860748529434204,
      "delta_coverage_error": -0.011861786246299744,
      "delta_interval_score": -0.23334848880767822,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "weather",
      "comparison": "Fixed-t vs Gaussian",
      "candidate": "student_t_fixed_df8_head",
      "reference": "gaussian_head",
      "delta_nll": -0.450160413980484,
      "delta_crps": -0.03698074406820864,
      "delta_mse": -0.0012427270412445068,
      "delta_coverage_error": -0.031035542488098145,
      "delta_interval_score": -0.3368455320596695,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "weather",
      "comparison": "Learnable-t vs Fixed-t",
      "candidate": "student_t_head",
      "reference": "student_t_fixed_df8_head",
      "delta_nll": -0.006871134042739868,
      "delta_crps": -0.00018514427350221907,
      "delta_mse": -0.00015769898891448975,
      "delta_coverage_error": 0.0005244165658950806,
      "delta_interval_score": 0.0017693489789962769,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "weather",
      "comparison": "Learnable-t vs Laplace",
      "candidate": "student_t_head",
      "reference": "laplace_head",
      "delta_nll": -0.07510560750961304,
      "delta_crps": -0.0013129686119730533,
      "delta_mse": 0.0007856488227844238,
      "delta_coverage_error": -0.01864933967590332,
      "delta_interval_score": -0.101727694272995,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "electricity",
      "comparison": "Laplace vs Gaussian",
      "candidate": "laplace_head",
      "reference": "gaussian_head",
      "delta_nll": -0.07020944356918335,
      "delta_crps": -0.0006850173140195426,
      "delta_mse": 0.0025553107261657715,
      "delta_coverage_error": -0.02008223384618757,
      "delta_interval_score": 0.01985161006450653,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": false
    },
    {
      "dataset": "electricity",
      "comparison": "Fixed-t vs Gaussian",
      "candidate": "student_t_fixed_df8_head",
      "reference": "gaussian_head",
      "delta_nll": -0.07358911633491516,
      "delta_crps": -0.0016564445739762068,
      "delta_mse": 0.0005919784307479858,
      "delta_coverage_error": -0.019559666514396667,
      "delta_interval_score": 0.00064106285572052,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "electricity",
      "comparison": "Learnable-t vs Fixed-t",
      "candidate": "student_t_head",
      "reference": "student_t_fixed_df8_head",
      "delta_nll": -0.0005597770214080811,
      "delta_crps": -1.0441398084315434e-05,
      "delta_mse": 1.3023614883422852e-05,
      "delta_coverage_error": -0.0002422034740447998,
      "delta_interval_score": -2.8118491172790527e-05,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    },
    {
      "dataset": "electricity",
      "comparison": "Learnable-t vs Laplace",
      "candidate": "student_t_head",
      "reference": "laplace_head",
      "delta_nll": -0.003939449787139893,
      "delta_crps": -0.0009818686580409797,
      "delta_mse": -0.0019503086805343628,
      "delta_coverage_error": 0.00028036385774610206,
      "delta_interval_score": -0.0192386656999588,
      "nll_win": true,
      "crps_noninferior": true,
      "point_noninferior": true
    }
  ]
}
```
