# Gate 4c Interval Metrics

| Head | Method | Coverage | PICP | MPIW | Abs coverage error | Interval score |
|---|---|---:|---:|---:|---:|---:|
| gaussian_head | native | 0.90 | 0.904533 | 2.040512 | 0.004533 | 2.936425 |
| gaussian_head | native | 0.95 | 0.937263 | 2.431420 | 0.012737 | 3.571486 |
| gaussian_head | marginal | 0.90 | 0.928261 | 2.294196 | 0.028261 | 2.979930 |
| gaussian_head | marginal | 0.95 | 0.964711 | 2.991181 | 0.014711 | 3.598406 |
| gaussian_head | conditioned | 0.90 | 0.927797 | 2.289391 | 0.027797 | 2.979145 |
| gaussian_head | conditioned | 0.95 | 0.964530 | 2.986506 | 0.014530 | 3.598434 |
| student_t_head | native | 0.90 | 0.897581 | 1.953189 | 0.002419 | 2.946898 |
| student_t_head | native | 0.95 | 0.937700 | 2.424187 | 0.012300 | 3.596873 |
| student_t_head | marginal | 0.90 | 0.927636 | 2.267989 | 0.027636 | 2.990528 |
| student_t_head | marginal | 0.95 | 0.964505 | 2.998730 | 0.014505 | 3.625259 |
| student_t_head | conditioned | 0.90 | 0.927136 | 2.262613 | 0.027136 | 2.989775 |
| student_t_head | conditioned | 0.95 | 0.964262 | 2.993500 | 0.014262 | 3.625242 |

Elementwise marginal empirical coverage under a fixed time-series split; no exact exchangeability guarantee is claimed.
