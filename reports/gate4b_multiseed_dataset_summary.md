# Gate 4b Multi-Seed Dataset Summary

All standard deviations use ddof=1. No statistical-significance claim is made.

| summary_type | dataset | variant | seed_count | nll_mean | nll_std_ddof1 | crps_mean | mse_mean | nll_win_count | full_pair_pass_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| variant | etth1 | gaussian_head | 4 | 0.915013745 | 0.006642870 | 0.342011317 | 0.416862234 |  |  |
| variant | etth1 | student_t_head | 4 | 0.871113330 | 0.007800988 | 0.337655659 | 0.415610723 |  |  |
| paired_delta | etth1 | student_t_minus_gaussian | 4 |  |  |  |  | 4 | 4 |
| variant | weather | gaussian_head | 4 | 0.577226385 | 0.017061327 | 0.244788497 | 0.210305888 |  |  |
| variant | weather | student_t_head | 4 | 0.103669519 | 0.020066415 | 0.207543669 | 0.209817987 |  |  |
| paired_delta | weather | student_t_minus_gaussian | 4 |  |  |  |  | 4 | 4 |
| variant | electricity | gaussian_head | 4 | 0.439644143 | 0.002138778 | 0.209365171 | 0.167418517 |  |  |
| variant | electricity | student_t_head | 4 | 0.365347899 | 0.000800332 | 0.207394762 | 0.167702977 |  |  |
| paired_delta | electricity | student_t_minus_gaussian | 4 |  |  |  |  | 4 | 4 |
