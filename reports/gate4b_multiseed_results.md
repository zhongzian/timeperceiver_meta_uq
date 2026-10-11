# Gate 4b Multi-Seed Results

The three official rows are fixed-checkpoint references; the 24 probability rows are the formal multi-seed evidence. Historical unmasked runs are excluded.

| dataset | variant | experiment_seed | reused | status | test_mean_nll | test_crps | test_mse | test_mae | mean_absolute_coverage_error | mean_df |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| etth1 | official | fixed_official_checkpoint | True | SUCCESS | NA | NA | 0.425394421 | 0.450336466 | NA |  |
| etth1 | gaussian_head | 2021 | False | SUCCESS | 0.905062735 | 0.338694924 | 0.412684888 | 0.443066776 | 0.043280050 | NA |
| etth1 | student_t_head | 2021 | False | SUCCESS | 0.862314880 | 0.335386192 | 0.412625045 | 0.441828758 | 0.024558914 | 7.961117268 |
| etth1 | gaussian_head | 2022 | False | SUCCESS | 0.917847514 | 0.342288433 | 0.415999353 | 0.448343724 | 0.042614430 | NA |
| etth1 | student_t_head | 2022 | False | SUCCESS | 0.878187120 | 0.338362638 | 0.415481389 | 0.446291357 | 0.022441384 | 7.935829163 |
| etth1 | gaussian_head | 2023 | False | SUCCESS | 0.918609381 | 0.342935384 | 0.418432683 | 0.450503141 | 0.040111512 | NA |
| etth1 | student_t_head | 2023 | False | SUCCESS | 0.866797447 | 0.336611181 | 0.415159643 | 0.443720400 | 0.023668987 | 7.957873821 |
| etth1 | gaussian_head | 2025 | True | SUCCESS | 0.918535352 | 0.344126528 | 0.420332015 | 0.453445941 | 0.038131833 | NA |
| etth1 | student_t_head | 2025 | True | SUCCESS | 0.877153873 | 0.340262627 | 0.419176817 | 0.450663358 | 0.020301414 | 7.860172272 |
| weather | official | fixed_official_checkpoint | True | SUCCESS | NA | NA | 0.240809701 | 0.309629111 | NA |  |
| weather | gaussian_head | 2021 | False | SUCCESS | 0.551677823 | 0.240093560 | 0.209886312 | 0.291154295 | 0.073642412 | NA |
| weather | student_t_head | 2021 | False | SUCCESS | 0.109830081 | 0.207504068 | 0.210133255 | 0.277232230 | 0.053236136 | 7.083511353 |
| weather | gaussian_head | 2022 | False | SUCCESS | 0.586558759 | 0.246905550 | 0.208722815 | 0.290555924 | 0.109608689 | NA |
| weather | student_t_head | 2022 | False | SUCCESS | 0.089817308 | 0.207731133 | 0.209768593 | 0.276547045 | 0.055220032 | 6.041284084 |
| weather | gaussian_head | 2023 | False | SUCCESS | 0.584344029 | 0.246712806 | 0.210377827 | 0.293027848 | 0.094286287 | NA |
| weather | student_t_head | 2023 | False | SUCCESS | 0.085737303 | 0.206663294 | 0.208533928 | 0.276530832 | 0.046608368 | 6.098047256 |
| weather | gaussian_head | 2025 | True | SUCCESS | 0.586324930 | 0.245442069 | 0.212236598 | 0.290644825 | 0.082667494 | NA |
| weather | student_t_head | 2025 | True | SUCCESS | 0.129293382 | 0.208276181 | 0.210836172 | 0.278476864 | 0.052156368 | 7.637012482 |
| electricity | official | fixed_official_checkpoint | True | SUCCESS | NA | NA | 0.200572190 | 0.318891334 | NA |  |
| electricity | gaussian_head | 2021 | False | SUCCESS | 0.437367886 | 0.209547119 | 0.166876808 | 0.278455526 | 0.042001405 | NA |
| electricity | student_t_head | 2021 | False | SUCCESS | 0.365097225 | 0.207457775 | 0.167422667 | 0.277587950 | 0.013830930 | 7.730639458 |
| electricity | gaussian_head | 2022 | False | SUCCESS | 0.442143619 | 0.209325391 | 0.167555764 | 0.278542936 | 0.036094993 | NA |
| electricity | student_t_head | 2022 | False | SUCCESS | 0.364545196 | 0.207370222 | 0.167845964 | 0.277207255 | 0.012274083 | 7.733106613 |
| electricity | gaussian_head | 2023 | False | SUCCESS | 0.438467294 | 0.209134957 | 0.167360812 | 0.278792411 | 0.034701869 | NA |
| electricity | student_t_head | 2023 | False | SUCCESS | 0.365300298 | 0.206964718 | 0.167057589 | 0.276952207 | 0.011789343 | 7.581747532 |
| electricity | gaussian_head | 2025 | True | SUCCESS | 0.440597773 | 0.209453217 | 0.167880684 | 0.278251439 | 0.038304076 | NA |
| electricity | student_t_head | 2025 | True | SUCCESS | 0.366448879 | 0.207786331 | 0.168485686 | 0.277236998 | 0.018502206 | 7.879412651 |
