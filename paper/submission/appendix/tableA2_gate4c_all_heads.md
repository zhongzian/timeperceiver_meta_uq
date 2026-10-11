# Gate 4c All-Heads Interval Metrics

| head | method | nominal_coverage | picp | mpiw | absolute_coverage_error | interval_score | source_evidence_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gaussian | Mask-severity-conditioned conformal | 0.900000 | 0.927797 | 2.289391 | 0.027797 | 2.979145 | EV-d70d229df506cf8c |
| Gaussian | Marginal conformal | 0.900000 | 0.928261 | 2.294196 | 0.028261 | 2.979930 | EV-724465b13e16f2af |
| Gaussian | Native | 0.900000 | 0.904533 | 2.040512 | 0.004533 | 2.936425 | EV-666b448a2046df87 |
| Gaussian | Mask-severity-conditioned conformal | 0.950000 | 0.964530 | 2.986506 | 0.014530 | 3.598434 | EV-144fdbc443148140 |
| Gaussian | Marginal conformal | 0.950000 | 0.964711 | 2.991181 | 0.014711 | 3.598406 | EV-d10c8172f6cb5873 |
| Gaussian | Native | 0.950000 | 0.937263 | 2.431420 | 0.012737 | 3.571486 | EV-8ba90ed7caad12e1 |
| Student-t | Mask-severity-conditioned conformal | 0.900000 | 0.927136 | 2.262613 | 0.027136 | 2.989775 | EV-718cff6dc517b162 |
| Student-t | Marginal conformal | 0.900000 | 0.927636 | 2.267989 | 0.027636 | 2.990528 | EV-00474e8f67fb1a27 |
| Student-t | Native | 0.900000 | 0.897581 | 1.953189 | 0.002419 | 2.946898 | EV-368a160f3e940999 |
| Student-t | Mask-severity-conditioned conformal | 0.950000 | 0.964262 | 2.993500 | 0.014262 | 3.625242 | EV-77a5dcecc76dfe6f |
| Student-t | Marginal conformal | 0.950000 | 0.964505 | 2.998730 | 0.014505 | 3.625259 | EV-095f64f7ca76f766 |
| Student-t | Native | 0.950000 | 0.937700 | 2.424187 | 0.012300 | 3.596873 | EV-5b5deaf425596c67 |

Note: Strict raw-time split with a 479-row embargo and an independently retrained probability head. Conformal intervals widened and worsened absolute coverage error; the negative result is retained.
