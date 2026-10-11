# Gate 4c Raw-Time Split Audit

Status: **PASS**

- Official validation raw range: `[8256, 11520]`
- Selection: `[8256, 9648]`, windows `913`
- Embargo: `[9648, 10127]`, rows `479`
- Calibration: `[10127, 11520]`, windows `914`
- Raw timestamp overlap: `0`
- Future-target overlap: `0`

The split is chronological at raw-row level; no overlapping generated windows were randomly partitioned.
