# Gate 3a Missingness Evaluation Protocol

Gate 3a fixes reusable test input/history masks. It does not run training or missingness evaluation.

## Mask Semantics

- `1 = observed / keep`
- `0 = missing / mask`
- Shape: `[num_test_windows, seq_len, num_features]`
- Scope: only the official test `seq_x` history window.
- Future prediction labels are not read or masked by generation.

## Dataset Protocols

| Dataset | Rows | Columns | Numeric features | seq_len | pred_len | Test windows | Split rule |
|---|---:|---:|---:|---:|---:|---:|---|
| etth1 | 17420 | 8 | 7 | 384 | 96 | 2785 | Dataset_ETT_hour fixed borders: 12 months train, 4 months val, 4 months test, with seq_len overlap |
| weather | 52696 | 22 | 21 | 384 | 96 | 10444 | Dataset_Custom borders: 70% train, 10% val, 20% test, with seq_len overlap |
| electricity | 26304 | 322 | 321 | 384 | 96 | 5165 | Dataset_Custom borders: 70% train, 10% val, 20% test, with seq_len overlap |

## Normalization

The official loaders fit `StandardScaler` on the training split only, then transform all splits. Gate 3a records and saves those train-only statistics under `artifacts/normalization/` for auditability. Test statistics are not used.

## Missingness Settings

- Mask types: `random`, `block`, `channel`, `mixed`
- Missing rates: `0.1`, `0.3`, `0.5`, `0.7`
- Seed: `2025`

## Leakage Checks

| Check | Status |
|---|---|
| Mask only applies to input/history window | OK |
| Future prediction labels are not read for mask generation | OK |
| Split rules come from official loader logic | OK |
| Normalization statistics use train split only | OK |
| Same masks can be reused by baseline and innovations | OK |
| Seed is fixed | OK |
| Evaluation-time random resampling is forbidden by protocol | OK |
| Official model code modified in Gate 3a | NO |
