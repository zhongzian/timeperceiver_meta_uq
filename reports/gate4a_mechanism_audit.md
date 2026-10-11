# Gate 4a Mechanism Audit

Status: **PASS**

No training was run. Existing checkpoints and fixed masks were inspected.

## Condition diversity

- Test global-rate std: `0.00881173`
- Mean z feature std across samples: `0.00350057`
- Gamma mean absolute magnitude: `0.02858514`
- Beta mean absolute magnitude: `0.00901651`

## Parameter updates

- Static relative update L2: `0.072442`
- Missing-aware relative update L2: `0.575428`

## Ordering

Validation and test rankings agree: `true`. Static is better on both.

## Interpretation

the fixed random-0.3 condition has limited but nonzero sample diversity, and trained FiLM modulation is measurably nonzero; the failure is therefore closer to ineffective conditioning/injection or optimization than to a completely constant condition. This is a single-seed mechanism diagnosis, not a statistical claim. Per-step gradients and complete epoch curves were not persisted and cannot be reconstructed exactly.
