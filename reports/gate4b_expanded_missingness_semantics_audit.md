# Gate 4b Expanded Missingness Semantics Audit

| missing_type | generator_function | semantic_definition | actual_rate_rule | future_target_used |
|---|---|---|---|---|
| random | gate3a.make_mask->random_mask | independent element masking with P(missing)=target_rate | 1-mask.mean() | False |
| block | gate3a.make_mask->block_mask | 2%-20% temporal blocks, sometimes all channels and sometimes sampled channel subsets, until target count reached | 1-mask.mean() | False |
| channel | gate3a.make_mask->channel_mask | round(rate*C) whole channels per window, capped at C-1 | 1-mask.mean() | False |
| mixed | gate3a.make_mask->mixed_mask | channel(rate/3) union block(rate/3), then random observed entries to target total | 1-mask.mean() | False |
