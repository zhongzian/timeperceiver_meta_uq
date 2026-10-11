# Gate 4a Architecture Audit

Status: **PASS**

The official model produces query-cross-attention representations with shape
`[B, enc_in, future_patch_count, d_model]`. For ETTh1, `d_model=256` and
`patch_len=12`. The adapter insertion point is immediately before the official
`output_projection`, preserving the checkpoint namespace and output head.

The upstream tree is not modified. The wrapper must preserve RevIN, patch
indices, official output projection, reshape, and inverse RevIN restoration.
