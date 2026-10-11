# Reproducibility

- Upstream TimePerceiver commit: `7e30cc07b51c709f408409fd60a34c81ae8990be`
- Upstream PatchTST commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- Python: 3.9.25
- PyTorch: 2.1.0+cu121
- CUDA runtime: 12.1
- Seed scope: official checkpoint seed 2025; probability-head training seeds 2021, 2022, 2023, and 2025; fixed test-mask seed 2025
- Official train/validation/test boundaries and train-only normalization are retained.
- Validation and test masks are fixed and shared by matched methods; future targets are never masked or used for selection.
- The evidence package records checkpoint, mask, training-protocol, metric-protocol, and upstream commit hashes.

Historical runs that did not apply the input mask remain in the internal audit registry but are excluded from every paper metric, table, figure, claim, and reproduction asset.
