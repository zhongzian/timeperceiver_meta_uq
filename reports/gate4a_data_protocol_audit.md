# Gate 4a Data Protocol Audit

Status: **PASS**. ETTh1 uses the official `seq_len=384`, `pred_len=96`, and
seed 2025. Training uses deterministic on-the-fly random history masks with
base seed 42025; validation uses a separate fixed validation mask with seed
43025; testing reuses the Gate 3a test mask seed 2025. `batch_y` and future
targets are never used to construct the condition.
