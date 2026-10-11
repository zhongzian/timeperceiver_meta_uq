#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate3b_smoke_missing_eval.py --dataset etth1 --mask_path artifacts/masks/etth1/random/rate_0.3/seed_2025.npz --mask_meta artifacts/masks/etth1/random/rate_0.3/seed_2025.json --run_dir /mnt/data/zza/code/timeperceiver_meta_uq/runs/timeperceiver__etth1__official_missing_eval_smoke__random_rate0.3__seed2025__20260709_153942 --max_eval_batches 8 --eval_batch_size 32 --device cpu
