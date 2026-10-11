#!/usr/bin/env bash
# eval-only; no optimizer/backward
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate3b_official_missing_eval.py --task-file runtime/gate3b_parity_task.json --run-dir runs/timeperceiver__etth1__official_missing_eval_parity__random_rate0.3__seed2025__20260711_185031 --device cpu --max-eval-batches 8 --eval-batch-size 32
