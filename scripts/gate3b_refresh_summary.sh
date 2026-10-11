#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate3b_summarize_full_missing_eval.py
