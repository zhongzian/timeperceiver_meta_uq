#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/data/zza/code/timeperceiver_meta_uq
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate4b_summarize_expanded_matrix.py
/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python scripts/gate4b_expanded_formal_decision.py
