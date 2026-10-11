#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/mnt/data/zza/code/timeperceiver_meta_uq
PYTHON=/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python
cd "$ROOT"
"$PYTHON" scripts/gate4b_summarize_multiseed.py
"$PYTHON" scripts/gate4b_multiseed_formal_decision.py
