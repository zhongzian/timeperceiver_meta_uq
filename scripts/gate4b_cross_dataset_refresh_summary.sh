#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
"$PYTHON" scripts/gate4b_summarize_cross_dataset.py
"$PYTHON" scripts/gate4b_cross_dataset_formal_decision.py
