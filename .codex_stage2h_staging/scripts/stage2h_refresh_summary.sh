#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/data/zza/code/BudgetedActiveChannelTokenFormer"
PYTHON="/mnt/data/zza/anaconda3/envs/bactf/bin/python"
TS="$(date +%Y%m%d_%H%M%S)"

cd "${ROOT}"
"${PYTHON}" scripts/stage2h_summarize_robustness.py 2>&1 | tee "logs/stage2h_summarize_${TS}.log"
"${PYTHON}" scripts/stage2h_formal_decision.py 2>&1 | tee "logs/stage2h_decision_${TS}.log"
