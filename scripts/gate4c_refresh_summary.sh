#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
cd "$ROOT"

"$PYTHON" scripts/gate4c_summarize_minimal.py
"$PYTHON" scripts/gate4c_formal_decision.py
