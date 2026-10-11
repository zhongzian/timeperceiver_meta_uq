#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
LOCK="$ROOT/runtime/locks/gate5b_continue_after_smokes.lock"

cd "$ROOT"
mkdir -p runtime/locks logs
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "DUPLICATE_GATE5B_CONTINUATION_BLOCKED"
  exit 3
fi

echo "===== Gate 5b prerequisite audit ====="
"$PYTHON" - <<'PY'
import json
from pathlib import Path
checks = {
    "source": (Path("results/gate5b_patchtst_source_audit.json"), "PASS"),
    "environment": (Path("results/gate5b_environment_audit.json"), "ENV_COMPATIBLE"),
    "architecture": (Path("results/gate5b_patchtst_architecture_audit.json"), "PASS"),
    "data": (Path("results/gate5b_data_protocol_validation.json"), "PASS"),
}
for name, (path, expected) in checks.items():
    actual = json.loads(path.read_text()).get("status") if path.is_file() else "MISSING"
    if actual != expected:
        raise SystemExit(f"{name} gate failed: {actual}, expected {expected}")
print("PREREQUISITE_AUDITS_PASS")
PY

echo "===== Wait for and run remaining backbone smokes ====="
"$PYTHON" scripts/gate5b_run_backbone_smokes.py --wait-seconds 604800

echo "===== Build sequential queues ====="
"$PYTHON" scripts/gate5b_prepare_queues.py

echo "===== Validate both queues ====="
"$PYTHON" scripts/gate5b_validate_queues.py

echo "===== Recheck protected upstream tracked diffs ====="
test -z "$(git -C upstream/TimePerceiver diff --name-only)"
test -z "$(git -C upstream/PatchTST diff --name-only)"

echo "===== Start Gate 5b0 backbone scheduler nonblocking ====="
bash scripts/gate5b_scheduler.sh start configs/queues/gate5b_patchtst_backbones.yaml

