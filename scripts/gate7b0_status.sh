#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"

cd "$ROOT"

echo "===== Gate 7B0 status ====="
"$PYTHON" -c 'import json; d=json.load(open("results/gate7b0_status.json", encoding="utf-8")); print(json.dumps({k:d[k] for k in ["status","primary_venue","backup_venue_1","backup_venue_2","manual_verification_required","ready_for_venue_choice"]}, indent=2))'

echo "===== Final gates ====="
"$PYTHON" -c 'import json; d=json.load(open("results/gate7b0_status.json", encoding="utf-8")); print("\n".join(f"{k}={d[k]}" for k in sorted(d) if k.startswith("GATE7B0_")))'

echo "===== Source validation ====="
"$PYTHON" -c 'import json; d=json.load(open("results/gate7b0_source_validation.json", encoding="utf-8")); print(json.dumps({k:d[k] for k in ["source_validation_pass","unsupported_source_claims","fabricated_metadata","score_arithmetic_errors"]}, indent=2))'

echo "===== Protected asset hashes ====="
sha256sum -c artifacts/gate7b0_venue_sources/protected_hashes.sha256

echo "===== TimePerceiver tracked status ====="
git -C upstream/TimePerceiver status --short --untracked-files=no

echo "===== PatchTST tracked status ====="
git -C upstream/PatchTST status --short --untracked-files=no
