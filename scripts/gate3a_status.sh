#!/usr/bin/env bash
set -Eeuo pipefail

cd /mnt/data/zza/code/timeperceiver_meta_uq

echo "===== Gate 3a status ====="
cat reports/gate3a_status.json 2>/dev/null || echo "NO reports/gate3a_status.json"
echo
echo "===== Gate 3a verify status ====="
cat reports/gate3a_verify_status.json 2>/dev/null || echo "NO reports/gate3a_verify_status.json"
echo
echo "===== Gate 3a inventory head ====="
head -n 20 reports/gate3a_mask_inventory.csv 2>/dev/null || true
