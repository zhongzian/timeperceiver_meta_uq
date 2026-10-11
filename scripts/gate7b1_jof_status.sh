#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS="$ROOT/results/gate7b1_jof_status.json"
PACKAGE="$ROOT/paper/venues/journal_of_forecasting/final_submission_package"
TARBALL="$ROOT/paper/venues/journal_of_forecasting/journal_of_forecasting_submission_package.tar.gz"

printf '%s\n' "===== Gate 7B1 status ====="
if [ -f "$STATUS" ]; then
  cat "$STATUS"
else
  printf '%s\n' "status file not found"
fi

printf '%s\n' "===== Package ====="
if [ -d "$PACKAGE" ]; then
  find "$PACKAGE" -type f | wc -l
else
  printf '%s\n' "package directory not found"
fi

if [ -f "$TARBALL" ]; then
  sha256sum "$TARBALL"
else
  printf '%s\n' "tarball not found"
fi
