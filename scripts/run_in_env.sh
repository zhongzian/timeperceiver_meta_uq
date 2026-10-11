#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.project_env"

if [[ ! -s "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; run bash scripts/gate0_init.sh first." >&2
  exit 2
fi

ENV_NAME="$(tr -d '[:space:]' < "${ENV_FILE}")"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Empty conda environment name in ${ENV_FILE}" >&2
  exit 2
fi

if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
  CONDA_BIN="${CONDA_EXE}"
elif [[ -x "/mnt/data/zza/anaconda3/bin/conda" ]]; then
  CONDA_BIN="/mnt/data/zza/anaconda3/bin/conda"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
else
  echo "conda is not available." >&2
  exit 127
fi

exec "${CONDA_BIN}" run -n "${ENV_NAME}" "$@"
