#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/mnt/data/zza/code/timeperceiver_meta_uq"
ENV_NAME="timeperceiver_meta_uq"
PY_BIN="/mnt/data/zza/anaconda3/envs/timeperceiver_meta_uq/bin/python"
TARGET_CVD="${TARGET_CVD:-0}"
MIN_FREE_MIB="${MIN_FREE_MIB:-}"
FAILED_RUN="runs/timeperceiver__electricity__official_baseline__seed2025__20260701_212908"
DATASET="electricity"
MODEL="timeperceiver"
MODE="official_baseline"
SEED="2025"

cd "${PROJECT_ROOT}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${MODEL}__${DATASET}__${MODE}__seed${SEED}__${TS}"
RUN_DIR="runs/${RUN_ID}"
LOG_DIR="logs/gate2/${TS}_resume_electricity"
MAIN_LOG="${LOG_DIR}/resume_electricity.log"
mkdir -p "${RUN_DIR}/checkpoints" "${LOG_DIR}" reports runtime/locks

exec > >(tee -a "${MAIN_LOG}") 2>&1

status_json() {
  local status="$1"
  local exit_code="$2"
  local error_summary="$3"
  local end_time="${4:-}"
  cat > "${RUN_DIR}/status.json" <<EOF
{
  "dataset": "electricity",
  "end_time": "${end_time}",
  "error_summary": "${error_summary}",
  "exit_code": ${exit_code},
  "experiment_id": "${RUN_ID}",
  "metrics_path": "${PROJECT_ROOT}/${RUN_DIR}/metrics.json",
  "run_dir": "${PROJECT_ROOT}/${RUN_DIR}",
  "start_time": "${START_TIME}",
  "status": "${status}"
}
EOF
}

write_placeholder_metrics() {
  local exit_code="$1"
  local elapsed="$2"
  cat > "${RUN_DIR}/metrics.json" <<EOF
{
  "best_epoch": null,
  "checkpoint_path": null,
  "dataset": "electricity",
  "elapsed_seconds": ${elapsed},
  "final_epoch": null,
  "mae": null,
  "mape": null,
  "metrics_source": null,
  "mode": "official_baseline",
  "mse": null,
  "mspe": null,
  "official_protocol": true,
  "paper_reference": {},
  "parse_warnings": [],
  "rmse": null,
  "seed": 2025,
  "train_exit_code": ${exit_code}
}
EOF
}

update_reports() {
  /mnt/data/zza/anaconda3/bin/conda run -n "${ENV_NAME}" python scripts/gate2_official_runner.py --action reports || true
}

echo "===== Gate 2.1 Electricity resume started at $(date) ====="
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"

START_TIME="$(date -Iseconds)"
START_SEC="$(date +%s)"
status_json "WAITING_RESOURCE" "null" "created retry run; waiting for RTX 4090 free memory" ""
write_placeholder_metrics "null" "0"

if [ ! -d "${FAILED_RUN}" ]; then
  status_json "DATA_ERROR" "10" "failed run not found: ${FAILED_RUN}" "$(date -Iseconds)"
  exit 10
fi
if [ ! -f "${FAILED_RUN}/command.sh" ]; then
  status_json "DATA_ERROR" "11" "failed command.sh not found" "$(date -Iseconds)"
  exit 11
fi
if [ ! -f "${FAILED_RUN}/resolved_config.yaml" ]; then
  status_json "DATA_ERROR" "12" "failed resolved_config.yaml not found" "$(date -Iseconds)"
  exit 12
fi

cp "${FAILED_RUN}/resolved_config.yaml" "${RUN_DIR}/original_failed_resolved_config.yaml"
cp "${FAILED_RUN}/command.sh" "${RUN_DIR}/original_failed_command.sh"

"${PY_BIN}" - "${FAILED_RUN}/resolved_config.yaml" "${RUN_DIR}/resolved_config.yaml" "${RUN_ID}" "${PROJECT_ROOT}/${RUN_DIR}" <<'PY'
from pathlib import Path
import sys

old_path = Path(sys.argv[1])
new_path = Path(sys.argv[2])
run_id = sys.argv[3]
run_dir = sys.argv[4]
replacements = {
    "experiment_id": run_id,
    "run_dir": run_dir,
    "selected_cuda_visible_devices": "TARGET_CVD_AT_LAUNCH",
    "selected_gpu_index_before_visibility": "TARGET_CVD_AT_LAUNCH",
    "official_gpu_argument": "0",
}
lines = []
for line in old_path.read_text().splitlines():
    key = line.split(":", 1)[0] if ":" in line else None
    if key in replacements:
        value = replacements[key]
        if key in {"selected_gpu_index_before_visibility", "official_gpu_argument"}:
            lines.append(f"{key}: {value}")
        else:
            lines.append(f'{key}: "{value}"')
    else:
        lines.append(line)
lines.append('retry_from: "runs/timeperceiver__electricity__official_baseline__seed2025__20260701_212908"')
lines.append('retry_reason: "Gate 2.1 retry after OOM without changing official formal config"')
new_path.write_text("\n".join(lines) + "\n")
PY

echo "===== Environment snapshot ====="
{
  date
  hostname
  whoami
  echo "CONDA_ENV=${ENV_NAME}"
  "${PY_BIN}" --version
  CUDA_VISIBLE_DEVICES="${TARGET_CVD}" "${PY_BIN}" -c "import torch; print('torch', torch.__version__); print('torch_cuda_runtime', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader || true
} | tee "${RUN_DIR}/env.txt"

echo "===== Git state ====="
{
  git -C upstream/TimePerceiver rev-parse HEAD
  git -C upstream/TimePerceiver status --short
  git -C upstream/TimePerceiver diff --stat
} | tee "${RUN_DIR}/git_state.txt"
git -C upstream/TimePerceiver diff --binary > "${RUN_DIR}/code.diff" || true

echo "===== Build retry command from failed command ====="
RUN_LINE="$(grep -E 'python .*run.py|python3 .*run.py|run.py' "${FAILED_RUN}/command.sh" | tail -n 1 || true)"
if [ -z "${RUN_LINE}" ]; then
  status_json "CONFIG_ERROR" "13" "cannot find run.py command in failed command.sh" "$(date -Iseconds)"
  update_reports
  exit 13
fi

OLD_ABS_RUN_DIR="${PROJECT_ROOT}/${FAILED_RUN}"
NEW_ABS_RUN_DIR="${PROJECT_ROOT}/${RUN_DIR}"
RUN_LINE="${RUN_LINE/${OLD_ABS_RUN_DIR}/${NEW_ABS_RUN_DIR}}"
RUN_LINE="${RUN_LINE/--checkpoints ${OLD_ABS_RUN_DIR}\/checkpoints/--checkpoints ${NEW_ABS_RUN_DIR}\/checkpoints}"
RUN_LINE="${RUN_LINE/--gpu 1/--gpu 0}"
RUN_LINE="${RUN_LINE/--gpu 0/--gpu 0}"

"${PY_BIN}" - "${FAILED_RUN}/command.sh" "${RUN_LINE}" <<'PY'
import shlex
import sys
from pathlib import Path

old_text = Path(sys.argv[1]).read_text().splitlines()
old_line = [line for line in old_text if "run.py" in line][-1]
new_line = sys.argv[2]
protected = {
    "--data_path", "--model_id", "--model", "--data", "--features", "--target", "--freq",
    "--seq_len", "--label_len", "--pred_len", "--enc_in", "--d_model", "--d_ff", "--n_heads",
    "--itr", "--num_latents", "--latent_dim", "--latent_d_ff", "--dropout", "--standard",
    "--generalized", "--use_latent", "--num_latent_blocks", "--patch_len", "--separate_ratio",
    "--batch_size", "--train_epochs", "--warmup_epochs", "--patience", "--learning_rate",
    "--weight_decay", "--num_workers"
}

def args_map(line):
    parts = shlex.split(line)
    out = {}
    for i, part in enumerate(parts):
        if part in protected and i + 1 < len(parts):
            out[part] = parts[i + 1]
    return out

old = args_map(old_line)
new = args_map(new_line)
diff = {k: (old.get(k), new.get(k)) for k in protected if old.get(k) != new.get(k)}
if diff:
    raise SystemExit(f"protected official config changed: {diff}")
PY

echo "===== GPU wait policy ====="
echo "TARGET_CVD=${TARGET_CVD}"
TARGET_GPU_NAME="$(CUDA_VISIBLE_DEVICES="${TARGET_CVD}" "${PY_BIN}" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_CUDA')" | sed '/^$/d' | tail -n 1)"
if [ -z "${TARGET_GPU_NAME}" ] || [ "${TARGET_GPU_NAME}" = "NO_CUDA" ]; then
  status_json "CONFIG_ERROR" "15" "could not resolve target CUDA_VISIBLE_DEVICES=${TARGET_CVD} to a CUDA GPU" "$(date -Iseconds)"
  update_reports
  exit 15
fi
echo "TARGET_GPU_NAME=${TARGET_GPU_NAME}"
MAX_WAIT_MINUTES="${MAX_WAIT_MINUTES:-720}"
POLL_SECONDS="${POLL_SECONDS:-120}"
WAITED=0
OOM_RETRY_COUNT=0
MAX_OOM_RETRIES="${MAX_OOM_RETRIES:-2}"

while true; do
  echo "----- GPU poll at $(date) -----"
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader | tee -a "${LOG_DIR}/gpu_poll.log" || true
  TARGET_NVIDIA_INDEX="$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F, -v name="${TARGET_GPU_NAME}" '$2 ~ name {gsub(/ /,"",$1); print $1; exit}')"
  FREE_TARGET="$(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader,nounits | awk -F, -v name="${TARGET_GPU_NAME}" '$2 ~ name {gsub(/ /,"",$3); print $3; exit}')"
  TOTAL_TARGET="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | awk -F, -v name="${TARGET_GPU_NAME}" '$2 ~ name {gsub(/ /,"",$3); print $3; exit}')"
  echo "target cvd=${TARGET_CVD} name=${TARGET_GPU_NAME:-NA} nvidia_index=${TARGET_NVIDIA_INDEX:-NA} free=${FREE_TARGET:-NA} MiB total=${TOTAL_TARGET:-NA} MiB"

  if [ -n "${TARGET_NVIDIA_INDEX:-}" ] && [ -n "${FREE_TARGET:-}" ] && [ -n "${TOTAL_TARGET:-}" ]; then
    if [ -n "${MIN_FREE_MIB}" ]; then
      NEED="${MIN_FREE_MIB}"
    else
      NEED=$(( TOTAL_TARGET * 90 / 100 ))
    fi
    if [ "${FREE_TARGET}" -ge "${NEED}" ]; then
      echo "GPU resource looks sufficient; start Electricity retry on ${TARGET_GPU_NAME} CVD ${TARGET_CVD}."
      break
    fi
    status_json "WAITING_RESOURCE" "null" "${TARGET_GPU_NAME} free ${FREE_TARGET} MiB below required ${NEED} MiB" ""
    update_reports
  else
    status_json "WAITING_RESOURCE" "null" "could not read target GPU free memory for CVD ${TARGET_CVD}; waiting conservatively" ""
    update_reports
  fi

  if [ "${WAITED}" -ge "${MAX_WAIT_MINUTES}" ]; then
    END_TIME="$(date -Iseconds)"
    ELAPSED=$(( $(date +%s) - START_SEC ))
    status_json "WAITING_RESOURCE" "null" "target GPU not sufficiently free within wait window" "${END_TIME}"
    write_placeholder_metrics "null" "${ELAPSED}"
    update_reports
    exit 30
  fi

  sleep "${POLL_SECONDS}"
  WAITED=$(( WAITED + POLL_SECONDS / 60 ))
done

"${PY_BIN}" - "${RUN_DIR}/resolved_config.yaml" "${TARGET_CVD}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
idx = sys.argv[2]
text = path.read_text()
text = text.replace('"TARGET_CVD_AT_LAUNCH"', f'"{idx}"')
text = text.replace("TARGET_CVD_AT_LAUNCH", idx)
path.write_text(text)
PY

cat > "${RUN_DIR}/command.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd "${NEW_ABS_RUN_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/upstream/TimePerceiver\${PYTHONPATH:+:\$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${TARGET_CVD}"
${RUN_LINE}
EOF
chmod +x "${RUN_DIR}/command.sh"

echo "===== Retry command ====="
cat "${RUN_DIR}/command.sh"

echo "===== Start training ====="
TRAIN_START="$(date -Iseconds)"
TRAIN_START_SEC="$(date +%s)"
status_json "RUNNING" "null" "" ""
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader > "${RUN_DIR}/gpu_before.txt" || true

set +e
bash "${RUN_DIR}/command.sh" > "${RUN_DIR}/train.log" 2>&1
EXIT_CODE=$?
set -e

END_TIME="$(date -Iseconds)"
ELAPSED=$(( $(date +%s) - TRAIN_START_SEC ))
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader > "${RUN_DIR}/gpu_after.txt" || true

STATUS="UNKNOWN_ERROR"
ERROR_SUMMARY=""
if [ "${EXIT_CODE}" -eq 0 ]; then
  STATUS="SUCCESS"
elif grep -qiE "out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED" "${RUN_DIR}/train.log"; then
  STATUS="OOM"
  ERROR_SUMMARY="CUDA out of memory under unchanged official config"
else
  ERROR_SUMMARY="$(tail -n 20 "${RUN_DIR}/train.log" | tr '\n' ' ' | sed 's/"/\\"/g')"
fi

if [ "${STATUS}" = "SUCCESS" ]; then
  /mnt/data/zza/anaconda3/bin/conda run -n "${ENV_NAME}" python scripts/parse_timeperceiver_metrics.py "${RUN_DIR}" --output "${RUN_DIR}/metrics.json" || true
fi
if [ ! -s "${RUN_DIR}/metrics.json" ] || [ "${STATUS}" != "SUCCESS" ]; then
  write_placeholder_metrics "${EXIT_CODE}" "${ELAPSED}"
fi

status_json "${STATUS}" "${EXIT_CODE}" "${ERROR_SUMMARY}" "${END_TIME}"
cat > "${RUN_DIR}/resource_usage.json" <<EOF
{
  "cuda_visible_devices": "${TARGET_CVD}",
  "dataset": "electricity",
  "elapsed_seconds": ${ELAPSED},
  "end_time": "${END_TIME}",
  "experiment_id": "${RUN_ID}",
  "gpu_after_path": "${PROJECT_ROOT}/${RUN_DIR}/gpu_after.txt",
  "gpu_before_path": "${PROJECT_ROOT}/${RUN_DIR}/gpu_before.txt",
  "gpu_policy": "wait for target CUDA_VISIBLE_DEVICES mapping to have enough free memory; pass --gpu 0 to official run.py",
  "target_gpu_name": "${TARGET_GPU_NAME}",
  "target_nvidia_smi_index": "${TARGET_NVIDIA_INDEX:-}",
  "oom_retry_count": ${OOM_RETRY_COUNT},
  "start_time": "${TRAIN_START}"
}
EOF

update_reports
echo "===== Final Gate 2 status ====="
cat reports/gate2_status.json
echo "===== Finished at $(date) with STATUS=${STATUS} ====="

if [ "${STATUS}" = "SUCCESS" ]; then
  exit 0
elif [ "${STATUS}" = "OOM" ]; then
  exit 40
else
  exit "${EXIT_CODE}"
fi
