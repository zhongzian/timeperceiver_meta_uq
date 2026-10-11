#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OFFICIAL_URL="https://github.com/efficient-learning-lab/TimePerceiver.git"
TARGET_ENV_BASE="timeperceiver_meta_uq"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="gate0__environment_init__${TIMESTAMP}"
LOG_DIR="${ROOT_DIR}/logs/gate0/${RUN_ID}"
mkdir -p "${LOG_DIR}" environment reports upstream configs outputs runs results artifacts runtime/locks data

EXIT_CODES="${LOG_DIR}/exit_codes.tsv"
: > "${EXIT_CODES}"

BLOCKERS=()
WARNINGS=()
AGENTS_READ=("/mnt/data/zza/code/AGENTS.md" "${ROOT_DIR}/AGENTS.md")
while IFS= read -r override; do
  [[ -n "${override}" ]] && AGENTS_READ+=("${override}")
done < <(find "${ROOT_DIR}" -name AGENTS.override.md -print 2>/dev/null | sort)

log_code() {
  local name="$1"
  local code="$2"
  printf "%s\t%s\n" "${name}" "${code}" >> "${EXIT_CODES}"
}

record_blocker() {
  BLOCKERS+=("$1")
}

record_warning() {
  WARNINGS+=("$1")
}

run_capture() {
  local name="$1"
  local outfile="$2"
  shift 2
  {
    printf "\n===== %s =====\n" "${name}"
    printf "command:"
    printf " %q" "$@"
    printf "\n"
    "$@"
  } >> "${outfile}" 2>&1
  local code=$?
  log_code "${name}" "${code}"
  return "${code}"
}

run_shell_capture() {
  local name="$1"
  local outfile="$2"
  local command_text="$3"
  {
    printf "\n===== %s =====\n" "${name}"
    printf "command: %s\n" "${command_text}"
    bash -lc "${command_text}"
  } >> "${outfile}" 2>&1
  local code=$?
  log_code "${name}" "${code}"
  return "${code}"
}

capture_optional() {
  local name="$1"
  local outfile="$2"
  shift 2
  if command -v "$1" >/dev/null 2>&1; then
    run_capture "${name}" "${outfile}" "$@" || true
  else
    {
      printf "\n===== %s =====\n" "${name}"
      printf "%s NOT_AVAILABLE\n" "$1"
    } >> "${outfile}"
    log_code "${name}" "NOT_AVAILABLE"
  fi
}

find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    printf "%s\n" "${CONDA_EXE}"
  elif [[ -x "/mnt/data/zza/anaconda3/bin/conda" ]]; then
    printf "%s\n" "/mnt/data/zza/anaconda3/bin/conda"
  elif command -v conda >/dev/null 2>&1; then
    command -v conda
  else
    return 1
  fi
}

json_string_array() {
  python3 - "$@" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:], ensure_ascii=False))
PY
}

SYSTEM_LOG="${LOG_DIR}/system_inventory.log"
GPU_LOG="${LOG_DIR}/gpu_inventory.log"
CONDA_LOG="${LOG_DIR}/conda_setup.log"
DEPS_LOG="${LOG_DIR}/dependency_install.log"
REPO_LOG="${LOG_DIR}/repository_setup.log"
VALIDATION_LOG="${LOG_DIR}/validation.log"

{
  echo "Gate 0 run id: ${RUN_ID}"
  echo "Project root: ${ROOT_DIR}"
  echo "Timestamp: ${TIMESTAMP}"
  echo "AGENTS files read:"
  printf "%s\n" "${AGENTS_READ[@]}"
} > "${SYSTEM_LOG}"

run_capture "pwd" "${SYSTEM_LOG}" pwd || true
run_capture "date" "${SYSTEM_LOG}" date || true
run_capture "hostname" "${SYSTEM_LOG}" hostname || true
run_capture "whoami" "${SYSTEM_LOG}" whoami || true
run_capture "uname" "${SYSTEM_LOG}" uname -a || true
run_capture "os_release" "${SYSTEM_LOG}" cat /etc/os-release || true
run_capture "df_h" "${SYSTEM_LOG}" df -h || true
run_capture "free_h" "${SYSTEM_LOG}" free -h || true
capture_optional "lscpu" "${SYSTEM_LOG}" lscpu
capture_optional "git_version" "${SYSTEM_LOG}" git --version
capture_optional "gcc_version" "${SYSTEM_LOG}" gcc --version
capture_optional "gpp_version" "${SYSTEM_LOG}" g++ --version
if CONDA_BIN="$(find_conda)"; then
  run_capture "conda_version" "${SYSTEM_LOG}" "${CONDA_BIN}" --version || true
else
  echo "conda NOT_AVAILABLE" >> "${SYSTEM_LOG}"
  log_code "conda_version" "NOT_AVAILABLE"
fi
capture_optional "python_version_system" "${SYSTEM_LOG}" python --version
capture_optional "pip_version_system" "${SYSTEM_LOG}" pip --version
capture_optional "nvcc_version" "${SYSTEM_LOG}" nvcc --version
capture_optional "nvidia_smi" "${SYSTEM_LOG}" nvidia-smi
for scheduler_cmd in squeue sinfo sbatch qstat qsub; do
  if command -v "${scheduler_cmd}" >/dev/null 2>&1; then
    printf "%s=%s\n" "${scheduler_cmd}" "$(command -v "${scheduler_cmd}")" >> "${SYSTEM_LOG}"
  else
    printf "%s=NOT_AVAILABLE\n" "${scheduler_cmd}" >> "${SYSTEM_LOG}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  run_shell_capture "gpu_query" "${GPU_LOG}" "nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader" || true
  run_capture "nvidia_smi_pmon" "${GPU_LOG}" nvidia-smi pmon -c 1 || true
  run_capture "nvidia_smi_topo" "${GPU_LOG}" nvidia-smi topo -m || true
  nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free --format=csv,noheader > environment/gpu_inventory.txt 2>>"${GPU_LOG}" || true
else
  echo "nvidia-smi NOT_AVAILABLE" | tee environment/gpu_inventory.txt >> "${GPU_LOG}"
  log_code "gpu_query" "NOT_AVAILABLE"
  record_blocker "nvidia-smi is not available; GPU inventory could not be recorded."
fi

REPO_PATH="${ROOT_DIR}/upstream/TimePerceiver"
if [[ -e "${REPO_PATH}" && ! -d "${REPO_PATH}/.git" ]]; then
  ALT_PATH="${ROOT_DIR}/upstream/TimePerceiver_official_${TIMESTAMP}"
  record_warning "upstream/TimePerceiver exists but is not a Git repository; using ${ALT_PATH}."
  REPO_PATH="${ALT_PATH}"
fi

if [[ -d "${REPO_PATH}/.git" ]]; then
  run_capture "upstream_status" "${REPO_LOG}" git -C "${REPO_PATH}" status --short || true
  run_capture "upstream_remote" "${REPO_LOG}" git -C "${REPO_PATH}" remote -v || true
  run_capture "upstream_head" "${REPO_LOG}" git -C "${REPO_PATH}" rev-parse HEAD || true
  run_capture "upstream_fetch" "${REPO_LOG}" git -C "${REPO_PATH}" fetch --all --tags || record_warning "git fetch failed; using existing upstream checkout."
else
  mkdir -p "$(dirname "${REPO_PATH}")"
  run_capture "git_clone_timeperceiver" "${REPO_LOG}" git clone "${OFFICIAL_URL}" "${REPO_PATH}" || record_blocker "TimePerceiver official repository clone failed."
fi

REPO_COMMIT="UNKNOWN"
REPO_BRANCH="UNKNOWN"
REPO_REMOTE_HEAD="UNKNOWN"
REPO_LICENSE="UNKNOWN"
REPO_STATUS=""
if [[ -d "${REPO_PATH}/.git" ]]; then
  REPO_COMMIT="$(git -C "${REPO_PATH}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  REPO_BRANCH="$(git -C "${REPO_PATH}" branch --show-current 2>/dev/null || echo UNKNOWN)"
  REPO_REMOTE_HEAD="$(git -C "${REPO_PATH}" ls-remote origin HEAD 2>/dev/null | awk '{print $1}' || echo UNKNOWN)"
  REPO_STATUS="$(git -C "${REPO_PATH}" status --short 2>/dev/null || true)"
  if [[ -f "${REPO_PATH}/LICENSE" ]]; then
    REPO_LICENSE="$(head -n 5 "${REPO_PATH}/LICENSE" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g')"
  fi
  {
    echo "official_url=${OFFICIAL_URL}"
    echo "clone_path=${REPO_PATH}"
    echo "branch=${REPO_BRANCH}"
    echo "head_commit=${REPO_COMMIT}"
    echo "remote_head_commit=${REPO_REMOTE_HEAD}"
    echo "clone_fetch_time=${TIMESTAMP}"
    echo "git_status_short:"
    printf "%s\n" "${REPO_STATUS:-CLEAN}"
    echo "git_log_1:"
    git -C "${REPO_PATH}" log -1 --oneline 2>/dev/null || true
    echo "license=${REPO_LICENSE}"
    if [[ -n "${REPO_STATUS}" ]]; then
      echo "has_uncommitted_changes=true"
    else
      echo "has_uncommitted_changes=false"
    fi
  } > environment/upstream_repo.txt
  if [[ -f "${REPO_PATH}/requirements.txt" ]]; then
    cp "${REPO_PATH}/requirements.txt" environment/requirements_upstream.txt
  else
    echo "requirements.txt NOT_FOUND" > environment/requirements_upstream.txt
  fi
else
  {
    echo "official_url=${OFFICIAL_URL}"
    echo "clone_path=${REPO_PATH}"
    echo "head_commit=UNKNOWN"
    echo "repository_status=NOT_AVAILABLE"
  } > environment/upstream_repo.txt
fi

ENV_NAME="${TARGET_ENV_BASE}"
if CONDA_BIN="$(find_conda)"; then
  "${CONDA_BIN}" env list > "${LOG_DIR}/conda_env_list.txt" 2>>"${CONDA_LOG}" || true
  if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV_BASE}"; then
    PY_VER="$("${CONDA_BIN}" run -n "${TARGET_ENV_BASE}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>>"${CONDA_LOG}" || echo UNKNOWN)"
    if [[ "${PY_VER}" == "3.9" ]]; then
      ENV_NAME="${TARGET_ENV_BASE}"
      echo "Reusing existing conda env ${ENV_NAME} with Python ${PY_VER}." >> "${CONDA_LOG}"
    else
      ENV_NAME="${TARGET_ENV_BASE}_py39_v2"
      if "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
        echo "Using existing fallback conda env ${ENV_NAME}." >> "${CONDA_LOG}"
      else
        run_capture "conda_create_fallback_env" "${CONDA_LOG}" "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.9 -y || record_blocker "Failed to create fallback conda env ${ENV_NAME}."
      fi
    fi
  else
    run_capture "conda_create_env" "${CONDA_LOG}" "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.9 -y || record_blocker "Failed to create conda env ${ENV_NAME}."
  fi
  printf "%s\n" "${ENV_NAME}" > .project_env
else
  record_blocker "conda is not available; cannot create or reuse project environment."
  ENV_NAME="UNKNOWN"
fi

if [[ "${ENV_NAME}" != "UNKNOWN" ]] && CONDA_BIN="$(find_conda)"; then
  run_capture "env_python_version" "${CONDA_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python --version || true
  if "${CONDA_BIN}" run -n "${ENV_NAME}" python -c 'import torch' >/dev/null 2>&1; then
    echo "torch already importable; not reinstalling torch." >> "${DEPS_LOG}"
    log_code "pip_install_torch" "SKIPPED_ALREADY_AVAILABLE"
  else
    run_capture "pip_install_torch" "${DEPS_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 || record_blocker "Failed to install torch==2.1.0 CUDA 12.1 wheel."
  fi

  if [[ -f environment/requirements_upstream.txt ]] && ! grep -q "NOT_FOUND" environment/requirements_upstream.txt; then
    python3 - <<'PY'
from pathlib import Path
src = Path("environment/requirements_upstream.txt")
dst = Path("environment/requirements_without_torch.txt")
lines = []
for line in src.read_text().splitlines():
    stripped = line.strip().lower()
    if stripped.startswith("torch") or stripped.startswith("torchvision") or stripped.startswith("torchaudio"):
        continue
    lines.append(line)
dst.write_text("\n".join(lines) + ("\n" if lines else ""))
PY
    if [[ -s environment/requirements_without_torch.txt ]]; then
      run_capture "pip_install_requirements_without_torch" "${DEPS_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r environment/requirements_without_torch.txt || record_warning "Installing upstream non-torch requirements returned non-zero; import validation will decide Gate 0 status."
    else
      log_code "pip_install_requirements_without_torch" "SKIPPED_EMPTY"
    fi
  fi

  run_capture "python_version_snapshot" "${DEPS_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python --version || true
  "${CONDA_BIN}" run -n "${ENV_NAME}" python --version > environment/python_version.txt 2>&1 || true
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip freeze > environment/pip_freeze.txt 2>>"${DEPS_LOG}" || true
  "${CONDA_BIN}" list -n "${ENV_NAME}" > environment/conda_list.txt 2>>"${DEPS_LOG}" || true
  "${CONDA_BIN}" list -n "${ENV_NAME}" --explicit > environment/conda_explicit.txt 2>>"${DEPS_LOG}" || true
  cp environment/requirements_without_torch.txt environment/requirements_effective.txt 2>/dev/null || touch environment/requirements_effective.txt

  run_shell_capture "import_validation" "${VALIDATION_LOG}" "${CONDA_BIN} run -n ${ENV_NAME} python - <<'PY'
mods = ['torch','numpy','pandas','scipy','sklearn','einops','matplotlib','sktime','sympy','pywt','reformer_pytorch','local_attention']
missing = []
for mod in mods:
    try:
        __import__(mod)
        print(f'{mod}: OK')
    except Exception as exc:
        print(f'{mod}: FAIL {exc!r}')
        missing.append(mod)
raise SystemExit(1 if missing else 0)
PY" || record_blocker "One or more required Python imports failed."

  if [[ -d "${REPO_PATH}" ]]; then
    run_capture "compileall_upstream" "${VALIDATION_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python -m compileall "${REPO_PATH}" || record_warning "compileall returned non-zero."
    if [[ -f "${REPO_PATH}/run.py" ]]; then
      run_capture "run_py_help" "${VALIDATION_LOG}" "${CONDA_BIN}" run -n "${ENV_NAME}" python "${REPO_PATH}/run.py" --help || record_warning "run.py --help returned non-zero; official entry may not support --help."
    else
      record_warning "upstream run.py not found."
      log_code "run_py_help" "NOT_FOUND"
    fi
  fi

  run_shell_capture "torch_cuda_probe" "${VALIDATION_LOG}" "${CONDA_BIN} run -n ${ENV_NAME} python scripts/probe_torch_cuda.py > environment/torch_cuda_probe.json" || record_blocker "Torch/CUDA probe failed to import torch."
else
  echo "{}" > environment/torch_cuda_probe.json
fi

python3 - <<'PY'
import csv
import json
from pathlib import Path

gpu_rows = []
inv = Path("environment/gpu_inventory.txt")
if inv.exists():
    for row in csv.reader(inv.read_text().splitlines()):
        if len(row) >= 7:
            gpu_rows.append({
                "nvidia_smi_index": row[0].strip(),
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "driver_version": row[3].strip(),
                "memory_total": row[4].strip(),
                "memory_used": row[5].strip(),
                "memory_free": row[6].strip(),
            })

probe_path = Path("environment/torch_cuda_probe.json")
try:
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else {}
except Exception:
    probe = {}

mapping = {
    "note": "nvidia-smi physical indices are listed separately from PyTorch logical indices; UUID exposure is from nvidia-smi.",
    "nvidia_smi_devices": gpu_rows,
    "pytorch_devices": probe.get("devices", []),
    "cuda_visible_devices": probe.get("cuda_visible_devices", ""),
}
Path("environment/gpu_mapping.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True))
PY

PYTHON_VERSION="UNKNOWN"
TORCH_VERSION="UNKNOWN"
TORCH_CUDA_RUNTIME="UNKNOWN"
CUDA_AVAILABLE="false"
GPU_COUNT="0"
TENSOR_TEST_OK="false"
if [[ -s environment/torch_cuda_probe.json ]]; then
  read -r PYTHON_VERSION TORCH_VERSION TORCH_CUDA_RUNTIME CUDA_AVAILABLE GPU_COUNT TENSOR_TEST_OK < <(python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path("environment/torch_cuda_probe.json").read_text())
print(
    (d.get("python_version") or "UNKNOWN").split()[0],
    d.get("torch_version") or "UNKNOWN",
    d.get("torch_cuda_runtime") or "UNKNOWN",
    str(bool(d.get("cuda_available"))).lower(),
    str(d.get("device_count", 0)),
    str(bool((d.get("tensor_test") or {}).get("ok"))).lower(),
)
PY
)
fi

if [[ "${PYTHON_VERSION}" != 3.9* ]]; then
  record_blocker "Python in selected environment is not 3.9: ${PYTHON_VERSION}."
fi
if [[ "${TORCH_VERSION}" == "UNKNOWN" ]]; then
  record_blocker "PyTorch is not importable in selected environment."
fi
if [[ "${CUDA_AVAILABLE}" != "true" ]]; then
  record_blocker "torch.cuda.is_available() is not true."
fi
if [[ "${TENSOR_TEST_OK}" != "true" ]]; then
  record_blocker "Minimal CUDA tensor test did not pass."
fi
if [[ ! -d "${REPO_PATH}/.git" ]]; then
  record_blocker "Official TimePerceiver Git repository is not available."
fi

STATUS="PASS"
if [[ ${#BLOCKERS[@]} -gt 0 ]]; then
  STATUS="PARTIAL"
  if [[ "${ENV_NAME}" == "UNKNOWN" || ! -d "${REPO_PATH}/.git" ]]; then
    STATUS="BLOCKED"
  fi
fi

python3 - "${STATUS}" "${TIMESTAMP}" "${ROOT_DIR}" "${ENV_NAME}" "${PYTHON_VERSION}" "${TORCH_VERSION}" "${TORCH_CUDA_RUNTIME}" "${CUDA_AVAILABLE}" "${GPU_COUNT}" "${REPO_PATH}" "${REPO_COMMIT}" "${LOG_DIR}" "$(json_string_array "${AGENTS_READ[@]}")" "$(json_string_array "${BLOCKERS[@]}")" "$(json_string_array "${WARNINGS[@]}")" <<'PY'
import json
import sys
from pathlib import Path

(
    status,
    timestamp,
    root,
    env_name,
    pyver,
    torch_ver,
    torch_cuda,
    cuda_available,
    gpu_count,
    repo_path,
    repo_commit,
    log_dir,
    agents_json,
    blockers_json,
    warnings_json,
) = sys.argv[1:]

exit_codes = {}
for line in Path(log_dir, "exit_codes.tsv").read_text().splitlines():
    if "\t" in line:
        k, v = line.split("\t", 1)
        exit_codes[k] = v

data = {
    "gate": "Gate 0",
    "status": status,
    "timestamp": timestamp,
    "project_root": root,
    "conda_env": env_name,
    "python_version": pyver,
    "torch_version": torch_ver,
    "torch_cuda_runtime": torch_cuda,
    "cuda_available": cuda_available == "true",
    "gpu_count": int(gpu_count) if str(gpu_count).isdigit() else 0,
    "repository_path": repo_path,
    "repository_commit": repo_commit,
    "agents_read": json.loads(agents_json),
    "checks": exit_codes,
    "blockers": json.loads(blockers_json),
    "warnings": json.loads(warnings_json),
    "latest_log_dir": log_dir,
    "next_gate": "Gate 1",
}
Path("reports/gate0_status.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
PY

{
  echo "# Gate 0 Environment Report"
  echo
  echo "## 1. Stage Goal"
  echo
  echo "Initialize the TimePerceiver Meta-UQ project environment without starting Gate 1, downloading full datasets, or running formal training."
  echo
  echo "## 2. Executed Content"
  echo
  echo "- Read AGENTS files and checked for AGENTS.override.md."
  echo "- Created minimal project directories."
  echo "- Audited server, scheduler commands, GPU inventory, TimePerceiver upstream repository, Conda environment, dependencies, imports, syntax, and CUDA tensor availability."
  echo
  echo "## 3. Server Summary"
  echo
  echo "See ${SYSTEM_LOG}."
  echo
  echo "## 4. GPU Mapping"
  echo
  echo "Inventory: environment/gpu_inventory.txt"
  echo
  echo "Mapping: environment/gpu_mapping.json"
  echo
  echo "## 5. Conda Environment"
  echo
  echo "- Environment: ${ENV_NAME}"
  echo "- Python: ${PYTHON_VERSION}"
  echo
  echo "## 6. PyTorch/CUDA Compatibility"
  echo
  echo "- PyTorch: ${TORCH_VERSION}"
  echo "- CUDA runtime: ${TORCH_CUDA_RUNTIME}"
  echo "- CUDA available: ${CUDA_AVAILABLE}"
  echo "- GPU count: ${GPU_COUNT}"
  echo "- Minimal tensor test: ${TENSOR_TEST_OK}"
  echo
  echo "## 7. Official Repository"
  echo
  echo "- URL: ${OFFICIAL_URL}"
  echo "- Path: ${REPO_PATH}"
  echo "- Commit: ${REPO_COMMIT}"
  echo
  echo "## 8. Dependency Installation"
  echo
  echo "See ${DEPS_LOG} and environment snapshots under environment/."
  echo
  echo "## 9. Validation Commands And Exit Codes"
  echo
  echo '```text'
  cat "${EXIT_CODES}"
  echo '```'
  echo
  echo "## 10. Gate Decision"
  echo
  echo "${STATUS}"
  echo
  echo "## 11. Unresolved Issues"
  if [[ ${#BLOCKERS[@]} -eq 0 && ${#WARNINGS[@]} -eq 0 ]]; then
    echo
    echo "None."
  else
    echo
    for item in "${BLOCKERS[@]}"; do
      echo "- BLOCKER: ${item}"
    done
    for item in "${WARNINGS[@]}"; do
      echo "- WARNING: ${item}"
    done
  fi
  echo
  echo "## 12. Shortest Path To Gate 1"
  echo
  echo "After Gate 0 is PASS, proceed to Gate 1 data-loading smoke tests for ETTh1, Weather, and Electricity without starting formal training first."
} > reports/gate0_environment_report.md

echo "Gate 0 status: ${STATUS}"
echo "Log dir: ${LOG_DIR}"
echo "Status JSON: reports/gate0_status.json"
