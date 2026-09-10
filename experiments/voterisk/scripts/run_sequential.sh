#!/bin/bash
# Sequential fair comparison:
#   train EDAS -> merge/eval EDAS -> train VoteRisk -> merge/eval VoteRisk

set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "verl" ]]; then
  # shellcheck disable=SC1091
  source /home/luorongchuan/miniconda3/etc/profile.d/conda.sh
  conda activate verl
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
while [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; do
  REPO_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
done
cd "${REPO_ROOT}"

SEQ_METHODS="${SEQ_METHODS:-edas voterisk}"
SEQ_DATASETS="${SEQ_DATASETS:-MATH-500,AMC23,AIME24,AIME25}"
SEQ_EVAL_N="${SEQ_EVAL_N:-64}"
SEQ_MAJ_RESAMPLES="${SEQ_MAJ_RESAMPLES:-2000}"
SEQ_SMOKE="${SEQ_SMOKE:-0}"
SEQ_BASE_EVAL="${SEQ_BASE_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NUM_GPUS="${NUM_GPUS:-2}"
EVAL_GPUS="${EVAL_GPUS:-1}"
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"

mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/checkpoints" "${EXP_ROOT}/results"
LOG_DIR="${EXP_ROOT}/logs"

latest_run_dir() {
  local method="$1"
  ls -td "${EXP_ROOT}/checkpoints/${method}_qwen3_4b_base_"* 2>/dev/null | head -1 || true
}

latest_step_dir() {
  local run_dir="$1"
  find "${run_dir}" -maxdepth 1 -type d -name 'global_step_*' -print 2>/dev/null \
    | sort -V | tail -1
}

prepare_hf_checkpoint() {
  local actor_dir="$1"
  local merged_dir="${actor_dir%/actor}/merged_hf"

  # Already a directly loadable HF checkpoint.
  if [[ -f "${actor_dir}/config.json" ]] && compgen -G "${actor_dir}/*.safetensors" >/dev/null; then
    echo "${actor_dir}"
    return 0
  fi
  if [[ -f "${merged_dir}/config.json" ]] && compgen -G "${merged_dir}/*.safetensors" >/dev/null; then
    echo "${merged_dir}"
    return 0
  fi

  echo "[seq] merging FSDP actor -> ${merged_dir}" >&2
  rm -rf "${merged_dir}"
  python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${merged_dir}" >&2

  if [[ ! -f "${merged_dir}/config.json" ]]; then
    echo "[seq][ERROR] merged checkpoint missing config.json: ${merged_dir}" >&2
    return 1
  fi
  echo "${merged_dir}"
}

if [[ "${SEQ_BASE_EVAL}" == "1" ]]; then
  BASE_PATH="${BASE_PATH:-/home/luorongchuan/workspace_135/models/Qwen3-4B-Base}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
  python3 "${EXP_ROOT}/scripts/eval_model.py" \
    --model_path "${BASE_PATH}" --method base \
    --datasets "${SEQ_DATASETS}" --n "${SEQ_EVAL_N}" \
    --maj_resamples "${SEQ_MAJ_RESAMPLES}" \
    --tp_size "${EVAL_GPUS}" --gpu_mem 0.85 --out_root "${EXP_ROOT}" \
    2>&1 | tee "${LOG_DIR}/seq_base_eval.log"
fi

for METHOD in ${SEQ_METHODS}; do
  echo "=========================================="
  echo "[seq] method=${METHOD}"
  echo "=========================================="

  if [[ "${SKIP_TRAIN}" != "1" ]]; then
    if [[ "${SEQ_SMOKE}" == "1" ]]; then
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
      bash "${EXP_ROOT}/scripts/run_train.sh" "${METHOD}" --smoke \
        2>&1 | tee "${LOG_DIR}/seq_train_${METHOD}.log"
    else
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
      bash "${EXP_ROOT}/scripts/run_train.sh" "${METHOD}" \
        2>&1 | tee "${LOG_DIR}/seq_train_${METHOD}.log"
    fi
  fi

  if [[ "${SKIP_EVAL}" == "1" ]]; then
    continue
  fi

  RUN_DIR="$(latest_run_dir "${METHOD}")"
  if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "[seq][ERROR] no ${METHOD} run directory found" >&2
    exit 1
  fi
  STEP_DIR="$(latest_step_dir "${RUN_DIR}")"
  if [[ -z "${STEP_DIR}" || ! -d "${STEP_DIR}/actor" ]]; then
    echo "[seq][ERROR] no actor checkpoint under ${RUN_DIR}" >&2
    exit 1
  fi

  # The previous script used `ls -td ... | tail -1`, which selected the
  # oldest checkpoint.  sort -V + tail selects the largest global_step.
  echo "[seq] latest checkpoint=${STEP_DIR}"
  HF_CKPT="$(prepare_hf_checkpoint "${STEP_DIR}/actor")"
  echo "[seq] eval model=${HF_CKPT}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
  python3 "${EXP_ROOT}/scripts/eval_model.py" \
    --model_path "${HF_CKPT}" --method "${METHOD}" \
    --datasets "${SEQ_DATASETS}" --n "${SEQ_EVAL_N}" \
    --maj_resamples "${SEQ_MAJ_RESAMPLES}" \
    --tp_size "${EVAL_GPUS}" --gpu_mem 0.85 --out_root "${EXP_ROOT}" \
    2>&1 | tee "${LOG_DIR}/seq_eval_${METHOD}.log"
done

if [[ "${SKIP_EVAL}" != "1" ]]; then
  python3 "${EXP_ROOT}/scripts/build_final_report.py" \
    --exp_root "${EXP_ROOT}" --methods ${SEQ_METHODS} \
    --datasets "${SEQ_DATASETS}" \
    2>&1 | tee "${LOG_DIR}/seq_final_report.log"
fi

echo "[seq] done. results: ${EXP_ROOT}/results/"
