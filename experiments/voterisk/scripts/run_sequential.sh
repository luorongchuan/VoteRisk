#!/bin/bash
# run_sequential.sh — runs EDAS and VoteRisk training+eval sequentially on the
# same 2 GPUs (avoid having one model per card, which serialises 4 long tasks
# instead of letting the resources do one at a time).
#
# Pipeline:
#   1. Train EDAS (2 GPUs)
#   2. Eval  EDAS (1-2 GPUs)
#   3. Train VoteRisk (2 GPUs)
#   4. Eval  VoteRisk (1-2 GPUs)
#   5. Write FINAL_COMPARISON.md + plots
#
# Each step is logged to experiments/voterisk/logs/seq_<step>.log.
# Override anything via environment variables, e.g.:
#   SKIP_TRAIN=1 SKIP_EVAL=0 CUDA_VISIBLE_DEVICES=0,1 bash run_sequential.sh
#
# Set SEQ_METHODS=base to only run the base-model eval (sanity-check before
# training); or SEQ_METHODS=edas to skip VoteRisk.

set -euo pipefail

# Ray / vLLM only exist inside the verl conda env — activate it up front so
# subprocesses (ray, python -m recipe.dapo.main_dapo) inherit PATH.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "verl" ]]; then
  # shellcheck disable=SC1091
  source /home/luorongchuan/miniconda3/etc/profile.d/conda.sh
  conda activate verl
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EDAS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
while [[ ! -f "${EDAS_ROOT}/pyproject.toml" ]]; do
  EDAS_ROOT="$(cd "${EDAS_ROOT}/.." && pwd)"
done
mkdir -p "${EXP_ROOT}/logs" "${EXP_ROOT}/checkpoints" "${EXP_ROOT}/results"
cd "${EDAS_ROOT}"

SEQ_METHODS="${SEQ_METHODS:-edas voterisk}"
SEQ_DATASETS="${SEQ_DATASETS:-MATH-500,AMC23,AIME24,AIME25}"
SEQ_EVAL_N="${SEQ_EVAL_N:-64}"
SEQ_SMOKE="${SEQ_SMOKE:-0}"
SEQ_BASE_EVAL="${SEQ_BASE_EVAL:-0}"        # also eval the base model first
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
NUM_GPUS="${NUM_GPUS:-2}"

EVAL_GPUS="${EVAL_GPUS:-1}"                # vLLM tp_size for eval
CUDA_VISIBLE_DEVICES_VAL="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}"

LOG_DIR="${EXP_ROOT}/logs"
TRAIN_DONE_FLAG() { echo "${EXP_ROOT}/checkpoints/${1}/_DONE"; }
ENSURE_DONE() {
  local flag
  flag="$(TRAIN_DONE_FLAG "$1")"
  if [[ -f "${flag}" ]]; then
    echo "[seq] ${1} already done (flag exists at ${flag}); skipping training"
    return 1
  fi
  return 0
}
MARK_DONE() {
  local flag
  flag="$(TRAIN_DONE_FLAG "$1")"
  mkdir -p "$(dirname "${flag}")"
  touch "${flag}"
  echo "[seq] marked ${1} done -> ${flag}"
}

# ──────────────────────────────────────────────────────────────────────────
# Optional: evaluate the base model first as a reference (sanity check).
# ──────────────────────────────────────────────────────────────────────────
if [[ "${SEQ_BASE_EVAL}" == "1" ]]; then
  BASE_PATH="${BASE_PATH:-/home/luorongchuan/workspace_135/models/Qwen3-4B-Base}"
  echo "[seq] >>> step 0: base eval ${BASE_PATH}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
  python3 "${EXP_ROOT}/scripts/eval_model.py" \
      --model_path "${BASE_PATH}" \
      --method base \
      --datasets "${SEQ_DATASETS}" \
      --n "${SEQ_EVAL_N}" \
      --tp_size "${EVAL_GPUS}" \
      --gpu_mem 0.85 \
      --out_root "${EXP_ROOT}" \
      2>&1 | tee "${LOG_DIR}/seq_base_eval.log"
fi

# ──────────────────────────────────────────────────────────────────────────
# Train + eval per method.
# ──────────────────────────────────────────────────────────────────────────
declare -A CKPT_PATH

for METHOD in ${SEQ_METHODS}; do
  echo "=========================================="
  echo "[seq] method = ${METHOD}"
  echo "=========================================="

  # ── Train ─────────────────────────────────────────────────────────────
  if [[ "${SKIP_TRAIN}" == "1" ]]; then
    echo "[seq] SKIP_TRAIN=1 — skipping training"
  elif ! ENSURE_DONE "${METHOD}"; then
    :
  else
    echo "[seq] >>> step train: ${METHOD}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
    bash "${EXP_ROOT}/scripts/run_train.sh" "${METHOD}" $([[ "${SEQ_SMOKE}" == "1" ]] && echo "--smoke") \
        2>&1 | tee "${LOG_DIR}/seq_train_${METHOD}.log"

    # Locate the produced checkpoint directory (timestamped name).
    LATEST_CKPT="$(ls -td "${EXP_ROOT}/checkpoints/${METHOD}_qwen3_4b_"* 2>/dev/null | head -1 || true)"
    if [[ -z "${LATEST_CKPT}" ]]; then
      echo "[seq][ERROR] no checkpoint directory found under ${EXP_ROOT}/checkpoints/${METHOD}_*"
      exit 1
    fi
    CKPT_PATH["${METHOD}"]="${LATEST_CKPT}"
    MARK_DONE "${METHOD}"
  fi

  # Re-discover the checkpoint directory (in case training was skipped).
  LATEST_CKPT="${CKPT_PATH[${METHOD}]:-}"
  if [[ -z "${LATEST_CKPT}" ]]; then
    LATEST_CKPT="$(ls -td "${EXP_ROOT}/checkpoints/${METHOD}_qwen3_4b_"* 2>/dev/null | head -1 || true)"
    CKPT_PATH["${METHOD}"]="${LATEST_CKPT}"
  fi
  if [[ -z "${LATEST_CKPT}" || ! -d "${LATEST_CKPT}" ]]; then
    echo "[seq][ERROR] cannot find ${METHOD} checkpoint"
    exit 1
  fi

  # ── Eval ───────────────────────────────────────────────────────────────
  if [[ "${SKIP_EVAL}" == "1" ]]; then
    echo "[seq] SKIP_EVAL=1 — skipping eval"
    continue
  fi

  # Pick the final-step checkpoint under LATEST_CKPT.
  FINAL_CKPT="${LATEST_CKPT}"
  LAST_STEP_DIR="$(ls -td "${LATEST_CKPT}"/global_step_* 2>/dev/null | tail -1 || true)"
  if [[ -n "${LAST_STEP_DIR}" && -d "${LAST_STEP_DIR}/actor" ]]; then
    FINAL_CKPT="${LAST_STEP_DIR}/actor"
  fi

  echo "[seq] >>> step eval: ${METHOD} (ckpt=${FINAL_CKPT})"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
  python3 "${EXP_ROOT}/scripts/eval_model.py" \
      --model_path "${FINAL_CKPT}" \
      --method "${METHOD}" \
      --datasets "${SEQ_DATASETS}" \
      --n "${SEQ_EVAL_N}" \
      --tp_size "${EVAL_GPUS}" \
      --gpu_mem 0.85 \
      --out_root "${EXP_ROOT}" \
      2>&1 | tee "${LOG_DIR}/seq_eval_${METHOD}.log"
done

# ──────────────────────────────────────────────────────────────────────────
# Build the final report + plots.
# ──────────────────────────────────────────────────────────────────────────
echo "[seq] >>> building FINAL_COMPARISON.md"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VAL}" \
python3 "${EXP_ROOT}/scripts/build_final_report.py" \
    --exp_root "${EXP_ROOT}" \
    --methods ${SEQ_METHODS} \
    --datasets "${SEQ_DATASETS}" \
    2>&1 | tee "${LOG_DIR}/seq_final_report.log"

echo "[seq] all done. See ${EXP_ROOT}/results/FINAL_COMPARISON.md"