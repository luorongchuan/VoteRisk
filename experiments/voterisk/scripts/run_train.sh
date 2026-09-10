#!/bin/bash
# run_train.sh — single-mode DAPO launcher for Qwen3-4B (2 × A100 80GB).
#
# Modes (positional arg):
#   edas       — DAPO + EDAS branch_advantage_adjustment (paper baseline)
#   voterisk   — DAPO + VoteRisk-16 (advantage shaping via P(N_j >= N_c))
#
# Both modes share every hyperparameter; the only difference is which
# `algorithm.*_enabled` flag is on at runtime. See
# experiments/voterisk/ORIGINAL_EDAS_CONFIG.md for the parameter inventory.
#
# Usage:
#   bash experiments/voterisk/scripts/run_train.sh <mode> [--smoke]
#
# Environment overrides:
#   ROLLOUT_GPU_MEM_UTIL   vLLM memory fraction (default 0.7 for 4B)
#   TRAIN_BATCH_SIZE       effective prompts per train step (default 256)
#   GEN_BATCH_SIZE         prompts sampled per gen round (default 512)
#   ACTOR_PPO_MICRO_BSZ_PER_GPU  actor micro batch per GPU (default 1)
#   ROLLOUT_TP             vLLM tensor parallel size (default 1)
#   NUM_GPUS               total GPUs (default 2)
#   TOTAL_EPOCHS           training epochs (default 2)

set -euo pipefail

# Ray / vLLM only exist inside the verl conda env. Auto-activate when called
# from a shell that doesn't already have it (e.g. when invoked directly).
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "verl" ]]; then
  # shellcheck disable=SC1091
  source /home/luorongchuan/miniconda3/etc/profile.d/conda.sh
  conda activate verl
fi

if [[ $# -lt 1 ]]; then
  echo "[ERROR] missing mode. usage: $0 <edas|voterisk> [--smoke]" >&2
  exit 1
fi
MODE="$1"
shift

SMOKE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    *) echo "[ERROR] unknown arg: $1" >&2; exit 1 ;;
  esac
done

case "${MODE}" in
  edas)      METHOD="DAPO + EDAS"        ; BRANCH_ADV=True; VOTE_RISK=False ;;
  voterisk)  METHOD="DAPO + VoteRisk-16" ; BRANCH_ADV=False; VOTE_RISK=True  ;;
  *) echo "[ERROR] unknown mode: ${MODE} (expected 'edas' or 'voterisk')" >&2; exit 1 ;;
esac

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # EDAS/ candidate
# Walk up until we hit pyproject.toml (handles nested checkouts robustly).
while [[ ! -f "${VR_ROOT}/pyproject.toml" ]]; do
  VR_ROOT="$(cd "${VR_ROOT}/.." && pwd)"
done
cd "${VR_ROOT}"

# Datasets and base model — paths overridable via env (no hardcoded absolute
# paths in source so this repo is git-safe). Defaults match the local layout
# at /home/luorongchuan/workspace_135/{datasets,models}; change as needed.
TRAIN_FILE="${TRAIN_FILE:-/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.train.parquet}"
VAL_FILE="${VAL_FILE:-/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.val500.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/luorongchuan/workspace_135/models/Qwen3-4B-Base}"

# Output paths (kept separate so runs don't clobber each other).
EXP_ROOT="/home/luorongchuan/workspace_135/VoteRisk/experiments/voterisk"
OUTPUT_BASE="${EXP_ROOT}/checkpoints"
LOG_BASE="${EXP_ROOT}/logs"
mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}"

TASK_ID="$(date +%Y%m%d_%H%M%S)"
TASK_NAME="${MODE}_qwen3_4b_${TASK_ID}"
SAVE_DIR="${OUTPUT_BASE}/${TASK_NAME}"
mkdir -p "${SAVE_DIR}"

# ── DAPO + EDAS / VoteRisk hyperparameters (identical across methods) ───────
# All values are 1:1 with EDAS run_dapo_branch_adv.sh Qwen3-8B config; the
# only deviations are listed in HARDWARE_ADAPTATION.md.

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
DISABLE_KL=True
ONLINE_FILTERING=True
FILTER_METRIC=accuracy
MAX_TRY_MAKE_BATCH=20

# Save / test cadence.
SAVE_FREQ=10
TEST_FREQ=5
SAVE_LIMIT=20

# Length budget (per EDAS run_8b_non_thinking_04_30_20_wo_format_17k.sh).
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
ENABLE_DYNAMIC_MAX_TOKENS=True
PENALTY_MAX_LENGTH=50000
OVERLONG_BUFFER_LENGTH=0

# Rollout / vLLM.
ROLLOUT_N=10
# 4B on 2 GPUs in colocated mode: FSDP actor+ref already reserves ~50 GiB on
# each card; vLLM must fit in the remaining ~28 GiB. 0.4×79.25=31.7 GiB leaves
# ~3 GiB headroom. Override via env to bump if your environment has more free.
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.4}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768

# Optimizer.
ACTOR_LR=1e-6
ACTOR_WEIGHT_DECAY=1.0e-2
ACTOR_ULYSSES=1
ACTOR_GRAD_NORM=1.0

# Batch sizes — kept at the EDAS 8B defaults. Two A100 80GB easily fits 4B
# with these (see HARDWARE_ADAPTATION.md). Override via env if needed.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE=1024
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-64}"
ACTOR_PPO_MICRO_BSZ_PER_GPU="${ACTOR_PPO_MICRO_BSZ_PER_GPU:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 2))}"

# Branch-adv (EDAS) defaults — only applied when MODE=edas.
BRANCH_ADV_ALPHA=0.2
BRANCH_ADV_BETA=0.2
BRANCH_ADV_KAPPA=2.0

# VoteRisk defaults — only applied when MODE=voterisk.
VOTE_RISK_G=10
VOTE_RISK_K_TARGET=16
VOTE_RISK_ALPHA_SMOOTH=0.5
VOTE_RISK_LAMBDA=0.4

# Reward function — EDAS' own math_no_format.compute_score_batch (with LLM
# judge disabled because we have no judge cluster locally; rule + sympy
# equivalence is enough for DAPO-Math-17k and matches the val pipeline).
REWARD_FN_PATH="${VR_ROOT}/examples/branch_adv/reward_function/math_no_format.py"
REWARD_FN_NAME="compute_score_batch"
REWARD_MANAGER="batch"
ENABLE_LLM_JUDGE=False

# Validation override (matches EDAS).
VAL_TEMPERATURE=0.7
VAL_TOP_P=0.8
VAL_TOP_K=20
VAL_PRESENCE_PENALTY=1.5
VAL_N=1

NUM_GPUS="${NUM_GPUS:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"

# Smoke test override: short training, skip in-loop validation entirely
# (TEST_FREQ=999 means val never fires — keeps the smoke fast, just 1 step).
if [[ "${SMOKE}" == "1" ]]; then
  TOTAL_EPOCHS=2
  TEST_FREQ=999
  SAVE_FREQ=1
  SAVE_LIMIT=2
  echo "[smoke] reduced TOTAL_EPOCHS=${TOTAL_EPOCHS} + skip in-loop val (TEST_FREQ=999)"
fi

# ── disable_kl → kl knobs ───────────────────────────────────────────────────
if [[ "${DISABLE_KL,,}" == "true" ]]; then
  USE_KL_IN_REWARD=False
  USE_KL_LOSS=False
  KL_LOSS_COEF=0.0
  KL_COEF=0.0
else
  USE_KL_IN_REWARD=False
  USE_KL_LOSS=True
  KL_LOSS_COEF=1.0e-2
  KL_COEF=1.0e-2
fi

# ── Ray cluster (run on the local node) ─────────────────────────────────────
target_master_addr="${MASTER_ADDR:-127.0.0.1}"
target_master_port="${MASTER_PORT:-29500}"
target_nnodes="${WORLD_SIZE:-1}"
target_node_rank="${RANK:-0}"
ray_port="${RAY_PORT:-6379}"

# ── Output / logging paths ──────────────────────────────────────────────────
TENSORBOARD_DIR="${SAVE_DIR}/tensorboard_log"
mkdir -p "${TENSORBOARD_DIR}"
LENGTH_LOG_PATH="${SAVE_DIR}/lengthlog"

LOG_FILE="${LOG_BASE}/${TASK_NAME}.log"
BRANCH_LOG_PATH="${SAVE_DIR}/branch_adv.log"

# ── Hydra args ──────────────────────────────────────────────────────────────
PY_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
  algorithm.kl_ctrl.kl_coef=${KL_COEF}
  algorithm.kl_penalty=low_var_kl

  # DAPO dynamic sampling.
  algorithm.filter_groups.enable=${ONLINE_FILTERING}
  algorithm.filter_groups.metric=${FILTER_METRIC}
  algorithm.filter_groups.max_num_gen_batches=${MAX_TRY_MAKE_BATCH}

  # Mutual-exclusion advantage shaping flags.
  algorithm.branch_adv_enabled=${BRANCH_ADV}
  algorithm.branch_adv_alpha=${BRANCH_ADV_ALPHA}
  algorithm.branch_adv_beta=${BRANCH_ADV_BETA}
  algorithm.branch_adv_kappa=${BRANCH_ADV_KAPPA}
  algorithm.branch_adv_log_path="branch_adv.log"
  algorithm.branch_adv_warmup_start=0
  algorithm.branch_adv_warmup_steps=0

  algorithm.vote_risk_enabled=${VOTE_RISK}
  algorithm.vote_risk_G=${VOTE_RISK_G}
  algorithm.vote_risk_K_target=${VOTE_RISK_K_TARGET}
  algorithm.vote_risk_alpha_smooth=${VOTE_RISK_ALPHA_SMOOTH}
  algorithm.vote_risk_lambda=${VOTE_RISK_LAMBDA}
  algorithm.vote_risk_warmup_start=0
  algorithm.vote_risk_warmup_steps=0

  # Data.
  data.train_files=${TRAIN_FILE}
  data.val_files=${VAL_FILE}
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.train_batch_size=${TRAIN_BATCH_SIZE}
  data.gen_batch_size=${GEN_BATCH_SIZE}
  data.val_batch_size=${VAL_BATCH_SIZE}
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=True
  +data.apply_chat_template_kwargs.enable_thinking=False

  # Model.
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # Actor.
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
  actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY}
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
  actor_rollout_ref.actor.grad_clip=${ACTOR_GRAD_NORM}
  actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU}
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

  # Rollout.
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.enable_dynamic_max_tokens=${ENABLE_DYNAMIC_MAX_TOKENS}
  actor_rollout_ref.rollout.length_log_path=${LENGTH_LOG_PATH}
  actor_rollout_ref.rollout.length_log_interval=1
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
  actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
  actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
  actor_rollout_ref.rollout.val_kwargs.presence_penalty=${VAL_PRESENCE_PENALTY}
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
  actor_rollout_ref.rollout.val_kwargs.do_sample=True

  # Ref.
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  # Reward.
  reward.custom_reward_function.path=${REWARD_FN_PATH}
  reward.custom_reward_function.name=${REWARD_FN_NAME}
  reward.reward_manager.name=${REWARD_MANAGER}
  +reward.custom_reward_function.reward_kwargs.penalty_max_length=${PENALTY_MAX_LENGTH}
  +reward.custom_reward_function.reward_kwargs.overlong_buffer_length=${OVERLONG_BUFFER_LENGTH}
  +reward.custom_reward_function.reward_kwargs.enable_llm_judge=${ENABLE_LLM_JUDGE}
  reward.reward_kwargs.overlong_buffer_cfg.enable=False

  # Trainer.
  trainer.experiment_name=${TASK_NAME}
  trainer.project_name="voterisk_qwen3_4b"
  trainer.logger='[console]'
  trainer.n_gpus_per_node=${NUM_GPUS}
  trainer.nnodes=${target_nnodes}
  trainer.save_freq=${SAVE_FREQ}
  trainer.test_freq=${TEST_FREQ}
  trainer.max_actor_ckpt_to_keep=${SAVE_LIMIT}
  trainer.total_epochs=${TOTAL_EPOCHS}
  trainer.val_before_train=False
  trainer.default_local_dir=${SAVE_DIR}
)

echo "=========================================="
echo " Voterisk Qwen3-4B training"
echo "   method        : ${METHOD}"
echo "   mode          : ${MODE}"
echo "   train_batch   : ${TRAIN_BATCH_SIZE}"
echo "   gen_batch     : ${GEN_BATCH_SIZE}"
echo "   rollout_n     : ${ROLLOUT_N}"
echo "   rollout_tp    : ${ROLLOUT_TP}"
echo "   GPUs          : ${NUM_GPUS}"
echo "   epochs        : ${TOTAL_EPOCHS}"
echo "   save_dir      : ${SAVE_DIR}"
echo "   log_file      : ${LOG_FILE}"
echo "=========================================="

set -x
ray stop --force || true
if [[ "$target_node_rank" -eq 0 ]]; then
  ray start --head \
    --port="$ray_port" \
    --num-gpus="$NUM_GPUS" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats
else
  ray start --address="$target_master_addr:$ray_port" \
    --num-gpus="$NUM_GPUS" \
    --disable-usage-stats \
    --block
  exit 0
fi

cd "${VR_ROOT}"
python3 -m recipe.dapo.main_dapo "${PY_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"

ray stop --force || true
echo "[done] method=${METHOD} save_dir=${SAVE_DIR}"