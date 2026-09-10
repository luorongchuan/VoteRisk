#!/bin/bash
# run_train.sh — fair DAPO comparison for Qwen3-4B-Base on 2 × A100 80GB.
#
# Modes:
#   edas       — DAPO + official EDAS branch advantage shaping
#   voterisk   — DAPO + reviewed VoteRisk-16 advantage shaping
#
# All non-shaping hyperparameters are identical across the two modes.
# Hardware-only defaults below are tuned for 2 × A100 80GB. They increase
# rollout concurrency and keep FSDP states on GPU to trade spare memory for
# throughput without changing the optimization objective or sampling recipe.

set -euo pipefail

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "verl" ]]; then
  # shellcheck disable=SC1091
  source /home/luorongchuan/miniconda3/etc/profile.d/conda.sh
  conda activate verl
fi

if [[ $# -lt 1 ]]; then
  echo "[ERROR] usage: $0 <edas|voterisk> [--smoke]" >&2
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
  edas)      METHOD="DAPO + EDAS"        ; BRANCH_ADV=True;  VOTE_RISK=False ;;
  voterisk)  METHOD="DAPO + VoteRisk-16" ; BRANCH_ADV=False; VOTE_RISK=True  ;;
  *) echo "[ERROR] mode must be edas or voterisk" >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
while [[ ! -f "${VR_ROOT}/pyproject.toml" ]]; do
  VR_ROOT="$(cd "${VR_ROOT}/.." && pwd)"
done
cd "${VR_ROOT}"

# A100-SXM4 is compute capability 8.0. Restricting extension compilation to
# sm80 avoids compiling kernels for irrelevant architectures on this server.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

# verl's colocated vLLM rollout uses vLLM's CuMem memory pool. Current vLLM
# explicitly rejects PyTorch expandable_segments with that pool, so remove the
# option if it was inherited from the shell. Do not enable it for this launcher.
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* ]]; then
  unset PYTORCH_CUDA_ALLOC_CONF
fi

# These defaults match the user's server layout shown for workspace_135.
TRAIN_FILE="${TRAIN_FILE:-/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.train.parquet}"
VAL_FILE="${VAL_FILE:-/home/luorongchuan/workspace_135/datasets/dapo-math-17k.formatted.val500.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/luorongchuan/workspace_135/models/Qwen3-4B-Base}"

for p in "${TRAIN_FILE}" "${VAL_FILE}" "${MODEL_PATH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] required path does not exist: ${p}" >&2
    exit 2
  fi
done

EXP_ROOT="${EXP_ROOT:-/home/luorongchuan/workspace_135/VoteRisk/experiments/voterisk}"
OUTPUT_BASE="${EXP_ROOT}/checkpoints"
LOG_BASE="${EXP_ROOT}/logs"
mkdir -p "${OUTPUT_BASE}" "${LOG_BASE}"

TASK_ID="$(date +%Y%m%d_%H%M%S)"
TASK_NAME="${MODE}_qwen3_4b_base_${TASK_ID}"
SAVE_DIR="${OUTPUT_BASE}/${TASK_NAME}"
mkdir -p "${SAVE_DIR}"

# DAPO core hyperparameters: copied from EDAS official DAPO launcher.
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
DISABLE_KL=True
ONLINE_FILTERING=True
FILTER_METRIC=accuracy
MAX_TRY_MAKE_BATCH=20
SAVE_FREQ=10
TEST_FREQ=5
SAVE_LIMIT=20
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
ENABLE_DYNAMIC_MAX_TOKENS=True
PENALTY_MAX_LENGTH=50000
OVERLONG_BUFFER_LENGTH=0
ROLLOUT_N=10

# 2 × A100 80GB throughput profile.
# 0.65 is deliberately below the EDAS reference launcher's 0.8 because this
# two-GPU setup also keeps FSDP parameters and optimizer state on GPU. In
# practice GPU memory varies by rollout/update phase; the goal is a high
# 60–70+ GB peak rather than forcing a constant allocation.
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.65}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}"

ACTOR_LR=1e-6
ACTOR_WEIGHT_DECAY=1.0e-2
ACTOR_ULYSSES=1
ACTOR_GRAD_NORM=1.0
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
VAL_BATCH_SIZE=1024
ACTOR_PPO_MINI_BATCH_SIZE="${ACTOR_PPO_MINI_BATCH_SIZE:-64}"
ACTOR_PPO_MICRO_BSZ_PER_GPU="${ACTOR_PPO_MICRO_BSZ_PER_GPU:-1}"
# With dynamic batching, this controls how many tokens can be packed into one
# actor forward/backward microbatch. 64k makes better use of 80GB cards while
# preserving exactly the same examples and loss.
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
# CPU offload is a memory-saving mode and costs PCIe/host transfer time. The
# 4B model fits comfortably on 2 × 80GB with FSDP, so formal runs keep these
# states on GPU by default. All values remain environment-overridable.
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * 2))}"
NUM_GPUS="${NUM_GPUS:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

# Official EDAS DAPO recipe values. Do NOT use the module's generic 0.2/0.2
# defaults for the paper baseline.
BRANCH_ADV_ALPHA=0.4
BRANCH_ADV_BETA=0.3
BRANCH_ADV_KAPPA=2.0

# VoteRisk-only settings.
VOTE_RISK_G=10
VOTE_RISK_K_TARGET=16
VOTE_RISK_ALPHA_SMOOTH=0.5
VOTE_RISK_LAMBDA=0.4

REWARD_FN_PATH="${VR_ROOT}/examples/branch_adv/reward_function/math_no_format.py"
REWARD_FN_NAME="compute_score_batch"
REWARD_MANAGER="batch"
# Local adaptation: the official EDAS launcher can call a private LLM-judge
# cluster. This server does not have that service, so both methods use exactly
# the same rule + SymPy path with the judge disabled.
ENABLE_LLM_JUDGE=False

VAL_TEMPERATURE=0.7
VAL_TOP_P=0.8
VAL_TOP_K=20
VAL_PRESENCE_PENALTY=1.5
VAL_N=1

# A real smoke test must limit optimiser steps. Keeping the same hardware
# profile in smoke mode is intentional: it validates the memory settings that
# will be used by the formal run.
if [[ "${SMOKE}" == "1" ]]; then
  TOTAL_TRAINING_STEPS="${SMOKE_STEPS:-2}"
  TEST_FREQ=999999
  SAVE_FREQ=1
  SAVE_LIMIT=2
  echo "[smoke] trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
fi

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

target_master_addr="${MASTER_ADDR:-127.0.0.1}"
target_nnodes="${WORLD_SIZE:-1}"
target_node_rank="${RANK:-0}"
ray_port="${RAY_PORT:-6379}"

LENGTH_LOG_PATH="${SAVE_DIR}/lengthlog"
LOG_FILE="${LOG_BASE}/${TASK_NAME}.log"

PY_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
  algorithm.kl_ctrl.kl_coef=${KL_COEF}
  algorithm.kl_penalty=low_var_kl
  algorithm.filter_groups.enable=${ONLINE_FILTERING}
  algorithm.filter_groups.metric=${FILTER_METRIC}
  algorithm.filter_groups.max_num_gen_batches=${MAX_TRY_MAKE_BATCH}

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

  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

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
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES}
  actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}

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

  actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}

  reward.custom_reward_function.path=${REWARD_FN_PATH}
  reward.custom_reward_function.name=${REWARD_FN_NAME}
  reward.reward_manager.name=${REWARD_MANAGER}
  +reward.custom_reward_function.reward_kwargs.penalty_max_length=${PENALTY_MAX_LENGTH}
  +reward.custom_reward_function.reward_kwargs.overlong_buffer_length=${OVERLONG_BUFFER_LENGTH}
  +reward.custom_reward_function.reward_kwargs.enable_llm_judge=${ENABLE_LLM_JUDGE}
  reward.reward_kwargs.overlong_buffer_cfg.enable=False

  trainer.experiment_name=${TASK_NAME}
  trainer.project_name="voterisk_qwen3_4b_base"
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

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  PY_ARGS+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

echo "=========================================="
echo " VoteRisk fair comparison"
echo " method        : ${METHOD}"
echo " model         : ${MODEL_PATH}"
echo " train         : ${TRAIN_FILE}"
echo " val           : ${VAL_FILE}"
echo " EDAS alpha/beta/kappa : ${BRANCH_ADV_ALPHA}/${BRANCH_ADV_BETA}/${BRANCH_ADV_KAPPA}"
echo " VoteRisk K/lambda     : ${VOTE_RISK_K_TARGET}/${VOTE_RISK_LAMBDA}"
echo " rollout_n     : ${ROLLOUT_N}"
echo " GPUs          : ${NUM_GPUS}"
echo " epochs        : ${TOTAL_EPOCHS}"
echo " total_steps   : ${TOTAL_TRAINING_STEPS:-derived-from-epochs}"
echo " rollout mem   : ${ROLLOUT_GPU_MEM_UTIL}"
echo " rollout tokens: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
echo " actor tokens  : ${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
echo " actor offload : param=${ACTOR_PARAM_OFFLOAD}, optimizer=${ACTOR_OPTIMIZER_OFFLOAD}"
echo " save_dir      : ${SAVE_DIR}"
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

python3 -m recipe.dapo.main_dapo "${PY_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
ray stop --force || true
echo "[done] method=${METHOD} save_dir=${SAVE_DIR}"
