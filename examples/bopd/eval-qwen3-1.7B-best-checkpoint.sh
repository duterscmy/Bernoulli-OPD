#!/usr/bin/env bash
set -euo pipefail

# Eval-only runner for a retained Qwen3-1.7B Megatron checkpoint.
# Supported allocations on one GH200 node:
#   2 actor GPUs + 2 SGLang rollout GPUs (standard training-comparable eval)
#   1 actor GPU  + 1 SGLang rollout GPU  (lightweight diagnostic eval)

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [[ -f "$CONDA_SH" ]] || { echo "ERROR: missing $CONDA_SH" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV_NAME:-bopd-slime}"
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BOPD_ROOT="${BOPD_ROOT:-$SCRIPT_ROOT}"
export SLIME_DIR="${SLIME_DIR:-$SCRIPT_ROOT}"
export SGLANG_DIR="${SGLANG_DIR:-$BOPD_ROOT/sglang}"
export MEGATRON_DIR="${MEGATRON_DIR:-$BOPD_ROOT/Megatron-LM}"
export HF_CKPT="${HF_CKPT:-$BOPD_ROOT/models/Qwen3-1.7B}"
export MCORE_CKPT="${MCORE_CKPT:-$BOPD_ROOT/models/Qwen3-1.7B_torch_dist}"
export MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-scripts/models/qwen3-1.7B.sh}"
export AIME24_DATA="${AIME24_DATA:-$BOPD_ROOT/data/aime-2024/aime-2024.jsonl}"
export AIME25_DATA="${AIME25_DATA:-$BOPD_ROOT/data/aime-2025/aime-2025.jsonl}"
export HMMT25_DATA="${HMMT25_DATA:-$BOPD_ROOT/data/hmmt-feb-2025/hmmt-feb-2025.jsonl}"
export MATH500_DATA="${MATH500_DATA:-$BOPD_ROOT/data/math-500/math-500.jsonl}"
export AMC23_DATA="${AMC23_DATA:-$BOPD_ROOT/data/amc-2023/amc-2023.jsonl}"
export MBPP_PLUS_DATA="${MBPP_PLUS_DATA:-$BOPD_ROOT/data/mbpp-plus/mbpp-plus-eval.jsonl}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_ROOT="${CUDNN_ROOT:-${CONDA_PREFIX:-/usr}/lib/python3.12/site-packages/nvidia/cudnn}"
export NCCL_HOME="${NCCL_HOME:-}"
export GCC_RUNTIME_LIB="${GCC_RUNTIME_LIB:-}"
export TORCH_LIB="${TORCH_LIB:-${CONDA_PREFIX:-/usr}/lib/python3.12/site-packages/torch/lib}"
export LD_LIBRARY_PATH="${GCC_RUNTIME_LIB:+$GCC_RUNTIME_LIB:}$CUDNN_ROOT/lib${NCCL_HOME:+:$NCCL_HOME/lib}:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"

: "${RUN_ID:?RUN_ID is required}"

ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-32}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32168}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"
ROLLOUT_MEM_FRACTION="${ROLLOUT_MEM_FRACTION:-0.65}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
INCLUDE_AIME24="${INCLUDE_AIME24:-0}"
EVAL_SUITE="${EVAL_SUITE:-ood}"
BASE_MODEL="${BASE_MODEL:-0}"
SEED="${SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-0}"
LOG_FILE="${LOG_FILE:-$BOPD_ROOT/logs/eval-${RUN_ID}-${EVAL_SUITE}-k${N_SAMPLES_PER_EVAL_PROMPT}-rseed${ROLLOUT_SEED}-${SLURM_JOB_ID:-manual}.log}"
DEBUG_DUMP="${DEBUG_DUMP:-}"
DEBUG_EVAL_CHECKPOINT_INTERVAL="${DEBUG_EVAL_CHECKPOINT_INTERVAL:-0}"
SGLANG_DETERMINISTIC_INFERENCE="${SGLANG_DETERMINISTIC_INFERENCE:-0}"

for path in "$SLIME_DIR" "$SGLANG_DIR" "$MEGATRON_DIR" "$HF_CKPT" "$MCORE_CKPT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
[[ "$INCLUDE_AIME24" == "0" || "$INCLUDE_AIME24" == "1" ]] || {
  echo "ERROR: INCLUDE_AIME24 must be 0 or 1" >&2
  exit 1
}
[[ "$BASE_MODEL" == "0" || "$BASE_MODEL" == "1" ]] || {
  echo "ERROR: BASE_MODEL must be 0 or 1" >&2
  exit 1
}
[[ "$DISABLE_THINKING" == "0" || "$DISABLE_THINKING" == "1" ]] || {
  echo "ERROR: DISABLE_THINKING must be 0 or 1" >&2
  exit 1
}
[[ "$SGLANG_DETERMINISTIC_INFERENCE" == "0" || "$SGLANG_DETERMINISTIC_INFERENCE" == "1" ]] || {
  echo "ERROR: SGLANG_DETERMINISTIC_INFERENCE must be 0 or 1" >&2
  exit 1
}

case "$EVAL_SUITE" in
  ood)
    EVAL_PROMPT_DATA=(aime25 "$AIME25_DATA" hmmt25 "$HMMT25_DATA")
    if [[ "$INCLUDE_AIME24" == "1" ]]; then
      EVAL_PROMPT_DATA=(aime24 "$AIME24_DATA" "${EVAL_PROMPT_DATA[@]}")
    fi
    ;;
  base32)
    EVAL_PROMPT_DATA=(
      aime24 "$AIME24_DATA"
      aime25 "$AIME25_DATA"
      hmmt25 "$HMMT25_DATA"
      amc23 "$AMC23_DATA"
    )
    ;;
  aime_pair)
    EVAL_PROMPT_DATA=(
      aime24 "$AIME24_DATA"
      aime25 "$AIME25_DATA"
    )
    ;;
  hmmt_amc_pair)
    EVAL_PROMPT_DATA=(
      hmmt25 "$HMMT25_DATA"
      amc23 "$AMC23_DATA"
    )
    ;;
  amc23)
    EVAL_PROMPT_DATA=(amc23 "$AMC23_DATA")
    ;;
  remaining32)
    EVAL_PROMPT_DATA=(
      aime25 "$AIME25_DATA"
      hmmt25 "$HMMT25_DATA"
      amc23 "$AMC23_DATA"
    )
    ;;
  math500)
    EVAL_PROMPT_DATA=(math500 "$MATH500_DATA")
    ;;
  aime24)
    EVAL_PROMPT_DATA=(aime24 "$AIME24_DATA")
    ;;
  aime25)
    EVAL_PROMPT_DATA=(aime25 "$AIME25_DATA")
    ;;
  hmmt25)
    EVAL_PROMPT_DATA=(hmmt25 "$HMMT25_DATA")
    ;;
  mbpp_plus)
    EVAL_PROMPT_DATA=(mbpp_plus "$MBPP_PLUS_DATA")
    ;;
  *)
    echo "ERROR: unsupported EVAL_SUITE=$EVAL_SUITE (expected ood, base32, aime_pair, hmmt_amc_pair, amc23, remaining32, math500, aime24, aime25, hmmt25, or mbpp_plus)" >&2
    exit 1
    ;;
esac

for ((index = 1; index < ${#EVAL_PROMPT_DATA[@]}; index += 2)); do
  file="${EVAL_PROMPT_DATA[$index]}"
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

if [[ "$BASE_MODEL" == "1" ]]; then
  CHECKPOINT_ROOT="$MCORE_CKPT"
  CHECKPOINT_LABEL="base-release"
else
  : "${CHECKPOINT_ROOT:?CHECKPOINT_ROOT is required when BASE_MODEL=0}"
  [[ -d "$CHECKPOINT_ROOT" ]] || { echo "ERROR: missing $CHECKPOINT_ROOT" >&2; exit 1; }
  printf -v checkpoint_dir '%s/iter_%07d' "$CHECKPOINT_ROOT" "$CHECKPOINT_STEP"
  [[ -d "$checkpoint_dir" ]] || { echo "ERROR: missing checkpoint $checkpoint_dir" >&2; exit 1; }
  CHECKPOINT_LABEL="step${CHECKPOINT_STEP}"
fi

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
NEEDED_GPUS="$((ACTOR_GPUS + ROLLOUT_GPUS))"
(( VISIBLE_GPUS >= NEEDED_GPUS )) || {
  echo "ERROR: visible GPUs=$VISIBLE_GPUS, need $NEEDED_GPUS" >&2
  exit 1
}
(( (ACTOR_GPUS == 2 && ROLLOUT_GPUS == 2) || (ACTOR_GPUS == 1 && ROLLOUT_GPUS == 1) )) || {
  echo "ERROR: this runner supports ACTOR_GPUS/ROLLOUT_GPUS=2/2 or 1/1" >&2
  exit 1
}

mkdir -p "$BOPD_ROOT/logs"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  ray stop --force >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT INT TERM

echo "========== BOPD-compatible benchmark evaluation =========="
echo "host/job:              $(hostname) / ${SLURM_JOB_ID:-none}"
echo "run/checkpoint:        $RUN_ID / $CHECKPOINT_LABEL"
echo "checkpoint root:       $CHECKPOINT_ROOT"
echo "eval suite:            $EVAL_SUITE"
echo "datasets:              ${EVAL_PROMPT_DATA[*]}"
echo "samples per prompt:    $N_SAMPLES_PER_EVAL_PROMPT"
echo "temperature/top-p/k:   $EVAL_TEMPERATURE / $EVAL_TOP_P / $EVAL_TOP_K"
echo "max response length:   $EVAL_MAX_RESPONSE_LEN"
if [[ "$DISABLE_THINKING" == "1" ]]; then
  echo "thinking:              disabled"
else
  echo "thinking:              enabled/native"
fi
echo "actor/rollout GPUs:    $ACTOR_GPUS / $ROLLOUT_GPUS"
echo "log:                   $LOG_FILE"
echo "debug rollout dump:    ${DEBUG_DUMP:-disabled}"
echo "debug eval checkpoint: every $DEBUG_EVAL_CHECKPOINT_INTERVAL completed responses"
echo "========================================================="

cd "$SLIME_DIR"
source "$MODEL_ARGS_SCRIPT"

CKPT_ARGS=(
  --hf-checkpoint "$HF_CKPT"
  --ref-load "$MCORE_CKPT"
  --load "$CHECKPOINT_ROOT"
  --no-load-optim
  --no-load-rng
  --start-rollout-id 0
)
if [[ "$BASE_MODEL" == "0" ]]; then
  CKPT_ARGS+=(--ckpt-step "$CHECKPOINT_STEP")
fi

ROLLOUT_ARGS=(
  --num-rollout 0
  --rollout-batch-size 64
  --n-samples-per-prompt 1
  --rollout-max-response-len 12000
  --rollout-temperature 1.0
  --rollout-top-k -1
  --rollout-top-p 1.0
  --global-batch-size 64
)
if [[ "$EVAL_SUITE" != "mbpp_plus" ]]; then
  ROLLOUT_ARGS+=(--apply-chat-template)
else
  # The MBPP+ data contains EvalPlus's exact pre-rendered chat prompt,
  # including its assistant-side ```python prefix.
  ROLLOUT_ARGS+=(--rollout-stop $'\n```\n')
fi
if [[ "$DISABLE_THINKING" == "1" ]]; then
  ROLLOUT_ARGS+=(--disable-thinking)
fi

EVAL_ARGS=(
  --eval-interval 1
  --eval-prompt-data "${EVAL_PROMPT_DATA[@]}"
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT"
  --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
  --eval-top-p "$EVAL_TOP_P"
  --eval-temperature "$EVAL_TEMPERATURE"
  --eval-top-k "$EVAL_TOP_K"
  --eval-rm-type math
  --log-passrate
)
if [[ -n "$DEBUG_DUMP" ]]; then
  mkdir -p "$(dirname "$DEBUG_DUMP")"
  EVAL_ARGS+=(--save-debug-rollout-data "$DEBUG_DUMP")
fi
if (( DEBUG_EVAL_CHECKPOINT_INTERVAL > 0 )); then
  [[ -n "$DEBUG_DUMP" ]] || {
    echo "ERROR: DEBUG_EVAL_CHECKPOINT_INTERVAL requires DEBUG_DUMP" >&2
    exit 1
  }
  EVAL_ARGS+=(--debug-eval-checkpoint-interval "$DEBUG_EVAL_CHECKPOINT_INTERVAL")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "$TP_SIZE"
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384
)

ALGORITHM_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0
  --entropy-coef 0
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$ROLLOUT_MEM_FRACTION"
  --sglang-attention-backend triton
)
if [[ "$SGLANG_DETERMINISTIC_INFERENCE" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi

MISC_ARGS=(
  --seed "$SEED"
  --rollout-seed "$ROLLOUT_SEED"
  --log-interval 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

ray stop --force >/dev/null 2>&1 || true
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
DASHBOARD_PORT="${DASHBOARD_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"

ray start \
  --head \
  --node-ip-address="$MASTER_ADDR" \
  --num-gpus="$VISIBLE_GPUS" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --dashboard-port="$DASHBOARD_PORT"

RUNTIME_ENV_JSON="$(printf '{"env_vars":{"PYTHONPATH":"%s","CUDA_HOME":"%s","CUDA_PATH":"%s","LD_LIBRARY_PATH":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","TOKENIZERS_PARALLELISM":"false"}}' \
  "$PYTHONPATH" "$CUDA_HOME" "$CUDA_PATH" "$LD_LIBRARY_PATH")"

set -x
ray job submit \
  --address="http://127.0.0.1:$DASHBOARD_PORT" \
  --runtime-env-json="$RUNTIME_ENV_JSON" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  --num-gpus-per-node "$VISIBLE_GPUS" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${ALGORITHM_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
set +x

echo "BENCHMARK EVALUATION FINISHED: $RUN_ID $CHECKPOINT_LABEL suite=$EVAL_SUITE"
echo "Log: $LOG_FILE"
