#!/usr/bin/env bash
set -euo pipefail

# Four-GPU OPD comparison: 2 actor GPUs + 2 rollout/teacher GPUs.

if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [[ -f "$CONDA_SH" ]] || { echo "ERROR: missing $CONDA_SH" >&2; exit 1; }
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV_NAME:-opsa-slime}"
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPSA_ROOT="${OPSA_ROOT:-$SCRIPT_ROOT}"
export SLIME_DIR="${SLIME_DIR:-$SCRIPT_ROOT}"
export SGLANG_DIR="${SGLANG_DIR:-$OPSA_ROOT/sglang}"
export MEGATRON_DIR="${MEGATRON_DIR:-$OPSA_ROOT/Megatron-LM}"
export HF_CKPT="${HF_CKPT:-$OPSA_ROOT/models/Qwen3-1.7B}"
export MCORE_CKPT="${MCORE_CKPT:-$OPSA_ROOT/models/Qwen3-1.7B_torch_dist}"
export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-$OPSA_ROOT/models/Qwen3-4B-Instruct-2507}"
export MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-scripts/models/qwen3-1.7B.sh}"
export OPD_TEACHER_URL="${OPD_TEACHER_URL:-}"
export PROMPT_DATA="${PROMPT_DATA:-$OPSA_ROOT/data/dapo-math-17k/dapo-math-17k.jsonl}"
export EVAL_DATA="${EVAL_DATA:-$OPSA_ROOT/data/aime-2024/aime-2024.jsonl}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDNN_ROOT="${CUDNN_ROOT:-${CONDA_PREFIX:-/usr}/lib/python3.12/site-packages/nvidia/cudnn}"
export NCCL_HOME="${NCCL_HOME:-}"
export GCC_RUNTIME_LIB="${GCC_RUNTIME_LIB:-}"
export TORCH_LIB="${TORCH_LIB:-${CONDA_PREFIX:-/usr}/lib/python3.12/site-packages/torch/lib}"
export LD_LIBRARY_PATH="${GCC_RUNTIME_LIB:+$GCC_RUNTIME_LIB:}$CUDNN_ROOT/lib${NCCL_HOME:+:$NCCL_HOME/lib}:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$MEGATRON_DIR:$SLIME_DIR:$SGLANG_DIR/python:${PYTHONPATH:-}"

METHOD="${METHOD:-opd}"
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-vanilla}"
OPD_TOKEN_FILTER="${OPD_TOKEN_FILTER:-all}"
OPD_FILTER_NORMALIZATION="${OPD_FILTER_NORMALIZATION:-selected}"
OPD_HIGH_CONF_THRESHOLD="${OPD_HIGH_CONF_THRESHOLD:-0.5}"
OPD_HYBRID_THRESHOLD="${OPD_HYBRID_THRESHOLD:-0.5}"
OPD_BOTTOM_FRACTION="${OPD_BOTTOM_FRACTION:-0.2}"
OPD_TOP_FRACTION="${OPD_TOP_FRACTION:-0.2}"
OPD_GEOMETRY_ALPHA="${OPD_GEOMETRY_ALPHA:-0.5}"
OPD_POWER_ALPHA="${OPD_POWER_ALPHA:-0.5}"
OPD_SURE_ALPHA="${OPD_SURE_ALPHA:-1.0}"
OPD_BERNOULLI_CLIP="${OPD_BERNOULLI_CLIP:-0.0}"
OPD_BALANCE_ALPHA="${OPD_BALANCE_ALPHA:-0.5}"
OPD_BALANCE_BINS="${OPD_BALANCE_BINS:-10}"
OPD_BALANCE_MAX_WEIGHT="${OPD_BALANCE_MAX_WEIGHT:-5.0}"
OPD_DIAGNOSTICS_INTERVAL="${OPD_DIAGNOSTICS_INTERVAL:-0}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OVERRIDE_OPT_PARAM_SCHEDULER="${OVERRIDE_OPT_PARAM_SCHEDULER:-0}"
OPSA_MODE="${OPSA_MODE:-entropy}"
OPSA_TOKEN_FRACTION="${OPSA_TOKEN_FRACTION:-0.2}"
OPSA_FIXED_ADVANTAGE="${OPSA_FIXED_ADVANTAGE:--0.5}"

ACTOR_GPUS="${ACTOR_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
NUM_ROLLOUT="${NUM_ROLLOUT:-700}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES="${N_SAMPLES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12000}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-32168}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
RUN_EVAL="${RUN_EVAL:-1}"
RETAIN_BEST_AND_LAST="${RETAIN_BEST_AND_LAST:-1}"
RETAIN_STEPS="${RETAIN_STEPS:-}"
RETENTION_METRIC="${RETENTION_METRIC:-eval/aime}"
RETENTION_POLL_SECONDS="${RETENTION_POLL_SECONDS:-120}"
# Model-only checkpoints are the project default. Experiments that explicitly
# require optimizer-continuous resumption must opt in with NO_SAVE_OPTIM=0.
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
LOAD_DEBUG_ROLLOUT_DATA="${LOAD_DEBUG_ROLLOUT_DATA:-}"
SAVE_DEBUG_ROLLOUT_DATA="${SAVE_DEBUG_ROLLOUT_DATA:-}"
SAVE_DEBUG_TRAIN_DATA="${SAVE_DEBUG_TRAIN_DATA:-}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.98}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
SEED="${SEED:-1234}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
ROLLOUT_MEM_FRACTION="${ROLLOUT_MEM_FRACTION:-0.4}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-512}"
OPD_TEACHER_GPU_SLOT="${OPD_TEACHER_GPU_SLOT:-2}"
OPD_TEACHER_TP="${OPD_TEACHER_TP:-1}"
OPD_TEACHER_MEM_FRACTION="${OPD_TEACHER_MEM_FRACTION:-0.25}"
OPD_TEACHER_CHUNKED_PREFILL_SIZE="${OPD_TEACHER_CHUNKED_PREFILL_SIZE:-4096}"
OPD_TEACHER_DISABLE_CUDA_GRAPH="${OPD_TEACHER_DISABLE_CUDA_GRAPH:-0}"
OPD_TEACHER_HOST="${OPD_TEACHER_HOST:-127.0.0.1}"
OPD_TEACHER_PORT="${OPD_TEACHER_PORT:-$((30000 + ${SLURM_JOB_ID:-0} % 10000))}"
OPD_TEACHER_START_TIMEOUT="${OPD_TEACHER_START_TIMEOUT:-600}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"

case "$METHOD" in
  opd)
    DEFAULT_RUN_NAME="opd-${OPD_LOSS_TYPE}-${OPD_TOKEN_FILTER}"
    ;;
  opsa)
    DEFAULT_RUN_NAME="opsa-${OPSA_MODE}-bottom${OPSA_TOKEN_FRACTION}"
    ;;
  *)
    echo "ERROR: METHOD must be opd or opsa, got $METHOD" >&2
    exit 2
    ;;
esac

RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
SAVE_DIR="${SAVE_DIR:-$OPSA_ROOT/outputs/qwen3-1.7b-$RUN_NAME}"
LOG_FILE="${LOG_FILE:-$OPSA_ROOT/logs/qwen3-1.7b-$RUN_NAME-${SLURM_JOB_ID:-manual}.log}"

for path in "$SLIME_DIR" "$SGLANG_DIR" "$MEGATRON_DIR" "$HF_CKPT" "$MCORE_CKPT"; do
  [[ -e "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }
done
for file in "$PROMPT_DATA" "$EVAL_DATA"; do
  [[ -f "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done
if [[ "$METHOD" == "opd" && -z "$OPD_TEACHER_URL" && ! -f "$OPD_TEACHER_MODEL/config.json" ]]; then
  echo "ERROR: OPD requires a teacher checkpoint at $OPD_TEACHER_MODEL" >&2
  echo "       or an existing SGLang /generate endpoint via OPD_TEACHER_URL." >&2
  exit 1
fi
START_LOCAL_OPD_TEACHER=0
if [[ "$METHOD" == "opd" && -z "$OPD_TEACHER_URL" ]]; then
  START_LOCAL_OPD_TEACHER=1
  OPD_TEACHER_URL="http://$OPD_TEACHER_HOST:$OPD_TEACHER_PORT/generate"
fi
# A saved debug rollout already contains the teacher scores used by OPD.
# Slime treats this path as train-only replay, so no teacher server is needed.
if [[ -n "$LOAD_DEBUG_ROLLOUT_DATA" ]]; then
  START_LOCAL_OPD_TEACHER=0
fi
OPD_TEACHER_HEALTH_URL="${OPD_TEACHER_HEALTH_URL:-${OPD_TEACHER_URL%/generate}/health_generate}"

VISIBLE_GPUS="$(nvidia-smi -L | wc -l)"
if [[ -n "$LOAD_DEBUG_ROLLOUT_DATA" ]]; then
  # Slime turns fixed-rollout replay into debug_train_only mode.  No SGLang
  # engine or teacher is instantiated, so only the actor allocation is used.
  NEEDED_GPUS="$ACTOR_GPUS"
else
  NEEDED_GPUS="$((ACTOR_GPUS + ROLLOUT_GPUS))"
fi
(( VISIBLE_GPUS >= NEEDED_GPUS )) || {
  echo "ERROR: visible GPUs=$VISIBLE_GPUS, need $NEEDED_GPUS" >&2
  exit 1
}
(( ACTOR_GPUS % TP_SIZE == 0 )) || {
  echo "ERROR: ACTOR_GPUS must be divisible by TP_SIZE" >&2
  exit 1
}
(( ACTOR_GPUS == 2 && ROLLOUT_GPUS == 2 )) || {
  echo "ERROR: this runner requires ACTOR_GPUS=2 and ROLLOUT_GPUS=2" >&2
  exit 1
}
(( ROLLOUT_BATCH_SIZE * N_SAMPLES == GLOBAL_BATCH_SIZE )) || {
  echo "ERROR: ROLLOUT_BATCH_SIZE * N_SAMPLES must equal GLOBAL_BATCH_SIZE" >&2
  exit 1
}
[[ "$RUN_EVAL" == "0" || "$RUN_EVAL" == "1" ]] || {
  echo "ERROR: RUN_EVAL must be 0 or 1" >&2
  exit 1
}
[[ "$DISABLE_THINKING" == "0" || "$DISABLE_THINKING" == "1" ]] || {
  echo "ERROR: DISABLE_THINKING must be 0 or 1" >&2
  exit 1
}
[[ "$OVERRIDE_OPT_PARAM_SCHEDULER" == "0" || "$OVERRIDE_OPT_PARAM_SCHEDULER" == "1" ]] || {
  echo "ERROR: OVERRIDE_OPT_PARAM_SCHEDULER must be 0 or 1" >&2
  exit 1
}
[[ "$NO_SAVE_OPTIM" == "0" || "$NO_SAVE_OPTIM" == "1" ]] || {
  echo "ERROR: NO_SAVE_OPTIM must be 0 or 1" >&2
  exit 1
}
[[ "$NO_LOAD_OPTIM" == "0" || "$NO_LOAD_OPTIM" == "1" ]] || {
  echo "ERROR: NO_LOAD_OPTIM must be 0 or 1" >&2
  exit 1
}
[[ "$SAVE_CHECKPOINTS" == "0" || "$SAVE_CHECKPOINTS" == "1" ]] || {
  echo "ERROR: SAVE_CHECKPOINTS must be 0 or 1" >&2
  exit 1
}
[[ "$DEBUG_ROLLOUT_ONLY" == "0" || "$DEBUG_ROLLOUT_ONLY" == "1" ]] || {
  echo "ERROR: DEBUG_ROLLOUT_ONLY must be 0 or 1" >&2
  exit 1
}
if [[ "$DEBUG_ROLLOUT_ONLY" == "1" && -n "$LOAD_DEBUG_ROLLOUT_DATA" ]]; then
  echo "ERROR: DEBUG_ROLLOUT_ONLY and LOAD_DEBUG_ROLLOUT_DATA are mutually exclusive" >&2
  exit 1
fi
[[ -f "$SLIME_DIR/$MODEL_ARGS_SCRIPT" ]] || {
  echo "ERROR: missing model argument script $SLIME_DIR/$MODEL_ARGS_SCRIPT" >&2
  exit 1
}
if [[ "$START_LOCAL_OPD_TEACHER" == "1" ]]; then
  [[ "$OPD_TEACHER_GPU_SLOT" =~ ^[0-9]+$ ]] || {
    echo "ERROR: OPD_TEACHER_GPU_SLOT must be a non-negative integer" >&2
    exit 1
  }
  (( OPD_TEACHER_GPU_SLOT >= ACTOR_GPUS && OPD_TEACHER_GPU_SLOT < NEEDED_GPUS )) || {
    echo "ERROR: local teacher must share a rollout GPU slot [$ACTOR_GPUS, $((NEEDED_GPUS - 1))]" >&2
    exit 1
  }
  (( OPD_TEACHER_TP == 1 )) || {
    echo "ERROR: the four-GPU local teacher configuration requires OPD_TEACHER_TP=1" >&2
    exit 1
  }
  [[ "$OPD_TEACHER_DISABLE_CUDA_GRAPH" == "0" || "$OPD_TEACHER_DISABLE_CUDA_GRAPH" == "1" ]] || {
    echo "ERROR: OPD_TEACHER_DISABLE_CUDA_GRAPH must be 0 or 1" >&2
    exit 1
  }
fi

mkdir -p "$OPSA_ROOT/logs" "$SAVE_DIR"
OPD_TEACHER_LOG="${OPD_TEACHER_LOG:-$OPSA_ROOT/logs/qwen3-4b-teacher-${SLURM_JOB_ID:-manual}.log}"
OPD_TEACHER_PID=""
RETENTION_MONITOR_PID=""

stop_retention_monitor() {
  if [[ -n "$RETENTION_MONITOR_PID" ]]; then
    kill "$RETENTION_MONITOR_PID" >/dev/null 2>&1 || true
    wait "$RETENTION_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}

resolve_visible_gpu() {
  local logical_slot="$1"
  local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  local -a devices

  if [[ -z "$visible_devices" ]]; then
    printf '%s\n' "$logical_slot"
    return
  fi
  IFS=',' read -r -a devices <<< "$visible_devices"
  (( logical_slot < ${#devices[@]} )) || {
    echo "ERROR: GPU slot $logical_slot is absent from CUDA_VISIBLE_DEVICES=$visible_devices" >&2
    return 1
  }
  printf '%s\n' "${devices[$logical_slot]}"
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  stop_retention_monitor
  if [[ -n "$OPD_TEACHER_PID" ]] && kill -0 "$OPD_TEACHER_PID" 2>/dev/null; then
    kill "$OPD_TEACHER_PID" 2>/dev/null
    wait "$OPD_TEACHER_PID" 2>/dev/null
  fi
  ray stop --force >/dev/null 2>&1
  exit "$status"
}
trap cleanup EXIT INT TERM

start_opd_teacher() {
  [[ "$METHOD" == "opd" ]] || return
  # A replayed debug rollout already contains teacher scores. Starting another
  # teacher wastes a rollout GPU and can collide with train-only actor workers.
  [[ -z "$LOAD_DEBUG_ROLLOUT_DATA" ]] || return 0

  if [[ "$START_LOCAL_OPD_TEACHER" != "1" ]]; then
    curl -fsS --max-time 10 "$OPD_TEACHER_HEALTH_URL" >/dev/null || {
      echo "ERROR: external OPD teacher is not healthy: $OPD_TEACHER_HEALTH_URL" >&2
      return 1
    }
    return
  fi

  local teacher_device
  local -a teacher_args
  teacher_device="$(resolve_visible_gpu "$OPD_TEACHER_GPU_SLOT")"
  if curl -fsS --max-time 2 "$OPD_TEACHER_HEALTH_URL" >/dev/null 2>&1; then
    echo "ERROR: refusing to reuse an untracked teacher at $OPD_TEACHER_HEALTH_URL" >&2
    return 1
  fi

  echo "Starting OPD teacher $OPD_TEACHER_MODEL on rollout GPU slot $OPD_TEACHER_GPU_SLOT ($teacher_device)"
  teacher_args=(python3 -m sglang.launch_server \
    --model-path "$OPD_TEACHER_MODEL" \
    --host "$OPD_TEACHER_HOST" \
    --port "$OPD_TEACHER_PORT" \
    --tp "$OPD_TEACHER_TP" \
    --chunked-prefill-size "$OPD_TEACHER_CHUNKED_PREFILL_SIZE" \
    --mem-fraction-static "$OPD_TEACHER_MEM_FRACTION" \
    --attention-backend triton)
  if [[ "$OPD_TEACHER_DISABLE_CUDA_GRAPH" == "1" ]]; then
    teacher_args+=(--disable-cuda-graph)
  fi
  CUDA_VISIBLE_DEVICES="$teacher_device" "${teacher_args[@]}" >"$OPD_TEACHER_LOG" 2>&1 &
  OPD_TEACHER_PID=$!

  local waited=0
  until curl -fsS --max-time 2 "$OPD_TEACHER_HEALTH_URL" >/dev/null 2>&1; do
    if ! kill -0 "$OPD_TEACHER_PID" 2>/dev/null; then
      echo "ERROR: OPD teacher exited during startup; tail follows:" >&2
      tail -n 50 "$OPD_TEACHER_LOG" >&2 || true
      return 1
    fi
    if (( waited >= OPD_TEACHER_START_TIMEOUT )); then
      echo "ERROR: OPD teacher did not become healthy within ${OPD_TEACHER_START_TIMEOUT}s" >&2
      tail -n 50 "$OPD_TEACHER_LOG" >&2 || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "OPD teacher ready: $OPD_TEACHER_URL (log: $OPD_TEACHER_LOG)"
}

echo "========== Qwen3-1.7B OPD/OPSA study =========="
echo "host/job:              $(hostname) / ${SLURM_JOB_ID:-none}"
echo "method:                $METHOD"
echo "run:                   $RUN_NAME"
echo "OPD objective/filter:  $OPD_LOSS_TYPE / $OPD_TOKEN_FILTER"
echo "OPD filter norm:       $OPD_FILTER_NORMALIZATION"
echo "OPD hybrid threshold:  $OPD_HYBRID_THRESHOLD (strict >)"
echo "OPD power alpha:       $OPD_POWER_ALPHA"
echo "OPD SuRe alpha:        $OPD_SURE_ALPHA"
echo "OPD Bernoulli clip:    $OPD_BERNOULLI_CLIP (0=off)"
echo "OPD balance a/bins/max: $OPD_BALANCE_ALPHA / $OPD_BALANCE_BINS / $OPD_BALANCE_MAX_WEIGHT"
echo "OPD diagnostics every: $OPD_DIAGNOSTICS_INTERVAL rollout(s); 0=off"
echo "OPD coefficient:       $OPD_KL_COEF"
echo "OPSA mode/fraction:    $OPSA_MODE / $OPSA_TOKEN_FRACTION"
echo "actor/rollout GPUs:    $ACTOR_GPUS / $ROLLOUT_GPUS"
echo "rollouts:              $NUM_ROLLOUT"
echo "batch:                 $ROLLOUT_BATCH_SIZE x $N_SAMPLES = $GLOBAL_BATCH_SIZE"
echo "response/eval length:  $MAX_RESPONSE_LEN / $EVAL_MAX_RESPONSE_LEN"
echo "rollout sampling:        temp=$ROLLOUT_TEMPERATURE top_p=$ROLLOUT_TOP_P top_k=$ROLLOUT_TOP_K"
echo "eval samples/prompt:    $N_SAMPLES_PER_EVAL_PROMPT"
echo "eval sampling:           temp=$EVAL_TEMPERATURE top_p=$EVAL_TOP_P top_k=$EVAL_TOP_K"
echo "seeds (model/rollout):  $SEED / $ROLLOUT_SEED"
echo "evaluation enabled:      $RUN_EVAL"
echo "thinking:               $([[ "$DISABLE_THINKING" == "1" ]] && echo disabled || echo native)"
echo "checkpoint optimizer:   $([[ "$NO_SAVE_OPTIM" == "1" ]] && echo omitted || echo retained)"
echo "checkpoint saving:      $([[ "$SAVE_CHECKPOINTS" == "1" ]] && echo enabled || echo disabled)"
echo "protected checkpoints:  ${RETAIN_STEPS:-none}"
echo "debug rollout/replay:   $DEBUG_ROLLOUT_ONLY / ${LOAD_DEBUG_ROLLOUT_DATA:-none}"
echo "optimizer resume:       $([[ "$NO_LOAD_OPTIM" == "1" ]] && echo reinitialized || echo restored)"
echo "learning rate:          $LEARNING_RATE"
echo "LR warmup fraction:     $LR_WARMUP_FRACTION"
echo "Adam beta1/beta2:       $ADAM_BETA1 / $ADAM_BETA2"
echo "weight decay/clip grad: $WEIGHT_DECAY / $CLIP_GRAD"
echo "model args:             $MODEL_ARGS_SCRIPT"
if [[ "$METHOD" == "opd" ]]; then
  echo "OPD teacher:             ${OPD_TEACHER_URL:-$OPD_TEACHER_MODEL}"
  echo "teacher prefill/graphs:  $OPD_TEACHER_CHUNKED_PREFILL_SIZE / disable=$OPD_TEACHER_DISABLE_CUDA_GRAPH"
fi
echo "save dir:              $SAVE_DIR"
echo "log:                   $LOG_FILE"
echo "================================================"

python - <<'PY_PREFLIGHT'
import torch
import transformer_engine
import apex
import sglang
import megatron.core
import slime

print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count(), "SGLang:", sglang.__version__)
print("runtime preflight: OK")
PY_PREFLIGHT

start_opd_teacher

cd "$SLIME_DIR"
source "$MODEL_ARGS_SCRIPT"

LOAD_PATH="$MCORE_CKPT"
START_ROLLOUT_ARGS=(--start-rollout-id 0)
if [[ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]]; then
  LOAD_PATH="$SAVE_DIR"
  START_ROLLOUT_ARGS=()
fi

CKPT_ARGS=(
  --hf-checkpoint "$HF_CKPT"
  --ref-load "$MCORE_CKPT"
  --load "$LOAD_PATH"
  "${START_ROLLOUT_ARGS[@]}"
)
if [[ "$SAVE_CHECKPOINTS" == "1" ]]; then
  CKPT_ARGS+=(--save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL")
fi
if [[ "$NO_SAVE_OPTIM" == "1" ]]; then
  CKPT_ARGS+=(--no-save-optim)
fi
if [[ "$NO_LOAD_OPTIM" == "1" ]]; then
  CKPT_ARGS+=(--no-load-optim)
fi

DEBUG_ARGS=()
if [[ "$DEBUG_ROLLOUT_ONLY" == "1" ]]; then
  DEBUG_ARGS+=(--debug-rollout-only)
fi
if [[ -n "$LOAD_DEBUG_ROLLOUT_DATA" ]]; then
  DEBUG_ARGS+=(--load-debug-rollout-data "$LOAD_DEBUG_ROLLOUT_DATA")
fi
if [[ -n "$SAVE_DEBUG_ROLLOUT_DATA" ]]; then
  DEBUG_ARGS+=(--save-debug-rollout-data "$SAVE_DEBUG_ROLLOUT_DATA")
fi
if [[ -n "$SAVE_DEBUG_TRAIN_DATA" ]]; then
  DEBUG_ARGS+=(--save-debug-train-data "$SAVE_DEBUG_TRAIN_DATA")
fi

ROLLOUT_ARGS=(
  --prompt-data "$PROMPT_DATA"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES"
  --rollout-max-response-len "$MAX_RESPONSE_LEN"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --rollout-top-k "$ROLLOUT_TOP_K"
  --rollout-top-p "$ROLLOUT_TOP_P"
  --num-steps-per-rollout 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
)
if [[ "$DISABLE_THINKING" == "1" ]]; then
  ROLLOUT_ARGS+=(--disable-thinking)
fi

EVAL_ARGS=()
if [[ "$RUN_EVAL" == "1" ]]; then
  EVAL_ARGS=(
    --eval-interval "$EVAL_INTERVAL"
    --eval-prompt-data aime "$EVAL_DATA"
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
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi

if [[ "$METHOD" == "opd" ]]; then
  ALGORITHM_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-kl-coef "$OPD_KL_COEF"
    --opd-loss-type "$OPD_LOSS_TYPE"
    --opd-token-filter "$OPD_TOKEN_FILTER"
    --opd-filter-normalization "$OPD_FILTER_NORMALIZATION"
    --opd-high-conf-threshold "$OPD_HIGH_CONF_THRESHOLD"
    --opd-hybrid-threshold "$OPD_HYBRID_THRESHOLD"
    --opd-bottom-fraction "$OPD_BOTTOM_FRACTION"
    --opd-top-fraction "$OPD_TOP_FRACTION"
    --opd-geometry-alpha "$OPD_GEOMETRY_ALPHA"
    --opd-power-alpha "$OPD_POWER_ALPHA"
    --opd-sure-alpha "$OPD_SURE_ALPHA"
    --opd-bernoulli-clip "$OPD_BERNOULLI_CLIP"
    --opd-balance-alpha "$OPD_BALANCE_ALPHA"
    --opd-balance-bins "$OPD_BALANCE_BINS"
    --opd-balance-max-weight "$OPD_BALANCE_MAX_WEIGHT"
    --opd-diagnostics-interval "$OPD_DIAGNOSTICS_INTERVAL"
    --kl-coef 0
    --entropy-coef 0
  )
  RM_ARGS=(
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --rm-url "$OPD_TEACHER_URL"
  )
else
  ALGORITHM_ARGS=(
    --advantage-estimator opsa
    --opsa-mode "$OPSA_MODE"
    --opsa-token-fraction "$OPSA_TOKEN_FRACTION"
    --kl-coef 0
    --entropy-coef 0
  )
  if [[ "$OPSA_MODE" == "fixed" ]]; then
    ALGORITHM_ARGS+=(--opsa-fixed-advantage "$OPSA_FIXED_ADVANTAGE")
  fi
  RM_ARGS=(
    --custom-rm-path slime.rollout.opsa.reward_func
    --custom-reward-post-process-path slime.rollout.opsa.post_process_rewards
  )
fi

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LEARNING_RATE"
  --lr-warmup-fraction "$LR_WARMUP_FRACTION"
  --lr-decay-style constant
  --weight-decay "$WEIGHT_DECAY"
  --adam-beta1 "$ADAM_BETA1"
  --adam-beta2 "$ADAM_BETA2"
  --clip-grad "$CLIP_GRAD"
)
if [[ "$OVERRIDE_OPT_PARAM_SCHEDULER" == "1" ]]; then
  # Required when extending an optimizer-bearing checkpoint beyond the
  # original --num-rollout horizon. This preserves optimizer state and the
  # loaded step count while accepting the new total scheduler length.
  OPTIMIZER_ARGS+=(--override-opt-param-scheduler)
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
  --sglang-mem-fraction-static "$ROLLOUT_MEM_FRACTION"
  --sglang-server-concurrency "$SGLANG_SERVER_CONCURRENCY"
  --sglang-attention-backend triton
)

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
RETENTION_KEEP_ARGS=()
retention_steps_normalized="${RETAIN_STEPS//,/ }"
retention_steps_normalized="${retention_steps_normalized//:/ }"
for step in $retention_steps_normalized; do
  [[ "$step" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid RETAIN_STEPS entry: $step" >&2; exit 1; }
  RETENTION_KEEP_ARGS+=(--keep-step "$step")
done
if [[ "$RETAIN_BEST_AND_LAST" == "1" ]]; then
  (
    while true; do
      sleep "$RETENTION_POLL_SECONDS"
      python3 scripts/retain_best_checkpoint.py \
        --save-dir "$SAVE_DIR" \
        --log-file "$LOG_FILE" \
        --metric "$RETENTION_METRIC" \
        "${RETENTION_KEEP_ARGS[@]}" \
        --apply \
        >>"$OPSA_ROOT/logs/checkpoint-retention-${RUN_NAME}-${SLURM_JOB_ID:-manual}.log" 2>&1 || true
    done
  ) &
  RETENTION_MONITOR_PID=$!
fi

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
  "${DEBUG_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ALGORITHM_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" \
  "${RM_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"
set +x

stop_retention_monitor
RETENTION_MONITOR_PID=""

if [[ "$RETAIN_BEST_AND_LAST" == "1" ]]; then
  python3 scripts/retain_best_checkpoint.py \
    --save-dir "$SAVE_DIR" \
    --log-file "$LOG_FILE" \
    --metric "$RETENTION_METRIC" \
    "${RETENTION_KEEP_ARGS[@]}" \
    --apply
fi

echo "EXPERIMENT FINISHED: $RUN_NAME"
echo "Log: $LOG_FILE"
