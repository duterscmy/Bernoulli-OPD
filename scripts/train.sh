#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"

SETTING="${1:-}"
METHOD_NAME="${2:-}"
[[ -n "$SETTING" && -n "$METHOD_NAME" ]] || die "usage: $0 SETTING METHOD"
load_setting "$ROOT" "$SETTING"
load_method "$ROOT" "$METHOD_NAME"
resolve_release_paths "$ROOT"

export METHOD=opd
export OPD_LOSS_TYPE OPD_POWER_ALPHA
export MODEL_ARGS_SCRIPT DISABLE_THINKING OPD_TEACHER_MEM_FRACTION
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export N_SAMPLES="${N_SAMPLES:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-12000}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export ROLLOUT_TOP_K="${ROLLOUT_TOP_K:--1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.0}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export EVAL_DATA="${EVAL_DATA:-$AIME24_DATA}"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-8}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-$FINAL_MAX_RESPONSE_LEN}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-1}"
export RUN_NAME="${RUN_NAME:-$SETTING-$METHOD_ID-s${SEED:-1234}-avg8}"
export SAVE_DIR="${SAVE_DIR:-$BOPD_ROOT/outputs/$RUN_NAME}"

for variable in OPD_SURE_ALPHA OPD_GEOMETRY_ALPHA OPD_BERNOULLI_CLIP OPD_BALANCE_ALPHA OPD_BALANCE_BINS OPD_BALANCE_MAX_WEIGHT; do
  if [[ -n "${!variable:-}" ]]; then
    export "$variable"
  fi
done

exec "$ROOT/examples/bopd/run-qwen3-1.7B-bopd-study.sh"
