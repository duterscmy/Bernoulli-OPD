#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/common.sh
source "$ROOT/scripts/lib/common.sh"

SETTING="${1:-}"
CHECKPOINT_ROOT="${2:-}"
CHECKPOINT_STEP="${3:-}"
MODE="${4:-validation}"
[[ -n "$SETTING" && -n "$CHECKPOINT_ROOT" && -n "$CHECKPOINT_STEP" ]] || \
  die "usage: $0 SETTING CHECKPOINT_ROOT CHECKPOINT_STEP [validation|final]"
[[ "$MODE" == validation || "$MODE" == final ]] || die "mode must be validation or final"

load_setting "$ROOT" "$SETTING"
resolve_release_paths "$ROOT"

export MODEL_ARGS_SCRIPT DISABLE_THINKING CHECKPOINT_ROOT CHECKPOINT_STEP
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-$FINAL_MAX_RESPONSE_LEN}"
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
export EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
export EVAL_TOP_K="${EVAL_TOP_K:-20}"
RUN_STEM="${RUN_ID:-$SETTING-step$CHECKPOINT_STEP}"

run_suite() {
  local suite="$1"
  local samples="$2"
  RUN_ID="$RUN_STEM" EVAL_SUITE="$suite" N_SAMPLES_PER_EVAL_PROMPT="$samples" \
    "$ROOT/examples/opsa/eval-qwen3-1.7B-best-checkpoint.sh"
}

if [[ "$MODE" == validation ]]; then
  run_suite aime24 8
else
  run_suite base32 "$FINAL_COMP_SAMPLES"
  run_suite math500 4
fi
