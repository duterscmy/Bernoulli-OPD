#!/usr/bin/env bash

die() {
  echo "ERROR: $*" >&2
  exit 2
}

load_setting() {
  local root="$1"
  local setting="$2"
  local config="$root/configs/settings/$setting.env"
  [[ -f "$config" ]] || die "unknown setting '$setting'"
  # shellcheck disable=SC1090
  source "$config"
}

load_method() {
  local root="$1"
  local method="$2"
  local config="$root/configs/methods/$method.env"
  [[ -f "$config" ]] || die "unknown method '$method'"
  # shellcheck disable=SC1090
  source "$config"
}

resolve_release_paths() {
  local root="$1"
  export OPSA_ROOT="${OPSA_ROOT:-$root}"
  export SLIME_DIR="${SLIME_DIR:-$root}"
  export MODEL_ROOT="${MODEL_ROOT:-$OPSA_ROOT/models}"
  export DATA_ROOT="${DATA_ROOT:-$OPSA_ROOT/data}"
  export HF_CKPT="${HF_CKPT:-$MODEL_ROOT/$HF_CKPT_REL}"
  export MCORE_CKPT="${MCORE_CKPT:-$MODEL_ROOT/$MCORE_CKPT_REL}"
  export OPD_TEACHER_MODEL="${OPD_TEACHER_MODEL:-$MODEL_ROOT/$TEACHER_CKPT_REL}"
  export PROMPT_DATA="${PROMPT_DATA:-$DATA_ROOT/dapo-math-17k/dapo-math-17k.jsonl}"
  export AIME24_DATA="${AIME24_DATA:-$DATA_ROOT/aime-2024/aime-2024.jsonl}"
  export AIME25_DATA="${AIME25_DATA:-$DATA_ROOT/aime-2025/aime-2025.jsonl}"
  export HMMT25_DATA="${HMMT25_DATA:-$DATA_ROOT/hmmt-feb-2025/hmmt-feb-2025.jsonl}"
  export AMC23_DATA="${AMC23_DATA:-$DATA_ROOT/amc-2023/amc-2023.jsonl}"
  export MATH500_DATA="${MATH500_DATA:-$DATA_ROOT/math-500/math-500.jsonl}"
}
