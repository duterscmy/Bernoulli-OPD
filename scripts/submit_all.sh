#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS=(qwen3_4b qwen3_8b_rl deepseek_7b skywork_7b)
METHODS=(vanilla sure poweropd gc bernoulli balanced_bernoulli)

for setting in "${SETTINGS[@]}"; do
  for method in "${METHODS[@]}"; do
    "$ROOT/scripts/submit.sh" "$setting" "$method"
  done
done
