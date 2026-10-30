#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTING="${1:?usage: $0 SETTING}"
METHODS=(vanilla sure poweropd gc bernoulli balanced_bernoulli)

for method in "${METHODS[@]}"; do
  "$ROOT/scripts/train.sh" "$SETTING" "$method"
done
