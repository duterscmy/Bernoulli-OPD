#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTING="${1:?usage: $0 SETTING METHOD}"
METHOD_NAME="${2:?usage: $0 SETTING METHOD}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"

SBATCH_ARGS=(--time="$TIME_LIMIT" --export="ALL,RELEASE_ROOT=$ROOT,SETTING=$SETTING,METHOD_NAME=$METHOD_NAME")
[[ -z "${SLURM_PARTITION:-}" ]] || SBATCH_ARGS+=(--partition="$SLURM_PARTITION")
[[ -z "${SLURM_ACCOUNT:-}" ]] || SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")

sbatch "${SBATCH_ARGS[@]}" "$ROOT/scripts/slurm/train.sbatch"
