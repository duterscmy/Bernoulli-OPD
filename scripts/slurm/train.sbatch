#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --job-name=bernoulli-opd
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${SETTING:?SETTING is required}"
: "${METHOD_NAME:?METHOD_NAME is required}"
: "${RELEASE_ROOT:?RELEASE_ROOT is required}"

"$RELEASE_ROOT/scripts/train.sh" "$SETTING" "$METHOD_NAME"
