#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${DATA_ROOT:-$ROOT/data}}"

exec python3 "$ROOT/scripts/download_data.py" --data-root "$DATA_ROOT"
