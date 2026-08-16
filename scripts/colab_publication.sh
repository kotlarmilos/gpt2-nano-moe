#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
EXPERIMENT="${EXPERIMENT:-}"
cd "$ROOT"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e .

case "$EXPERIMENT" in
  train)
    : "${ARCHITECTURE:?Set ARCHITECTURE to dense or moe}"
    export ARCHITECTURE
    "$PYTHON" scripts/prepare_publication_data.py
    ./scripts/publication_train.sh
    ;;
  benchmark)
    : "${DENSE_CHECKPOINT:?Set DENSE_CHECKPOINT to the downloaded dense checkpoint}"
    : "${MOE_CHECKPOINT:?Set MOE_CHECKPOINT to the downloaded MoE checkpoint}"
    export DENSE_CHECKPOINT MOE_CHECKPOINT
    "$PYTHON" scripts/prepare_publication_data.py
    ./scripts/publication_benchmark.sh
    ;;
  *)
    echo "Set EXPERIMENT to train or benchmark." >&2
    exit 2
    ;;
esac
