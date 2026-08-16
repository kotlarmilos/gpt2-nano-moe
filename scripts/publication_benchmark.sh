#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
OUTPUT="${OUTPUT:-artifacts/publication-benchmark.json}"
ITERATIONS="${ITERATIONS:-50}"
DENSE_CHECKPOINT="${DENSE_CHECKPOINT:-}"
MOE_CHECKPOINT="${MOE_CHECKPOINT:-}"
CORPUS_PATH="${CORPUS_PATH:-data/train.bin}"
cd "$ROOT"

: "${DENSE_CHECKPOINT:?Set DENSE_CHECKPOINT to a trained dense checkpoint}"
: "${MOE_CHECKPOINT:?Set MOE_CHECKPOINT to a trained MoE checkpoint}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-17}"

"$PYTHON" -m gpt2_nano_moe.benchmark \
  --config configs/publication.toml \
  --iterations "$ITERATIONS" \
  --dense-checkpoint "$DENSE_CHECKPOINT" \
  --moe-checkpoint "$MOE_CHECKPOINT" \
  --corpus "$CORPUS_PATH" \
  --output "$OUTPUT"
