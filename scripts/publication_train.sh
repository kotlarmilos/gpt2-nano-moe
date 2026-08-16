#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
ARCHITECTURE="${ARCHITECTURE:-moe}"
SEED="${SEED:-17}"
CORPUS_PATH="${CORPUS_PATH:-data/train.bin}"
OUTPUT="${OUTPUT:-artifacts/publication-${ARCHITECTURE}-seed-${SEED}.json}"
CHECKPOINT="${CHECKPOINT:-checkpoints/publication-${ARCHITECTURE}-seed-${SEED}.pt}"
cd "$ROOT"

export PYTHONHASHSEED="${PYTHONHASHSEED:-17}"

"$PYTHON" -m gpt2_nano_moe.train \
  --config configs/publication.toml \
  --architecture "$ARCHITECTURE" \
  --seed "$SEED" \
  --corpus "$CORPUS_PATH" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT"
