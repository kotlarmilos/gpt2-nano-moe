#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CORPUS_PATH="${CORPUS_PATH:-data/train.bin}"
OUTPUT="${OUTPUT:-artifacts/publication-routing-study.json}"
cd "$ROOT"

export PYTHONHASHSEED="${PYTHONHASHSEED:-17}"

"$PYTHON" -m gpt2_nano_moe.publication_study \
  --config configs/publication_study.toml \
  --corpus "$CORPUS_PATH" \
  --output "$OUTPUT"
