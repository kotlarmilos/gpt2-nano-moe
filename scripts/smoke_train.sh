#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m gpt2_nano_moe.train \
  --config configs/smoke.toml \
  --output artifacts/smoke-train.json \
  --checkpoint checkpoints/smoke.pt

