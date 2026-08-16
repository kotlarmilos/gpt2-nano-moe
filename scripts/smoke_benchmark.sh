#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m gpt2_nano_moe.benchmark \
  --config configs/smoke.toml \
  --iterations 10 \
  --output artifacts/smoke-benchmark.json

