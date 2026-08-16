---
license: mit
library_name: pytorch
pipeline_tag: text-generation
tags:
  - mixture-of-experts
  - educational
  - causal-language-model
---

# GPT2 Nano MoE

This is the model card for the article project "Adding sparse Mixture of
Experts to a small transformer."

## Model description

The repository implements a small GPT-style causal decoder with interchangeable
dense and sparse MLP blocks. The sparse block uses deterministic top-1 routing,
per-expert capacity, a load-balancing auxiliary loss, and explicit routing
metrics.

MoE changes the MLP path only. Attention is identical in the dense and sparse
variants.

## Intended use

The code is intended for education, routing experiments, tests, and small local
benchmarks. It is not a pretrained production language model.

## Training data

No publication model has been trained. Committed smoke results use seeded
synthetic integer tokens and do not measure language quality.

## Metrics

Measured smoke outputs and complete environment metadata are stored in
`artifacts/smoke-train.json` and `artifacts/smoke-benchmark.json`.
`artifacts/manifest.json` distinguishes measured, simulated, and pending work.
Training records language-model cross entropy separately from the auxiliary
routing term and total optimization loss. Publication validation compares the
language-model cross entropy on a deterministic held-out corpus split.

A development-scale extension artifact is stored in
`artifacts/routing-topk-fixed-capacity.json`. It reuses the smoke MoE checkpoint
and one seeded synthetic fixed batch to compare top-1 and top-2 routing at fixed
per-expert capacity. Top-1 drops 6 of 256 routed slots. Top-2 drops 192 of 512
routed slots and balances accepted counts exactly at the capacity cap.

## Limitations

The expert dispatch prioritizes readability over optimized kernels. The
repository does not include a publication checkpoint. The Space includes only a
clearly labeled four-step synthetic checkpoint for routing inspection.
