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

This is the model card for the article project "Rescuing dropped tokens in
capacity-limited MoE routing."

## Model description

The repository implements a small GPT-style causal decoder with interchangeable
dense and sparse MLP blocks. Sparse routing modes include token-order top-1,
batch-prioritized top-1, shared-capacity fixed top-2, and capacity-aware selective
top-2.

MoE changes the MLP path only. Attention is identical in the dense and sparse
variants.

## Intended use

The code is intended for education, routing experiments, tests, and small local
benchmarks. It is not a pretrained production language model.

## Training data

The publication study trains matched models for 500 steps on the verified
FineWeb-Edu token shard. It does not select or distribute a trained checkpoint.
Committed smoke checkpoints use synthetic integer tokens and remain limited to
interface inspection.

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

The measured RTX A5000 artifact is stored in
`artifacts/publication-routing-study.json`. Across three seeds, CapSel-2 records
8.09 percent mean zero-expert token drop versus 13.84 percent for fixed top-2 at
the same capacity factor. It requests 15.4 percent fewer slots and rescues 58.27
percent of rejected primaries. The held-out cross-entropy gap is smaller than the
seed spread and does not establish a quality winner.

## Limitations

The expert dispatch prioritizes readability over optimized kernels. The study
compares 500-step early training rather than converged models. Its uncertainty
threshold is calibrated once at initialization and remains fixed. The repository
does not include a publication checkpoint. The Space includes only a clearly
labeled four-step synthetic checkpoint for routing inspection.
