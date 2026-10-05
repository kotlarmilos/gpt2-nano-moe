---
library_name: pytorch
tags:
  - mixture-of-experts
  - causal-language-model
  - routing
---

# CapSel-2 routing artifact

This card accompanies the article "Reducing dropped tokens in a small MoE model"
and the measured artifact in `artifacts/publication-routing-study.json`.

## Result

CapSel-2 uses one shared per-expert capacity. It admits primary routes first,
tries to rescue rejected primaries through their second choice, then uses
remaining capacity for uncertain tokens that already have one accepted route.

Across three seeds on an NVIDIA RTX A5000, mean zero-expert token drop is 8.09%
for CapSel-2 and 13.84% for fixed top-2 at the same capacity factor. CapSel-2
requests 15.4% fewer slots and rescues 58.27% of rejected primaries.

## Intended use

Use this repository to inspect or reproduce the measured routing experiment.
It does not provide a selected pretrained checkpoint.

## Training data

The study uses the verified FineWeb-Edu token shard recorded in
`artifacts/manifest.json`. Each condition trains for 500 steps and uses the same
initialization and batches within each seed.

## Limitations

The held-out cross-entropy difference is 0.0089 in favor of CapSel-2 on the
three-seed mean, but it is smaller than the seed spread and changes sign for one
seed. The study does not establish a quality winner. Requested slots are not
latency measurements. The uncertainty threshold is calibrated once at
initialization and stays fixed.

The implementation is available at
`https://github.com/kotlarmilos/gpt2-nano-moe`.
