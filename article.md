---
title: "Reducing dropped tokens in a small MoE model"
thumbnail: /blog/assets/nano-sparse-moe/thumbnail.png
authors:
- user: kotlarmilos
---

# Reducing dropped tokens in a small MoE model

## TL;DR

I test a routing rule for a mixture-of-experts model with limited capacity.
Before giving an already-served token a second expert, the rule tries to give
an expert to tokens whose first choice was full.

On a three-seed RTX A5000 experiment, this lowers the mean fraction of tokens
receiving no expert from 13.84% to 8.09% compared with fixed top-2 routing at
the same capacity factor. It also requests 15.4% fewer expert slots.

This is a routing result. The loss difference is small, one seed does not
improve token coverage, and 500 training steps do not establish which policy
produces a better language model.

## Why tokens can miss every expert

An MoE layer replaces one feed-forward network with several experts. A router
chooses which experts process each token. Attention is unchanged in this
experiment.

Each expert can accept only a fixed number of tokens from a batch. A router
can still send too many tokens to the same expert. When that happens, some
assignments are rejected.

There are two different outcomes to count. A token might receive one expert
instead of two, or it might receive no expert at all. The second case means
that it gets no expert update at that layer. The token itself is not removed
from the sequence; its residual stream continues.

Sparse routing and capacity limits are established parts of MoE systems [1, 2].
Batch-prioritized routing also appears in prior work [3]. My experiment builds
on those ideas and tests how to order primary, rescue, and optional requests
under one shared capacity limit.

## Give uncovered tokens priority

I call the policy `CapSel-2`, for capacity-aware selective top-2 routing. It uses
the router's two highest-probability experts and makes three passes.

1. Admit first-choice requests, ranked by router score within each expert.
2. Try the second-choice expert for tokens rejected in the first pass, ranked
   by second-choice score within each expert.
3. Use any remaining capacity to give a second expert to already-served tokens
   whose first two router probabilities are close, starting with the smallest
   probability margin.

The rescue pass is the important change. Without it, a high-scoring second
request from a token that already has an expert can compete with the only
remaining route for an uncovered token.

The fixed top-2 baseline uses the same score-prioritized first pass and the same
shared capacity. It then considers every second choice by score, without
separating rescue requests from optional ones. It does not receive a separate
capacity budget for second choices.

## Capacity and uncertainty

For `N` tokens, `E` experts, and capacity factor `phi`, each expert accepts at
most

$$
C = \left\lceil \frac{\phi N}{E} \right\rceil
$$

assignments. All three passes consume this same limit.

Fixed top-2 requests `2N` slots. The selective policy requests one primary slot
per token, one rescue slot per rejected primary, and one optional slot per
uncertain accepted primary.

To decide whether a token is uncertain, I use the difference between its two
largest router probabilities.

$$
m_i = p_{i,1} - p_{i,2}.
$$

A small margin means neither choice is much more likely than the other. The
threshold is `0.03658`, the median margin across 24,576 samples from eight
calibration batches at initialization. It stays fixed throughout the study and
is shared across seeds. This is a routing heuristic, not a calibrated estimate
of whether the selected expert is correct.

## The matched experiment

The experiment runs on one NVIDIA RTX A5000. The model has six layers, width
384, eight experts, and a 128-token context. Each condition trains for 500
optimizer steps with batch size four, processing 256,000 tokens per seed.

The five conditions share the token shard, initialization within each seed,
training batches, optimizer settings, and held-out batches. There are three
seeds and sixteen final validation batches per run. Deterministic CUDA
execution uses math SDPA.

The capacity factor is 1.25 except for one top-1 baseline at 2.0. That extra
baseline tests whether simply increasing capacity reduces drops.

The main measurements distinguish token coverage from assignment coverage.

| Metric | What it counts |
| --- | --- |
| Zero-expert token drop | Tokens with no accepted expert at an MoE layer |
| Requested-slot drop | Rejected expert requests, including optional second requests |
| Experts per token | Accepted assignments divided by routed tokens |
| Rescue rate | Rejected primaries accepted by their second-choice expert |
| Held-out CE | Next-token cross entropy, excluding the auxiliary routing loss |

Routing counts include token visits across MoE layers. They are not counts of
words removed from the generated output.

## Results

The table reports means and population standard deviations across three seeds.
`CF` is the capacity factor. `BPR` means batch-prioritized routing.

| Routing condition | Token drop | Slot drop | Experts per token | Held-out CE |
| --- | ---: | ---: | ---: | ---: |
| Top-1 token order, CF 1.25 | 27.37% +/- 1.26% | 27.37% +/- 1.26% | 0.726 +/- 0.013 | 4.357 +/- 0.034 |
| Top-1 BPR, CF 1.25 | 30.90% +/- 1.26% | 30.90% +/- 1.26% | 0.691 +/- 0.013 | 4.349 +/- 0.035 |
| Fixed top-2, CF 1.25 | 13.84% +/- 3.46% | 45.91% +/- 1.33% | 1.082 +/- 0.027 | 4.402 +/- 0.047 |
| Top-1 token order, CF 2.0 | 9.88% +/- 3.74% | 9.88% +/- 3.74% | 0.901 +/- 0.037 | 4.354 +/- 0.031 |
| CapSel-2, CF 1.25 | 8.09% +/- 0.87% | 38.67% +/- 0.75% | 1.037 +/- 0.017 | 4.393 +/- 0.040 |

CapSel-2 rescues 58.27% of rejected primaries on average, with a population
standard deviation of 4.62 percentage points. It requests about 83,150 slots
across final validation, compared with 98,304 for fixed top-2.

That is 15.4% fewer requested slots and 4.1% fewer accepted slots. Yet the mean
fraction of tokens receiving no expert falls by 5.74 percentage points, a 41.5%
relative reduction using the unrounded measurements. The policy gives more
tokens at least one expert rather than maximizing the number of second routes.

The result is not consistent across every seed. Two paired seeds improve token
coverage substantially. The third has token drop 0.12 percentage points worse
than fixed top-2. Three seeds do not establish a general advantage.

## The loss result is inconclusive

Mean held-out cross entropy improves by `0.0089` relative to fixed top-2.
One paired seed moves in the opposite direction, and the variation across seeds
is larger than that mean difference.

More coverage is not automatically better prediction. The expert chosen for a
rescued token may not help, and changing routing also changes training. These
500-step runs show early behavior rather than convergence. I do not claim that
CapSel-2 produces better text.

The implementation uses PyTorch indexing and Python-level expert iteration.
Requested slots are not latency measurements. This study does not establish a
decoding or training speedup.

## What remains unresolved

The uncertainty threshold is measured once before training. By final validation,
65.86% of routed tokens are considered uncertain on average, so the
optional-second-expert path remains active. The study does not test whether
recalibrating that threshold during training helps.

It also changes two choices together, rescue priority and selective second
requests. A separate rescue-only ablation would help measure how much each
choice contributes. The current result supports the combined policy.

## Code and reproduction

`src/gpt2_nano_moe/router.py` implements the shared-capacity policies.
`experts.py` runs the experts and `moe.py` combines their accepted outputs.
`publication_study.py` handles the matched initialization, calibration, training,
and validation. `runtime.py` records deterministic execution settings.

With the project dependencies installed, the measured workflow is

```bash
python scripts/prepare_publication_data.py
./scripts/publication_study.sh
```

The run settings are in `configs/publication_study.toml`. The raw evidence is
`artifacts/publication-routing-study.json`, including per-seed curves, routing
counts, input hashes, and environment details. `artifacts/manifest.json` records
the artifact hash. The recorded source is `3522c79`.

The repository includes the code and results, not a selected publication
checkpoint.

## What I take from the result

When expert slots are scarce, the order of requests matters. In this small
experiment, primary, rescue, then optional admission improves mean coverage
while requesting fewer slots than fixed top-2. Language quality and execution
speed remain open questions.

## References

1. Shazeer and collaborators. *Outrageously Large Neural Networks*.
   arXiv:1701.06538.
2. Fedus, Zoph, and Shazeer. *Switch Transformers*. arXiv:2101.03961.
3. Riquelme and collaborators. *Scaling Vision with Sparse Mixture of Experts*.
   arXiv:2106.05974.

The code and measured evidence are available in
[gpt2-nano-moe on GitHub](https://github.com/kotlarmilos/gpt2-nano-moe)
and
[gpt2-nano-moe on Hugging Face](https://huggingface.co/kotlarmilos/gpt2-nano-moe).
