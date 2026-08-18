---
title: "Rescuing dropped tokens in capacity-limited MoE routing"
thumbnail: /blog/assets/nano-sparse-moe/thumbnail.png
authors:
- user: kotlarmilos
---

# Rescuing dropped tokens in capacity-limited MoE routing

## TL;DR

I test a routing problem that appears after the usual sparse MoE demonstration.
Top-2 routing can improve expert choice, but two assignments per token compete for
one finite capacity budget per expert. Fixed top-2 then spends capacity on second
assignments while some tokens receive no expert at all. The capacity-aware
selective policy in this repository prioritizes one accepted route, rescues
rejected tokens through their second choice, and spends remaining capacity on a
second expert only for uncertain tokens.

## The problem

MoE changes the MLP path available to a token. It does not add attention capacity
or route attention heads. The engineering problem is allocation under congestion.
Every expert has a finite batch capacity, but the router can send many tokens to
the same expert. Top-1 can drop an overloaded token. Fixed top-2 creates twice as
many requested slots without doubling the shared capacity. A useful routing policy
must distinguish two failures. A token can receive no expert, or it can lose an
optional second route after already receiving one.

## Intuition

### Route primary assignments before optional work

The router projects each token to expert probabilities and selects its two highest
scores. Batch-prioritized routing first admits primary choices by score within each
expert capacity. The selective policy then tries to rescue every rejected primary
through its second-choice expert. Only after that rescue pass does it admit second
experts for accepted-primary tokens whose top-two probability margin is below a
calibrated threshold.

### Capacity and dropped tokens

Each expert accepts at most
`ceil(capacity_factor * token_count / expert_count)` slots across all routing
phases. Primary, rescue, and optional secondary assignments therefore share one
budget. A token is dropped only when neither its primary nor rescue route is
accepted. The residual stream still continues for that token.

### Keeping the router observable

The study reports token drop and slot drop separately. Token drop means that a
token receives no expert. Slot drop counts rejected requests, including optional
second assignments. It also reports effective experts per token, uncertainty rate,
and the fraction of rejected primaries rescued through their second choice. These
metrics expose whether a policy uses capacity to cover more tokens or to add more
expert computation to tokens already covered.

## The math

For `N` tokens, `E` experts, and capacity factor `phi`, every expert receives the
same maximum number of slots.

$$
C = \left\lceil \frac{\phi N}{E} \right\rceil
$$

Fixed top-2 requests `2N` slots. CapSel-2 requests one primary slot for every
token, one rescue slot for each rejected primary, and one optional slot for each
uncertain accepted primary.

$$
R_{\mathrm{CapSel}} =
N + |\mathrm{rejected\ primaries}| + |\mathrm{uncertain\ accepted\ primaries}|
$$

The primary uncertainty score is the probability margin.

$$
m_i = p_{i,1} - p_{i,2}
$$

A smaller margin means the router is less certain about its first choice. The
study calibrates one median margin threshold from eight training batches before
optimization starts.

The model still reports total, trainable, and structural active parameters. Active
parameter count is not a speed claim. The implementation dispatches with PyTorch
indexing and Python-level expert iteration. The publication study therefore tests
routing and early optimization, not an optimized MoE kernel.

## In the code

The shared-capacity fixed top-2 and CapSel-2 policies live in
`src/gpt2_nano_moe/router.py`. Expert execution lives in
`src/gpt2_nano_moe/experts.py`, and `src/gpt2_nano_moe/moe.py` combines accepted
gates and expert outputs. `src/gpt2_nano_moe/publication_study.py` calibrates the
threshold, reconstructs identical initial weights for every condition, resets the
batch generator for every matched run, trains three seeds, and evaluates sixteen
held-out batches. `src/gpt2_nano_moe/runtime.py` requires deterministic CUDA
algorithms and records the selected SDPA backends.

## Results and honesty boundary

I ran five matched routing conditions on an NVIDIA RTX A5000. Every condition uses
the same six-layer width-384 model, eight experts, real token shard, 500 optimizer
steps, three seeds, initial weights, token batches, and held-out batches. The run
uses deterministic math SDPA. Each seed processes 256,000 training tokens per
condition. The table reports population mean and standard deviation over three
seeds.

| Routing condition | Token drop | Slot drop | Experts per token | Rescue rate | Held-out CE |
|---|---:|---:|---:|---:|---:|
| Top-1 token order, CF 1.25 | 27.37% ± 1.26% | 27.37% ± 1.26% | 0.726 ± 0.013 | n/a | 4.357 ± 0.034 |
| Top-1 BPR, CF 1.25 | 30.90% ± 1.26% | 30.90% ± 1.26% | 0.691 ± 0.013 | n/a | 4.349 ± 0.035 |
| Fixed top-2, CF 1.25 | 13.84% ± 3.46% | 45.91% ± 1.33% | 1.082 ± 0.027 | n/a | 4.402 ± 0.047 |
| Top-1 token order, CF 2.0 | 9.88% ± 3.74% | 9.88% ± 3.74% | 0.901 ± 0.037 | n/a | 4.354 ± 0.031 |
| CapSel-2, CF 1.25 | 8.09% ± 0.87% | 38.67% ± 0.75% | 1.037 ± 0.017 | 58.27% ± 4.62% | 4.393 ± 0.040 |

CapSel-2 requests 83,150 slots on average across final validation, compared with
98,304 for fixed top-2. It requests 15.4 percent fewer slots and accepts 4.1
percent fewer, yet its mean zero-expert token drop falls by 5.74 percentage points.
This is a 41.5 percent relative reduction. Two of three paired seeds improve token
drop substantially. The third is 0.12 percentage points worse. The result supports
the capacity-allocation mechanism, not a universal dominance claim.

The held-out cross-entropy difference between CapSel-2 and fixed top-2 is only
`0.0089` in favor of CapSel-2 on the three-seed mean. One paired seed moves in the
opposite direction, and every condition has a larger seed standard deviation than
that mean gap. Five hundred steps are not convergence, so the study does not
establish a language-quality winner.

The threshold `0.03658` is the median router margin from 24,576 samples at
initialization. It remains fixed for all seeds. Final CapSel uncertainty averages
65.86 percent, so the optional-secondary path remains active, but this study does
not test threshold recalibration during training. Exact per-seed curves, routing
counts, initialization hashes, corpus hash, environment, and deterministic backend
settings live in `artifacts/publication-routing-study.json`.

The original smoke artifacts remain useful implementation checks. They are not
evidence for the publication routing claim. The Space also remains a synthetic
routing viewer and does not serve a selected publication checkpoint.

Expected publication locations, not yet live:

- Source repository at https://github.com/kotlarmilos/gpt2-nano-moe
- Model repository at https://huggingface.co/kotlarmilos/gpt2-nano-moe
- Interactive Space at https://huggingface.co/spaces/kotlarmilos/gpt2-nano-moe

## References

- Noam Shazeer and collaborators, Outrageously Large Neural Networks at
  https://arxiv.org/abs/1701.06538
- Dmitry Lepikhin and collaborators, GShard at
  https://arxiv.org/abs/2006.16668
- William Fedus, Barret Zoph, and Noam Shazeer, Switch Transformers at
  https://arxiv.org/abs/2101.03961
- Carlos Riquelme and collaborators, Scaling Vision with Sparse Mixture of
  Experts at https://arxiv.org/abs/2106.05974
