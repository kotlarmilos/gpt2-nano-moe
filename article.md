---
title: "Adding sparse Mixture of Experts to a small transformer"
thumbnail: /blog/assets/nano-sparse-moe/thumbnail.png
authors:
- user: kotlarmilos
---

# Adding sparse Mixture of Experts to a small transformer

## TL;DR

I wanted the smallest version of sparse Mixture of Experts that still exposes the
engineering decisions that matter. The result starts with a conventional causal
transformer and makes one focused substitution. Each dense MLP can become a bank of
MLP experts behind a learned router. The attention path does not change, so the
routing decision is the only variable under inspection.

## The problem

MoE changes the MLP width available to a token. It does not add attention capacity
and it does not route attention heads. Queries, keys, values, the causal mask, and
the residual connection all work exactly as they do in the dense baseline. The
engineering questions are how a token picks an expert, what happens when too many
tokens pick the same one, and how to count parameters and cost honestly when most
experts stay idle for any single token.

## Intuition

### Routing one token

The router is one linear projection from the model width to the expert count. A
softmax turns those logits into probabilities. Top-1 routing selects the largest
probability and uses it as a gate on the chosen expert output. Top-1 is useful for
an educational implementation because the active path is easy to inspect. Every
token chooses one expert per MoE layer. Total parameter count grows with the number
of experts while the expert parameters active for one token stay at the size of one
MLP.

### Capacity and dropped tokens

Capacity makes this sparse path less tidy. Each expert accepts
`ceil(capacity_factor * token_count / expert_count)` tokens. When too many tokens
select one expert, admission follows flattened batch and sequence order. Later
tokens are dropped from the expert path. The residual stream still continues, so a
dropped token does not disappear from the model. That admission rule is
deliberately deterministic. It is not an optimized dispatch kernel, but it gives
tests and readers a precise result to reason about.

### Keeping the router observable

The training loss adds a Switch-style load-balancing term. It compares the mean
router probability with the fraction of top-1 assignments for each expert. The
implementation also reports accepted counts, dropped token assignments, router
entropy, and the fraction of experts used. Entropy shows how concentrated the
router probabilities are. Utilization shows whether experts receive any accepted
tokens. Dropped counts show when capacity is the immediate bottleneck. The
auxiliary loss creates a gradient for routing balance, but it does not guarantee
equal expert use in every batch.

## The math

The model reports total parameters, trainable parameters, and structural active
parameters per token. For top-1 MoE, structural active count includes all shared
parameters, every router, and one expert in each MoE layer. Total count includes
every expert.

The smoke MoE has 47,168 total parameters and 22,016 active parameters per token.
The matched dense model has 21,760 total and active parameters. These counts come
from `artifacts/smoke-benchmark.json`. Active parameter count is not a speed claim.
This implementation dispatches with straightforward PyTorch indexing and
Python-level expert iteration. Real throughput depends on tensor shapes, device
kernels, batch balance, and dispatch overhead.

## In the code

The dense baseline MLP lives in `src/gpt2_nano_moe/dense.py`. The router
projection, softmax, and top-1 gate live in `src/gpt2_nano_moe/router.py`. Expert
banks and the deterministic capacity admission live in
`src/gpt2_nano_moe/experts.py`, and the MoE layer that combines gate and expert
output lives in `src/gpt2_nano_moe/moe.py`. The transformer that swaps a dense MLP
for the MoE layer lives in `src/gpt2_nano_moe/model.py`. Router observability
metrics live in `src/gpt2_nano_moe/metrics.py`, training and the load-balancing
term in `src/gpt2_nano_moe/train.py`, and dispatch iteration in
`src/gpt2_nano_moe/runtime.py`. `LEARNING.md` derives dense cost, top-1 gating
gradients, and the balancing loss with exact line references, and notes that the
repo implements top-1 gating rather than generic renormalized top-k routing.

## Results and honesty boundary

I ran the committed smoke configuration on an arm64 Mac with Python 3.13.3 and
PyTorch 2.13.0 using CPU execution. Four synthetic training steps processed 128
tokens. The final language-model cross entropy was 4.8489. Adding the weighted
routing term produced a 4.8590 optimization loss. Across the two MoE layers, the
final step recorded 2 dropped assignments out of 64 and full expert utilization.

The regenerated seeded forward benchmark processed 320 tokens over ten iterations
for each model. The dense path measured 96,890 tokens per second and the MoE path
measured 32,205 tokens per second. Routing metrics accumulate all 640 MoE
layer-token assignments from the timed iterations. No assignment was dropped and
all experts were used. This is a smoke measurement on tiny tensors, not a general
performance comparison. Exact timings and environment metadata live in
`artifacts/smoke-train.json` and `artifacts/smoke-benchmark.json`.

Development-scale extension result. A local run in
`artifacts/routing-topk-fixed-capacity.json` reuses the smoke MoE checkpoint and
one seeded synthetic fixed batch with 128 tokens per MoE layer. The study fixes 4
experts, capacity factor 1.25, and per-expert capacity 40, then recomputes
routing metrics from the same router probabilities for top-1 and top-2 slots.
Top-1 routes 256 slots across two layers, drops 6, and records a 0.0234 drop
rate. Top-2 routes 512 slots at the same capacity, drops 192, and records a
0.3750 drop rate. Accepted expert-count coefficient of variation changes from
0.2082 to 0.0000, and max-to-mean load changes from 1.2800 to 1.0000. The
slot-based load-balancing loss changes from 1.0045 to 1.0020. The hypothesis
holds in this development-scale setting because top-2 raises drops while the
accepted load becomes perfectly capped across experts. The result does not
measure language quality for a top-2 model because no top-2 dispatch path is
trained or evaluated.

The publication configuration uses the same 9,157-token vocabulary and verified
FineWeb-Edu token shard as the base `gpt2-nano` project. The trainer reads the
little-endian `uint16` corpus and creates a deterministic held-out split. Dense and
MoE jobs use the same seed, token windows, optimizer settings, and split.
Validation compares language-model cross entropy rather than total optimization
loss because only the MoE objective contains the auxiliary routing term. The
publication benchmark requires both trained checkpoints and samples the held-out
tokens. Those runs remain pending, and the artifact manifest records that status
without filling in results that do not exist.

A trained top-2 dispatch implementation remains a useful algorithmic comparison.
A grouped dispatch implementation remains the next performance step. The
publication run already fixes the tokenizer, real corpus, split, and training
budget. It still needs accelerator execution across multiple seeds. The local
Space package includes the synthetic smoke checkpoint so routing can be reviewed
before publication. It must be replaced by the selected trained checkpoint for
the final release.

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
