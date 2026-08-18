# Learning sparse Mixture of Experts from the implementation

This note derives the sparse Mixture of Experts transformer in this repository from first principles and ties each equation to the implementation. The attention stack stays dense. The only architectural substitution is that a transformer block can replace one dense feed-forward MLP with `SparseMoE`, which contains `Top1Router` and `Experts`.

## Dense feed-forward block

### Motivation

A decoder transformer alternates attention with a position-wise feed-forward block. Attention mixes information across sequence positions. The feed-forward block then applies the same nonlinear map to each token independently, so it provides much of the model capacity inside each block.

Let one token representation be `x` with width `d`. A dense feed-forward block expands to hidden width `h`, applies a nonlinearity, and projects back to width `d`.

```
x in R^d
u = W1 x + b1
m = GELU(u)
y = W2 m + b2
W1 in R^(h x d)
W2 in R^(d x h)
```

The parameter count for one dense feed-forward block is the count of both matrices and both bias vectors.

```
params_dense_ffn = h*d + h + d*h + d
                 = 2*d*h + h + d
```

This document counts one multiply-add as one unit of matrix work. A two-FLOP convention multiplies the leading matrix terms by two. The per-token dense feed-forward matrix work is dominated by the two matrix products.

```
work_dense_ffn_per_token = d*h + h*d
                         = 2*d*h
lower_order_work = O(h) for GELU and biases
```

The important constraint appears immediately. If we make the dense hidden layer wider, both parameters and per-token compute grow together.

```
h -> q*h
params_dense_ffn -> about q * params_dense_ffn
work_dense_ffn_per_token -> q * work_dense_ffn_per_token
```

A sparse MoE layer tries to break this coupling. It adds many feed-forward parameter sets, but it activates only a small number for each token.

### Derivation

The repository uses the standard transformer residual pattern. The block first computes attention, adds it to the residual stream, normalizes, and then calls either the dense MLP or the sparse MoE MLP.

```
hidden = x + Attention(LayerNorm(x))
normalized = LayerNorm(hidden)
out_dense = hidden + DenseMLP(normalized)
```

The dense MLP therefore defines the per-token baseline cost that the sparse layer replaces. If the model has `L` layers and every layer is dense, the feed-forward work for `N` token positions is approximately

```
work_dense_model_ffn = N * L * 2*d*h
```

This ignores attention and embeddings because MoE does not change those paths in this repository.

### Where it lives in the code

`DenseMLP` implements the two linear maps, `GELU`, and dropout in `src/gpt2_nano_moe/dense.py` lines 6 to 17. The expansion width is `hidden_size = config.d_model * config.mlp_multiplier` in `src/gpt2_nano_moe/model.py` line 72. `TransformerBlock` selects `DenseMLP` when `config.architecture` is not `"moe"` in `src/gpt2_nano_moe/model.py` lines 81 to 82, and it adds the dense result back to the residual stream in line 102.

`ModelConfig` defines the dense dimensions in `src/gpt2_nano_moe/config.py` lines 13 to 23 and validates positive sizes in lines 25 to 47. The article states the same boundary in `article.md` lines 8 to 11. Attention stays unchanged, and MoE changes the MLP width available to a token.

## Sparse experts and the router

### Motivation

The MoE idea replaces one feed-forward function with `E` feed-forward functions and a learned router. A token does not evaluate every expert. It evaluates only the selected expert set. This gives the central decoupling.

```
parameters grow with E
expert compute per token grows with k
```

Here `E` is the number of experts and `k` is the number of selected experts per token. The repository implements `k = 1`. The general top-k derivation explains the idea, and the code then specializes it to top-1.

### Derivation

Let expert `e` be a dense feed-forward function `f_e` with the same input and output width as the dense MLP.

```
f_e(x) = W2_e GELU(W1_e x + b1_e) + b2_e
for e in {1, ..., E}
```

The expert parameter count is now multiplied by `E`.

```
params_experts = E * (2*d*h + h + d)
```

The router maps the token from width `d` to `E` logits. The repository uses a bias-free linear projection.

```
z_e = r_e^T x
z in R^E
params_router = d*E
```

A softmax turns logits into a probability distribution over experts.

```
p_e = exp(z_e) / sum_j exp(z_j)
sum_e p_e = 1
p_e >= 0
```

Generic top-k routing chooses the `k` largest probabilities.

```
S_k(x) = indices of the k largest entries of p
```

If a generic top-k layer combines multiple experts, it usually renormalizes the selected probabilities over the selected set.

```
alpha_e = p_e / sum_{j in S_k(x)} p_j    if e in S_k(x)
alpha_e = 0                              if e not in S_k(x)
sum_e alpha_e = 1
```

The expert matrix work for one token is then approximately

```
work_moe_experts_per_token = k * 2*d*h
work_router_per_token = d*E
work_selection_per_token = O(E) for top-k selection
```

The dominant feed-forward matrix work depends on `k`, not `E`. The router still scans the expert count, but it is a small projection compared with evaluating all expert MLPs when `h` is large. The sparse layer therefore increases parameters with `E` while keeping active expert compute tied to `k`.

The repository sets `k = 1`. The selected expert is the argmax of the softmax probabilities.

```
s = argmax_e p_e
g = p_s
```

A strictly renormalized top-1 combine would make the selected weight equal to one because the selected set contains one element.

```
alpha_s = p_s / p_s = 1
```

This repository does not use that renormalized top-1 weight. It keeps the selected softmax probability as a gate, so the expert output is scaled by `g = p_s`. This matches the article statement that top-1 routing selects the largest probability and uses it as a gate.

### Where it lives in the code

`SparseMoE` constructs `Top1Router` and `Experts` in `src/gpt2_nano_moe/moe.py` lines 22 to 33. `TransformerBlock` replaces `DenseMLP` with `SparseMoE` when `config.architecture == "moe"` in `src/gpt2_nano_moe/model.py` lines 73 to 80.

`Top1Router` stores `expert_count`, `capacity_factor`, and the bias-free router projection in `src/gpt2_nano_moe/router.py` lines 53 to 59. The softmax distribution is computed in line 64. The top-1 selection is `probabilities.max(dim=-1)` in line 65, which returns both `gates` and `expert_indices`.

`Experts` stores one `DenseMLP` per expert in a `ModuleList` in `src/gpt2_nano_moe/experts.py` lines 9 to 16. The total and active parameter accounting appears in `GPT.parameter_counts` in `src/gpt2_nano_moe/model.py` lines 202 to 215. For each MoE block, it subtracts all expert parameter sets except one from the structural active count, so routers and shared transformer parameters remain active while only one expert per MoE layer is counted as active for a token.

`ModelConfig.expert_count`, `ModelConfig.capacity_factor`, and `ModelConfig.auxiliary_loss_weight` live in `src/gpt2_nano_moe/config.py` lines 20 to 23. The article motivates the same decoupling in `article.md` lines 15 to 22 and warns that active parameter count is not a speed claim in lines 58 to 61.

The `top_k` argument in `GPT.generate` is unrelated to MoE routing. It filters vocabulary logits during sampling in `src/gpt2_nano_moe/model.py` lines 175 to 193.

## Combining expert outputs and routing gradients

### Motivation

The router must influence the model output, or it cannot learn which expert is useful. The combine step gives the router a differentiable path through the selected gate values. The discrete choice of expert remains non-differentiable, but the selected probability value is differentiable.

### Derivation

For a generic top-k layer with renormalized selected weights, the combined output is a weighted sum of expert outputs.

```
y(x) = sum_{e in S_k(x)} alpha_e f_e(x)
```

The derivative with respect to a selected weight is direct.

```
dL/d alpha_e = < dL/dy, f_e(x) >
```

The selected weights depend on the softmax probabilities, and the softmax probabilities depend on router logits. The softmax derivative is

```
d p_s / d z_e = p_s * (1[e = s] - p_e)
```

For the repository top-1 combine, let `a` be the capacity admission indicator. It is one for an accepted token assignment and zero for a dropped assignment.

```
s = argmax_e p_e
g = p_s
y = a * g * f_s(x)
```

For an accepted token, the language-model loss gives the selected gate this derivative.

```
dL/dg = < dL/dy, f_s(x) >
```

The router logit derivative from the language-model loss follows through the selected softmax probability.

```
dL/dz_e = dL/dg * p_s * (1[e = s] - p_e)
```

This gradient updates every router logit for the token because the selected softmax probability depends on every logit through the denominator. The argmax index itself does not receive a gradient. The capacity mask also does not receive a gradient.

Dropped tokens have `a = 0`, so the MoE contribution is zero.

```
y = 0
dL/dg from the language-model path = 0
```

The router can still receive gradient from the auxiliary load-balancing loss for dropped tokens because that loss uses the full softmax probability matrix.

### Where it lives in the code

`SparseMoE.forward` flattens the batch and sequence dimensions in `src/gpt2_nano_moe/moe.py` lines 35 to 38. It passes the flat tokens, expert indices, and accepted mask to `Experts` in line 39. It multiplies each expert output by `route.gates.unsqueeze(-1)` in line 40 and reshapes back to the original tensor shape in lines 41 to 47.

`Experts.forward` creates a zero tensor in `src/gpt2_nano_moe/experts.py` line 24. It loops over experts in line 25, builds the selected accepted mask in line 26, and evaluates only selected tokens in lines 27 to 28. Tokens that are not accepted keep the initial zero output.

`RouterOutput` carries `probabilities`, `expert_indices`, `gates`, `accepted`, `auxiliary_loss`, and `metrics` in `src/gpt2_nano_moe/router.py` lines 40 to 47. `Top1Router.forward` returns the selected gates cast back to the input dtype in lines 94 to 101. `TransformerBlock.forward` adds `result.hidden_states` to the residual stream in `src/gpt2_nano_moe/model.py` lines 94 to 101.

`GPT.loss` forms the next-token cross entropy and then adds the weighted auxiliary loss in `src/gpt2_nano_moe/model.py` lines 154 to 168. The training loop calls `loss.backward()` in `src/gpt2_nano_moe/train.py` line 149, so gradients flow through the selected gate values, selected experts, shared transformer weights, and the auxiliary router term.

## Load balancing

### Motivation

A router trained only by the task loss can collapse. If many early updates make one expert slightly better, the router can send more tokens to that expert. That expert then receives more training signal, which can make it even better. Other experts receive little data, capacity drops increase, and the extra parameters stop being useful.

The auxiliary load-balancing loss gives the router a direct pressure to distribute assignments across experts. It does not guarantee perfectly equal expert use in each batch, but it makes collapse expensive.

### Derivation

For one MoE layer with `N` flattened tokens and `E` experts, define the top-1 assignment fraction for expert `e`.

```
f_e = (1/N) * sum_i 1[s_i = e]
```

Define the mean router probability mass assigned to expert `e` before top-1 selection.

```
P_e = (1/N) * sum_i p_i,e
```

The repository uses the Switch-style product loss.

```
L_aux_layer = E * sum_e f_e * P_e
```

If both assignment fractions and probability masses are uniform, the loss has value one.

```
f_e = 1/E
P_e = 1/E
L_aux_layer = E * sum_e (1/E) * (1/E)
            = E * E * (1/E^2)
            = 1
```

If every token selects one expert and the router probability mass also concentrates there, the loss grows to `E`.

```
f_1 = 1, P_1 = 1
f_e = 0, P_e = 0 for e != 1
L_aux_layer = E
```

The assignment fraction uses argmax indices, so it acts as a stop-gradient statistic in practice. The probability fraction remains differentiable. The auxiliary gradient for a token logit is

```
dL_aux_layer/dz_i,a = (E/N) * sum_e f_e * d p_i,e / d z_i,a
```

With the softmax derivative, every token contributes a balancing gradient to the router logits.

```
d p_i,e / d z_i,a = p_i,e * (1[e = a] - p_i,a)
```

The model averages auxiliary losses across transformer blocks and scales the average by `auxiliary_loss_weight` before adding it to the language-model cross entropy.

```
L_total = L_cross_entropy + lambda_aux * mean_layers(L_aux_layer)
```

### Where it lives in the code

`Top1Router.forward` computes `assignment_fraction` with `one_hot(expert_indices).float().mean(dim=0)` in `src/gpt2_nano_moe/router.py` lines 74 to 76. It computes `probability_fraction = probabilities.mean(dim=0)` in line 77. It computes `auxiliary_loss = expert_count * sum(assignment_fraction * probability_fraction)` in lines 78 to 80.

`GPT.forward` appends every block auxiliary loss in `src/gpt2_nano_moe/model.py` lines 136 to 146 and returns their mean in lines 146 to 152. `GPT.loss` adds `self.config.auxiliary_loss_weight * output.auxiliary_loss` to cross entropy in lines 154 to 168. `ModelConfig` defaults `auxiliary_loss_weight` to `0.01` and rejects negative values in `src/gpt2_nano_moe/config.py` lines 20 to 23 and 44 to 47.

The article describes this as a Switch-style load-balancing term in `article.md` lines 44 to 50. The training loop records the separate `language_model_loss`, `auxiliary_loss`, and `optimization_loss` in `src/gpt2_nano_moe/train.py` lines 181 to 199, which keeps validation and reporting honest about what comes from token prediction and what comes from routing regularization.

## Capacity and dropping

### Motivation

Sparse MoE needs a bounded dispatch shape. If one expert receives too many tokens, a real system cannot let that expert create an unbounded per-step buffer while other experts sit idle. A capacity factor gives each expert a fixed token budget for the batch.

This repository implements the admission rule directly with boolean masks instead of an optimized dispatch buffer. The mathematical effect is the same. Each expert accepts at most a fixed number of token assignments.

### Derivation

Let `N` be the number of flattened token positions in one MoE layer.

```
N = batch_size * sequence_length
```

Let `phi` be the capacity factor and `E` be the expert count. The per-expert capacity is

```
C = ceil(phi * N / E)
```

The total admitted assignments can be at most

```
max_admitted = E * C
```

If routing is balanced and `phi >= 1`, this capacity usually admits all tokens. If routing is skewed, some experts can overflow even when total capacity across experts is large enough.

For each token `i`, define its rank among tokens assigned to the same selected expert in flattened batch and sequence order.

```
rank_i = number of earlier or equal tokens j with s_j = s_i
a_i = 1 if rank_i <= C
      0 otherwise
```

The implemented top-1 MoE output is therefore

```
y_i = a_i * p_i,s_i * f_s_i(x_i)
```

Dropping means `a_i = 0`. The expert path contributes zero, and the transformer residual stream still carries the token forward.

```
block_output_i = hidden_i + y_i
if dropped, block_output_i = hidden_i
```

The tradeoff is direct. A larger capacity factor reduces drops under skewed routing, but it increases the amount of expert work and memory that an optimized implementation would need to reserve. A smaller capacity factor bounds expert work more tightly, but it can remove the MoE contribution from more tokens.

### Where it lives in the code

`SparseMoE.forward` creates flattened token order with `inputs.reshape(-1, shape[-1])` in `src/gpt2_nano_moe/moe.py` line 37. `Top1Router.forward` reads `token_count` in `src/gpt2_nano_moe/router.py` line 66 and computes capacity through `expert_capacity` in lines 67 to 69. `expert_capacity` defines `max(1, ceil(capacity_factor * token_count / expert_count))` in `src/gpt2_nano_moe/router.py` lines 12 to 17.

The deterministic admission rule lives in `deterministic_capacity_mask` in `src/gpt2_nano_moe/router.py` lines 20 to 37. `Top1Router.forward` calls it in lines 70 to 72. It initializes every assignment as not accepted, finds positions for each expert in flattened order, and marks only `positions[:capacity]` as accepted. Later overflowing tokens remain dropped.

`Experts.forward` preserves the zero output for dropped tokens in `src/gpt2_nano_moe/experts.py` lines 24 to 29. `TransformerBlock.forward` adds the MoE output to the residual stream in `src/gpt2_nano_moe/model.py` lines 94 to 101, so a dropped MoE assignment becomes a residual pass-through for that block.

Routing metrics record the capacity, token count, dropped token assignments, accepted counts, entropy, and utilization in `src/gpt2_nano_moe/router.py` lines 81 to 93. `RoutingMetrics.dropped_fraction` and `RoutingMetrics.to_dict` expose those values in `src/gpt2_nano_moe/metrics.py` lines 18 to 28. `aggregate_routing_metrics` accumulates them across MoE layers and benchmark iterations in `src/gpt2_nano_moe/metrics.py` lines 31 to 57.

The article states the same deterministic capacity rule in `article.md` lines 34 to 40. It also states that the residual stream continues when a token is dropped from the expert path in lines 37 to 38.

## Fixed-capacity top-k routing extension

The extension study isolates routing slots from expert computation.
`_capture_probabilities` in `src/gpt2_nano_moe/routing_study.py` lines 247 to
271 captures the probability matrices produced by `Top1Router` on one real
forward pass. `compare_top_k_routing` in lines 180 to 198 then applies
alternative top-k selection to those same probabilities. This keeps the router,
checkpoint, token batch, expert count, and capacity factor fixed while changing
only the number of selected slots per token.

The shared capacity code now lives in `expert_capacity` and
`deterministic_capacity_mask` in `src/gpt2_nano_moe/router.py`. `Top1Router`
uses those helpers for the original top-1 path. `routing_metrics_for_top_k` in
`src/gpt2_nano_moe/routing_study.py` lines 91 to 137 uses the same helpers for
top-1 and top-2 measurement, so the extension does not create a second capacity
rule.

Fixed capacity means the per-expert budget remains

```
C = ceil(phi * N / E)
```

for every top-k value. `N` is the flattened token count, `E` is the expert count,
and `phi` is the capacity factor. The top-k study changes the number of routed
slots to

```
routed_slots = N * k
```

The drop-rate denominator is `routed_slots`. At `k = 2`, the router sends twice
as many slots into the same total accepted capacity. If every expert reaches
capacity, accepted counts can look more balanced while the drop rate rises. That
is the mechanism measured in `artifacts/routing-topk-fixed-capacity.json`, where
top-1 drops 6 of 256 slots and top-2 drops 192 of 512 slots at per-expert
capacity 40.

## Capacity-aware selective top-2 routing

The publication extension changes the optimization target from accepting as many
slots as possible to covering tokens before adding optional work. Fixed top-2
requests two expert slots for every token. Under one shared per-expert capacity,
that rule can accept a second route for one token while another token receives no
expert.

CapSel-2 uses three ordered admissions.

1. Admit primary experts with batch-prioritized routing.
2. Route rejected primaries to their second-choice expert when capacity remains.
3. Add a second expert for accepted-primary tokens only when their router margin is
   below the calibrated uncertainty threshold.

The accepted primary and secondary gates are renormalized over accepted routes.
One accepted route therefore receives gate one. Two accepted routes share unit
gate mass according to their original router probabilities.

The token-level and slot-level objectives differ.

```text
token_drop = tokens with zero accepted routes / tokens
slot_drop = rejected requested slots / requested slots
effective_experts = accepted slots / tokens
rescue_rate = rescued rejected primaries / rejected primaries
```

In the three-seed RTX A5000 study, fixed top-2 requests 98,304 final-validation
slots per seed. CapSel-2 requests 83,150 on average because it does not ask for an
optional second expert on every token. Fixed top-2 accepts 53,172 slots and leaves
13.84 percent of tokens without an expert. CapSel-2 accepts 50,986 slots but
leaves only 8.09 percent without an expert. Ordering scarce capacity around token
coverage matters more than maximizing accepted slot count.

CapSel-2 rescues 58.27 percent of rejected primary routes on average. Its effective
expert count is 1.037 per token, compared with 1.082 for fixed top-2. The method
therefore trades a small amount of extra-expert computation for fewer tokens that
lose the complete expert path.

The held-out cross-entropy means are 4.393 for CapSel-2 and 4.402 for fixed top-2.
The mean gap is smaller than the seed spread and changes sign for one paired seed.
This early-training run supports the routing mechanism but does not establish a
language-quality improvement.

`GeneralRouter._route_top2_capsel` in `src/gpt2_nano_moe/router.py` implements the
three phases. `aggregate_routing_metrics` in
`src/gpt2_nano_moe/metrics.py` pools rescue counts before computing the rescue
rate. `src/gpt2_nano_moe/publication_study.py` calibrates the initial margin,
recreates matched weights and batches, and evaluates sixteen held-out batches per
seed.

## Expert parallelism

### Motivation

Sparse MoE becomes most useful when experts can live on different devices. The model can hold many more expert parameters than one device would evaluate for one token, and routing sends each token only to the devices that own its selected experts.

### Derivation

The logical tensors needed for expert parallel dispatch are already present in the single-device implementation.

```
flat_tokens = reshape inputs to [N, d]
selected_expert = s_i
accepted = a_i
gate = g_i
```

An expert-parallel runtime would group accepted tokens by `selected_expert`, send each group to the device that owns that expert, evaluate local expert MLPs, multiply by gates, and send outputs back to the original token positions.

```
local_input_e = { x_i where s_i = e and a_i = 1 }
local_output_e = g_i * f_e(x_i)
result_i = local_output_e returned to original position i
```

When experts are sharded across devices, the send and return steps form an all-to-all communication pattern. The computation stays sparse, but routing introduces communication and load-balance constraints. Capacity matters because it bounds the maximum number of token representations sent to any expert.

### Where it lives in the code

This repository does not implement expert parallelism or all-to-all communication. It keeps all experts in one `nn.ModuleList` in `src/gpt2_nano_moe/experts.py` lines 14 to 16 and dispatches with a Python loop over experts in lines 25 to 29. Device selection chooses one execution device in `src/gpt2_nano_moe/runtime.py` lines 14 to 25, and training moves the whole `GPT` module to that device in `src/gpt2_nano_moe/train.py` lines 101 to 103.

The article explicitly describes this as straightforward PyTorch indexing and Python-level expert iteration rather than an optimized dispatch kernel in `article.md` lines 62 to 65. It names grouped dispatch as future work in `article.md` lines 122 to 123. The learning point is therefore conceptual rather than implemented. The tensors that would drive all-to-all dispatch are `expert_indices`, `accepted`, and `gates`, but the current code evaluates them inside one process.
