# Adding sparse Mixture of Experts to a small transformer

This repository is a compact GPT-style decoder for studying the difference
between a dense MLP and a top-1 sparse Mixture of Experts MLP. Attention stays
unchanged. The MoE option replaces each block's MLP with a router and a
configurable bank of experts.

## Included

- Typed top-1 routing with deterministic token-order capacity admission
- Configurable expert count, capacity factor, and auxiliary loss weight
- Dropped-token, router entropy, expert utilization, and expert count metrics
- Explicit total, trainable, and active-per-token parameter counts
- Matching dense and sparse model, training, generation, and benchmark APIs
- CPU, MPS, and CUDA device selection
- Seeded synthetic smoke workloads with environment metadata

## Setup

Python 3.11 through 3.13 and PyTorch 2.2 through 2.x are supported.

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Validate

```sh
python -m unittest discover -s tests -v
./scripts/smoke_train.sh
./scripts/smoke_benchmark.sh
```

The smoke scripts write measured JSON to `artifacts/`. They use seeded
synthetic tokens to validate execution, not to estimate language-model
quality. `artifacts/manifest.json` separates measured, simulated, and pending
results.

## Use the interfaces

```python
import torch
from gpt2_nano_moe import GPT, ModelConfig

model = GPT(ModelConfig(architecture="moe", expert_count=4))
tokens = torch.randint(model.config.vocab_size, (2, 16))
loss, output = model.loss(tokens)
loss.backward()
generated = model.generate(tokens[:, :2], max_new_tokens=8, top_k=20)
print(model.parameter_counts().to_dict())
```

Run a configured training workload with:

```sh
gpt2-nano-moe-train \
  --config configs/smoke.toml \
  --output artifacts/local-train.json \
  --checkpoint checkpoints/local.pt
```

Run the matched dense and MoE forward benchmark with:

```sh
gpt2-nano-moe-benchmark \
  --config configs/smoke.toml \
  --iterations 10 \
  --output artifacts/local-benchmark.json
```

Run publication-scale training and benchmarking with:

```bash
python scripts/prepare_publication_data.py
ARCHITECTURE=dense SEED=17 ./scripts/publication_train.sh
ARCHITECTURE=moe SEED=17 ./scripts/publication_train.sh
./scripts/publication_benchmark.sh
```

The publication trainer reads the verified `uint16` token corpus, holds out ten
percent for validation, and records validation loss beside routing metrics.
Run one job per Colab runtime:

```bash
EXPERIMENT=train ARCHITECTURE=dense SEED=17 ./scripts/colab_publication.sh
EXPERIMENT=train ARCHITECTURE=moe SEED=17 ./scripts/colab_publication.sh
EXPERIMENT=benchmark ./scripts/colab_publication.sh
```

## Repository map

- `src/gpt2_nano_moe/` contains model, router, experts, metrics, and CLIs.
- `configs/` contains smoke and publication-scale configurations.
- `tests/` covers routing, capacity, gradients, forwards, validation, and counts.
- `space/` is a standalone Hugging Face Space root with vendored model source
  and a checkpoint-driven expert-routing viewer.
- `hf/README.md` is the Hugging Face model card.
- `article.md` contains the accompanying article draft.

This project uses a byte-sized synthetic vocabulary in smoke mode and does not
claim tokenizer or checkpoint compatibility with OpenAI GPT-2 weights.
