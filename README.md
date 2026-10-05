# Reducing dropped tokens in a small MoE model

This repository contains the deterministic CapSel-2 routing experiment described
in `article.md`.

On one NVIDIA RTX A5000, CapSel-2 reduces mean zero-expert token drop from 13.84%
for fixed top-2 to 8.09% while requesting 15.4% fewer expert slots. It rescues
58.27% of rejected primary routes.

The mean held-out cross-entropy difference is 0.0089. That difference is smaller
than the seed spread and changes sign for one seed. The experiment does not
establish a language-quality winner or a speed improvement.

## Reproduce

The measured run used Python 3.11.10 and PyTorch 2.4.1 with CUDA 12.4.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python scripts/prepare_publication_data.py
./scripts/publication_study.sh
```

The exact settings are in `configs/publication_study.toml`. The measured result
is `artifacts/publication-routing-study.json`. `artifacts/manifest.json` records
the result, config, source commit, input location, and SHA-256 values.

## Validate

```sh
python -m unittest -v tests.test_routing_experiment
python scripts/prepare_publication_data.py --verify-only
```

The focused test file covers capacity math, deterministic admission, rescue
ordering, metric denominators, gradient flow, provenance, input hash checks, and
a small CPU execution path.

The measured artifact is also published at
`https://huggingface.co/kotlarmilos/gpt2-nano-moe`.
