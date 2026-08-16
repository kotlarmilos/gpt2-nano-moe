---
title: GPT2 Nano MoE Router
emoji: 🧭
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# GPT2 Nano MoE Router

This standalone Space visualizes the top-1 expert selected for every byte token
at every MoE layer.

The staged `checkpoints/smoke.pt` is a four-step synthetic integration model.
It exists only to make expert routing visible and is not a language-quality
artifact. Replace it with the signed-off publication checkpoint before the
final Space release. If the file is absent, the Space starts in a gated state.

The local `gpt2_nano_moe` package contains the model implementation required by
`app.py`. No parent repository path or unpublished package is required.
