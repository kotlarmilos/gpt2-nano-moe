from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr
import torch

from gpt2_nano_moe import GPT, ModelConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "smoke.pt"


def configured_checkpoint() -> Path:
    return Path(os.environ.get("CHECKPOINT_PATH", DEFAULT_CHECKPOINT))


def load_model(checkpoint_path: str | Path) -> GPT:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Set CHECKPOINT_PATH to a compatible file."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError("Checkpoint must contain model_config and model_state_dict")
    config = ModelConfig(**payload["model_config"])
    if config.architecture != "moe":
        raise ValueError("Expert visualization requires an MoE checkpoint")
    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def prompt_to_tokens(prompt: str, vocab_size: int, context_length: int) -> torch.Tensor:
    encoded = list(prompt.encode("utf-8")) or [0]
    encoded = [token % vocab_size for token in encoded[-context_length:]]
    return torch.tensor([encoded], dtype=torch.long)


def inspect_prompt(prompt: str) -> str:
    checkpoint = configured_checkpoint()
    try:
        model = load_model(checkpoint)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise gr.Error(str(error)) from error
    token_ids = prompt_to_tokens(
        prompt, model.config.vocab_size, model.config.context_length
    )
    with torch.no_grad():
        output = model(token_ids)
    labels = [
        bytes([token]).decode("utf-8", errors="replace")
        if token < 256
        else str(token)
        for token in token_ids[0].tolist()
    ]
    header = "| token | id | " + " | ".join(
        f"layer {index}" for index in range(len(output.expert_indices))
    ) + " |\n"
    divider = "|---|---|" + "---|" * len(output.expert_indices) + "\n"
    rows = []
    for position, label in enumerate(labels):
        experts = [
            str(indices[0, position].item()) for indices in output.expert_indices
        ]
        rows.append(
            "| `"
            + html.escape(label)
            + "` | "
            + str(token_ids[0, position].item())
            + " | "
            + " | ".join(experts)
            + " |"
        )
    return header + divider + "\n".join(rows)


checkpoint = configured_checkpoint()
checkpoint_ready = checkpoint.is_file()
checkpoint_status = (
    f"Checkpoint ready at `{checkpoint}`."
    if checkpoint_ready
    else (
        f"Checkpoint missing at `{checkpoint}`. Add a compatible checkpoint "
        "or set `CHECKPOINT_PATH`, then restart the Space."
    )
)

with gr.Blocks(title="GPT2 Nano MoE Router") as demo:
    gr.Markdown("# GPT2 Nano MoE expert routing")
    gr.Markdown(
        "Enter text to inspect the top-1 expert selected for each byte token and layer."
    )
    gr.Markdown(checkpoint_status)
    prompt = gr.Textbox(label="Prompt", value="Sparse experts")
    result = gr.Markdown()
    gr.Button("Inspect routing", interactive=checkpoint_ready).click(
        inspect_prompt, inputs=prompt, outputs=result
    )


if __name__ == "__main__":
    demo.launch()
