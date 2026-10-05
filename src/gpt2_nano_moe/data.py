from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def synthetic_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(
        vocab_size,
        (batch_size, context_length),
        generator=generator,
        device="cpu",
    ).to(device)


def load_corpus(
    path: str | Path,
    *,
    vocab_size: int,
    context_length: int,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)
    header = np.fromfile(corpus_path, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError("corpus has a truncated shard header")
    if int(header[0]) != 20250320 or int(header[1]) != 1:
        raise ValueError("corpus has an unsupported shard header")
    num_tokens = int(header[2])
    expected_size = 256 * 4 + num_tokens * 2
    if num_tokens < 1 or corpus_path.stat().st_size != expected_size:
        raise ValueError("corpus size does not match its shard header")
    tokens = np.fromfile(
        corpus_path,
        dtype="<u2",
        count=num_tokens,
        offset=256 * 4,
    )
    if tokens.size < 2 * context_length:
        raise ValueError("corpus must contain at least two context windows")
    if int(tokens.max()) >= vocab_size:
        raise ValueError("corpus contains a token outside the configured vocabulary")
    split = int(tokens.size * (1.0 - validation_fraction))
    if split < context_length or tokens.size - split < context_length:
        raise ValueError("validation_fraction leaves a split shorter than one context")
    return tokens[:split], tokens[split:]


def corpus_batch(
    tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    starts = torch.randint(
        tokens.size - context_length + 1,
        (batch_size,),
        generator=generator,
    ).tolist()
    rows = np.stack(
        [tokens[start : start + context_length] for start in starts]
    ).astype(np.int64, copy=False)
    return torch.from_numpy(rows).to(device)
