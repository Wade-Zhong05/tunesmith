"""Sampling strategies for the composer: temperature, top-k and nucleus (top-p).

>>> STUDENT-IMPLEMENTED FILE <<<
Write the three functions marked TODO. `sample_next` (provided) chains them and is
what the generation loop calls. Disabled settings (temperature == 1, top_k == 0,
top_p >= 1) never call your functions, so the app works before you finish.

Check your work with:  pixi run pytest tests/test_sampling_filters.py -q

All functions take logits of shape (B, V) and return a tensor of the same shape.
Tokens that must never be sampled are already -inf (grammar mask); keep them -inf.

AI attribution: skeleton and docstrings drafted with Claude Code (Claude Opus 5.5),
2026-11-10; function bodies to be written by the student. Update ATTRIBUTION.md when done.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by `temperature` (> 0). Lower = more conservative, higher = more random."""
    raise NotImplementedError("TODO(student): apply_temperature")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits in each row and set the rest to -inf.

    If k >= V, return the logits unchanged. Hint: torch.topk gives the k-th largest
    value of each row; compare against it (ties may keep a few extra tokens, that is fine).
    """
    raise NotImplementedError("TODO(student): top_k_filter")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering (Holtzman et al., 2020).

    Per row: sort probabilities in descending order, keep the smallest prefix whose
    cumulative probability reaches p, set every other logit to -inf. The most likely
    token must always survive (even when its probability alone exceeds p).
    Hint: torch.sort, torch.softmax, torch.cumsum, then scatter the mask back to the
    original token order.
    """
    raise NotImplementedError("TODO(student): top_p_filter")


# ---------------------------------------------------------------- provided


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """(B, V) logits -> (B,) sampled token ids. temperature == 0 means greedy."""
    logits = logits.float()
    if temperature == 0:
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if top_k and top_k > 0:
        logits = top_k_filter(logits, int(top_k))
    if top_p is not None and top_p < 1.0:
        logits = top_p_filter(logits, float(top_p))
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
