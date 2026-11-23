"""Reference sampling filters: temperature, top-k and nucleus (top-p).

Used by the generation loop only while the student's own versions in
`model/sampling_filters.py` still raise NotImplementedError; the student file is untouched
and takes over automatically once it is implemented. `tests/test_sampling_filters.py`
checks both.

AI attribution: written by Claude Code (Claude Opus 5.5) on 2026-11-23 at the student's
request, because plain sampling was making generated melodies sound erratic. See
ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return logits / float(temperature)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    remove = (cum - probs) >= p          # tokens whose preceding mass already reaches p
    remove[..., 0] = False               # always keep the most likely token
    mask = torch.zeros_like(remove).scatter(-1, order, remove)
    return logits.masked_fill(mask, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Same contract as `sampling_filters.sample_next`."""
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
