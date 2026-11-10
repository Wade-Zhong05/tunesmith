"""Checks for the student-implemented sampling filters (src/tunesmith/model/sampling_filters.py).

Skipped while a function still raises NotImplementedError.

    pixi run pytest tests/test_sampling_filters.py -q

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math

import pytest
import torch

from tunesmith.model import sampling_filters as sf

NEG = float("-inf")


def todo(fn, *args):
    try:
        return fn(*args)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def test_temperature():
    out = todo(sf.apply_temperature, torch.tensor([[1.0, 2.0, 3.0]]), 2.0)
    assert torch.allclose(out, torch.tensor([[0.5, 1.0, 1.5]]))


def test_top_k_keeps_k_largest_per_row():
    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0], [0.0, -1.0, 5.0, 2.0]])
    out = todo(sf.top_k_filter, logits, 2)
    kept = torch.isfinite(out)
    assert kept.tolist() == [[False, True, False, True], [False, False, True, True]]
    assert torch.equal(out[kept], logits[kept])
    assert torch.equal(todo(sf.top_k_filter, logits, 10), logits)


def test_top_p_smallest_nucleus():
    probs = torch.tensor([[0.15, 0.5, 0.05, 0.3]])
    logits = probs.log()
    out = todo(sf.top_p_filter, logits, 0.7)
    assert torch.isfinite(out).tolist() == [[False, True, False, True]]  # 0.5 + 0.3 >= 0.7
    out = todo(sf.top_p_filter, logits, 0.1)
    assert torch.isfinite(out).tolist() == [[False, True, False, False]]  # the top token always survives
    out = todo(sf.top_p_filter, logits, 0.99)
    assert torch.isfinite(out).all()


def test_filters_keep_masked_tokens_masked():
    logits = torch.tensor([[NEG, 1.0, 2.0, NEG, 0.5]])
    for out in (todo(sf.top_k_filter, logits, 4), todo(sf.top_p_filter, logits, 0.999)):
        assert math.isinf(out[0, 0].item()) and math.isinf(out[0, 3].item())


def test_sample_next_uses_filters():
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 3.0, 2.9, -5.0]]).repeat(200, 1)
    try:
        picks = sf.sample_next(logits, temperature=0.7, top_k=2, top_p=0.9)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
    assert set(picks.tolist()) <= {1, 2}
