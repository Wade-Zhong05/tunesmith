"""Checks for the student-implemented Transformer (src/tunesmith/model/transformer.py).

Every test is skipped while the code still raises NotImplementedError, so the suite stays
green until you start; then each test checks one property of a correct implementation.

    pixi run pytest tests/test_transformer.py -q

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import pytest
import torch

from tunesmith.model.transformer import CausalSelfAttention, Composer, ComposerConfig

V = 50


def cfg(**kw) -> ComposerConfig:
    base = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, max_seq_len=64)
    base.update(kw)
    return ComposerConfig(**base)


def todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def make(**kw) -> Composer:
    torch.manual_seed(0)
    model = todo(Composer, cfg(**kw))
    model.eval()
    return model


def test_attention_shape():
    attn = todo(CausalSelfAttention, cfg())
    x = torch.randn(2, 7, 32)
    y, present = todo(attn, x)
    assert y.shape == (2, 7, 32)
    assert present is None


def test_forward_shapes():
    model = make()
    logits, present = todo(model, torch.randint(0, V, (3, 10)))
    assert logits.shape == (3, 10, V)
    assert present is None


def test_weight_tying():
    model = make()
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_causality():
    model = make()
    a = torch.randint(0, V, (1, 12))
    b = a.clone()
    b[0, 6] = (a[0, 6] + 1) % V
    la, _ = todo(model, a)
    lb, _ = todo(model, b)
    assert torch.allclose(la[:, :6], lb[:, :6], atol=1e-5), "positions before 6 must not see token 6"
    assert not torch.allclose(la[:, 6:], lb[:, 6:], atol=1e-5)


def test_max_seq_len_is_enforced():
    model = make()
    with pytest.raises(ValueError):
        todo(model, torch.randint(0, V, (1, 65)))


def test_kv_cache_matches_full_forward():
    model = make()
    idx = torch.randint(0, V, (2, 9))
    full, _ = todo(model, idx)
    logits, cache = todo(model, idx[:, :4], use_cache=True)
    if cache is None:
        pytest.skip("KV cache not implemented (optional extension)")
    steps = [logits]
    for t in range(4, 9):
        logits, cache = model(idx[:, t:t + 1], past_kv=cache, use_cache=True)
        steps.append(logits)
    inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), "cached decoding must reproduce the full forward pass"


def test_rope_option_if_implemented():
    try:
        model = make(positional="rope")
    except (ValueError, KeyError, AssertionError):
        pytest.skip("RoPE not implemented (optional extension)")
    logits, _ = todo(model, torch.randint(0, V, (1, 8)))
    assert logits.shape == (1, 8, V)


def test_can_overfit_one_sequence():
    torch.manual_seed(0)
    model = todo(Composer, cfg())
    model.train()
    seq = torch.randint(0, V, (1, 20))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(150):
        logits, _ = model(seq[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, f"loss {loss.item():.3f}: gradients may not reach every layer"


def test_base_config_size():
    model = todo(Composer, ComposerConfig(vocab_size=421, d_model=512, n_layers=8, n_heads=8, d_ff=2048, max_seq_len=1024))
    n = model.num_parameters()
    assert 20e6 < n < 35e6, f"{n / 1e6:.1f}M parameters; expected about 25M for configs/composer_base.yaml"
