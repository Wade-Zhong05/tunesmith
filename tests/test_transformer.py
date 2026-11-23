"""Checks for the Transformer composer, run against both implementations:

- `student`:   src/tunesmith/model/transformer.py (the student's own version; tests are
               skipped while it still raises NotImplementedError)
- `reference`: src/tunesmith/model/transformer_ref.py (AI-written reference)

    pixi run pytest tests/test_transformer.py -q -rs
    pixi run pytest tests/test_transformer.py -q -k student     # only your version

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; parametrised over
both implementations on 2026-11-14. See ATTRIBUTION.md.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from tunesmith.model.transformer import ComposerConfig

V = 50
IMPLS = {"student": "tunesmith.model.transformer", "reference": "tunesmith.model.transformer_ref"}


@pytest.fixture(params=list(IMPLS), ids=list(IMPLS))
def impl(request):
    return importlib.import_module(IMPLS[request.param])


def cfg(**kw) -> ComposerConfig:
    base = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, d_ff=64, dropout=0.0, max_seq_len=64)
    base.update(kw)
    return ComposerConfig(**base)


def todo(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


def make(impl, **kw):
    torch.manual_seed(0)
    model = todo(impl.Composer, cfg(**kw))
    model.eval()
    return model


def test_attention_shape(impl):
    attn = todo(impl.CausalSelfAttention, cfg())
    x = torch.randn(2, 7, 32)
    y, present = todo(attn, x)
    assert y.shape == (2, 7, 32)
    assert present is None


def test_forward_shapes(impl):
    model = make(impl)
    logits, present = todo(model, torch.randint(0, V, (3, 10)))
    assert logits.shape == (3, 10, V)
    assert present is None


def test_weight_tying(impl):
    model = make(impl)
    assert model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr()


def test_causality(impl):
    model = make(impl)
    a = torch.randint(0, V, (1, 12))
    b = a.clone()
    b[0, 6] = (a[0, 6] + 1) % V
    la, _ = todo(model, a)
    lb, _ = todo(model, b)
    assert torch.allclose(la[:, :6], lb[:, :6], atol=1e-5), "positions before 6 must not see token 6"
    assert not torch.allclose(la[:, 6:], lb[:, 6:], atol=1e-5)


def test_max_seq_len_is_enforced(impl):
    model = make(impl)
    with pytest.raises(ValueError):
        todo(model, torch.randint(0, V, (1, 65)))


def test_kv_cache_matches_full_forward(impl):
    model = make(impl)
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


def test_kv_cache_multi_token_chunk(impl):
    model = make(impl)
    idx = torch.randint(0, V, (1, 10))
    full, _ = todo(model, idx)
    _, cache = todo(model, idx[:, :4], use_cache=True)
    if cache is None:
        pytest.skip("KV cache not implemented (optional extension)")
    logits, _ = model(idx[:, 4:], past_kv=cache, use_cache=True)
    assert torch.allclose(full[:, 4:], logits, atol=1e-4), "a multi-token chunk after a cache must stay causal"


def test_rope_option_if_implemented(impl):
    try:
        model = make(impl, positional="rope")
    except (ValueError, KeyError, AssertionError):
        pytest.skip("RoPE not implemented (optional extension)")
    logits, _ = todo(model, torch.randint(0, V, (1, 8)))
    assert logits.shape == (1, 8, V)


def test_can_overfit_one_sequence(impl):
    torch.manual_seed(0)
    model = todo(impl.Composer, cfg())
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


def test_base_config_size(impl):
    model = todo(impl.Composer, ComposerConfig(vocab_size=421, d_model=512, n_layers=8, n_heads=8, d_ff=2048,
                                               max_seq_len=1024))
    n = model.num_parameters()
    assert 20e6 < n < 35e6, f"{n / 1e6:.1f}M parameters; expected about 25M for configs/composer_base.yaml"
