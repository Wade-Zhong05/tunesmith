"""Grammar-constrained generation with the baseline models and the GRU.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import torch

from tunesmith.attributes import MusicAttributes
from tunesmith.eval.baselines import BigramComposer, UniformComposer
from tunesmith.eval.metrics import music_metrics
from tunesmith.model import build_model
from tunesmith.model.generate import generate
from tunesmith.tokenizer import T_BAR, T_PITCH, T_TIMESIG


def test_uniform_composer_respects_grammar(tok):
    model = UniformComposer(tok.vocab_size)
    attrs = [MusicAttributes(meter="3/4", bars=8, key="E minor"), MusicAttributes(meter="6/8", bars=4),
             MusicAttributes(bars=16)]
    seqs = generate(model, tok, attrs, seed=0, key_bias=2.0)
    for a, s in zip(attrs, seqs):
        assert s[-1] == tok.eos_id
        assert sum(tok.types[i] == T_BAR for i in s) == a.bars
        sigs = {tok.tokens[i] for i in s if tok.types[i] == T_TIMESIG}
        if a.meter:
            assert sigs == {f"TimeSig_{a.meter}"}
        else:
            assert len(sigs) == 1 and sigs <= {"TimeSig_4/4", "TimeSig_3/4", "TimeSig_2/4", "TimeSig_6/8"}
        score = tok.decode_for(s, a)
        m = music_metrics(score, a, tok.count_bars(s))
        assert m["bars"] == a.bars and m.get("meter_match", True)
        ts = score.time_signatures[0]
        bar = score.tpq * 4 * ts.numerator // ts.denominator
        assert max(n.time + n.duration for n in score.tracks[0].notes) <= bar * a.bars  # clipped at the last bar


def test_generation_is_reproducible(tok):
    model = UniformComposer(tok.vocab_size)
    a = [MusicAttributes(bars=4)]
    assert generate(model, tok, a, seed=3) == generate(model, tok, a, seed=3)


def test_gru_with_cache_and_cfg(tok):
    torch.manual_seed(0)
    model = build_model({"type": "gru", "d_model": 32, "n_layers": 2, "dropout": 0.0, "max_seq_len": 512}, tok.vocab_size)
    attrs = [MusicAttributes(key="G major", meter="4/4", bars=4, instrument="piano"), MusicAttributes(bars=2)]
    seqs = generate(model, tok, attrs, cfg_scale=2.0, seed=1)
    for a, s in zip(attrs, seqs):
        assert sum(tok.types[i] == T_BAR for i in s) == a.bars or len(s) >= 511


def test_bigram_fit(tok, synth_data):
    _, processed = synth_data
    model = BigramComposer.fit(processed, tok)
    assert model.table.shape == (tok.vocab_size, tok.vocab_size)
    probs = model.table.exp().sum(-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)
    a = MusicAttributes(bars=4)
    assert a.bars == 8  # bar counts snap to the nearest of 8 / 16 / 32
    seqs = generate(model, tok, [a], seed=0)
    assert sum(tok.types[i] == T_BAR for i in seqs[0]) == 8


class _RestLover(UniformComposer):
    """A model that always wants to close the bar, i.e. writes silent bars if it may."""

    def __init__(self, tok):
        super().__init__(tok.vocab_size)
        self.bias = torch.zeros(tok.vocab_size)
        self.bias[torch.from_numpy(tok.types == T_BAR)] = 50.0

    def forward(self, idx, past_kv=None, use_cache=False):
        return self.bias.expand(*idx.shape, -1).clone(), None


def _notes_per_bar(tok, seq):
    counts = []
    for i in seq:
        if tok.types[i] == T_BAR:
            counts.append(0)
        elif tok.types[i] == T_PITCH and counts:
            counts[-1] += 1
    return counts


def test_grammar_limits_silent_bars(tok):
    model = _RestLover(tok)
    attrs = [MusicAttributes(meter="4/4", bars=16)]
    limited = _notes_per_bar(tok, generate(model, tok, attrs, seed=0)[0])
    assert len(limited) == 16 and limited[0] > 0  # the piece starts in bar 1
    assert all(a > 0 or b > 0 for a, b in zip(limited, limited[1:]))  # never two silent bars in a row
    free = _notes_per_bar(tok, generate(model, tok, attrs, seed=0, max_empty_bars=None)[0])
    assert len(free) == 16 and free[:-1] == [0] * 15  # without the limit the model only rests
