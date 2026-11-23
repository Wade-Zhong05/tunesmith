"""Ensemble (multi-instrument) mode: attributes, tokenizer, windows, grammar, metrics, rendering.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from tunesmith.attributes import MusicAttributes, control_fields, from_midicaps
from tunesmith.data.dataset import TokenDataset
from tunesmith.data.prepare import prepare
from tunesmith.data.synthetic import generate_dataset, make_piece
from tunesmith.eval.baselines import UniformComposer
from tunesmith.eval.metrics import music_metrics
from tunesmith.model.generate import FAMILY_RANGE, generate
from tunesmith.music.ensemble import cap_polyphony, ensemble_windows
from tunesmith.paths import load_yaml
from tunesmith.render.render import render_wav
from tunesmith.tokenizer import T_BAR, T_PITCH, T_PITCHDRUM, T_PROGRAM, T_TIMESIG, TuneTokenizer


@pytest.fixture(scope="module")
def etok() -> TuneTokenizer:
    return TuneTokenizer({"mode": "ensemble"})


@pytest.fixture(scope="module")
def excerpt():
    score, _ = make_piece(random.Random(3))
    return ensemble_windows(score, [8], min_notes=8, rng=random.Random(0))[0]


def test_melody_vocabulary_is_unchanged():
    tok = TuneTokenizer()
    assert tok.vocab_size == 421 and tok.prompt_len == 9 and tok.fields == control_fields("melody")


def test_instrument_attributes():
    a = MusicAttributes.lenient({"instruments": ["Piano", "strings", "kazoo", "piano", "bass", "flute", "cello"],
                                 "drums": "yes"})
    assert a.instruments == ["piano", "strings", "bass", "flute"] and a.drums is True and a.instrument == "piano"
    assert a.field_value("inst2") == "strings" and a.field_value("drums") == "yes"
    assert MusicAttributes(instrument="cello").field_value("inst1") == "cello"
    merged = a.merged({"instruments": ["violin"]})
    assert merged.instruments == ["violin"] and merged.instrument == "violin"
    row = {"instrument_numbers_sorted": "[48, 56, 128, 33, 48]"}
    m = from_midicaps(row)
    assert m.instruments == ["strings", "trumpet", "bass"] and m.drums is True and m.instrument == "strings"
    assert from_midicaps({"instrument_numbers_sorted": "[0, 25]"}).drums is False


def test_ensemble_tokenizer_round_trip(etok, excerpt):
    a = MusicAttributes(meter="3/4", bars=8, instruments=excerpt.families, drums=True)
    ids = etok.encode(excerpt.to_score(), a)
    assert len(ids[:etok.prompt_len]) == 13
    back = etok.attrs_from_prompt(ids[:etok.prompt_len])
    assert back.instruments == excerpt.families and back.drums is True
    score = etok.decode_for(ids, a)
    original = excerpt.to_score()
    assert [(t.name, t.program, t.is_drum, len(t.notes)) for t in score.tracks] == \
           [(t.name, t.program, t.is_drum, len(t.notes)) for t in original.tracks]
    for got, want in zip(score.tracks, original.tracks):  # compare in beats (tick resolutions differ)
        assert [(round(n.time / score.tpq * 4), n.pitch) for n in got.notes] == \
               [(round(n.time / original.tpq * 4), n.pitch) for n in want.notes]


def test_drums_are_not_transposed(etok):
    table = etok.transpose_table(3)
    drum = next(i for i in range(etok.vocab_size) if etok.types[i] == T_PITCHDRUM)
    pitch = etok.token_to_id["Pitch_60"]
    assert table[drum] == drum and etok.tokens[table[pitch]] == "Pitch_63"


def test_cap_polyphony():
    chord = [(0, 480, p, 80) for p in (48, 52, 55, 60, 64, 67, 72)]
    assert [n[2] for n in cap_polyphony(chord, 3)] == [72, 67, 64]
    assert [n[2] for n in cap_polyphony(chord, 2, keep_low=True)] == [48, 52]
    assert len(cap_polyphony(chord + chord, 10)) == 7  # duplicates removed


def test_grammar_respects_instruments_and_drums(etok):
    model = UniformComposer(etok.vocab_size)
    attrs = [MusicAttributes(meter="4/4", bars=8, instruments=["piano", "strings", "bass"], drums=True),
             MusicAttributes(meter="3/4", bars=8, instruments=["flute"], drums=False),
             MusicAttributes(bars=8)]
    for a, seq in zip(attrs, generate(model, etok, attrs, seed=0, key_bias=2.0)):
        assert sum(etok.types[i] == T_BAR for i in seq) == 8
        programs = [int(etok.program[i]) for i in seq if etok.types[i] == T_PROGRAM]
        score = etok.decode_for(seq, a)
        families = {t.name for t in score.tracks if not t.is_drum and len(t.notes)}
        has_drums = any(t.is_drum and len(t.notes) for t in score.tracks)
        if a.instruments:
            assert families == set(a.instruments)  # every requested family plays, nothing else
        else:
            assert 1 <= len(families) <= 4         # unspecified: at most four pitched families
        if a.drums is not None:
            assert has_drums == a.drums
        if a.meter:
            assert {etok.tokens[i] for i in seq if etok.types[i] == T_TIMESIG} == {f"TimeSig_{a.meter}"}
        # notes follow their program: PitchDrum only after the drum program, pitched ranges respected
        for prev, cur in zip(seq, seq[1:]):
            if etok.types[cur] == T_PITCHDRUM:
                assert etok.program[prev] == -1
            if etok.types[cur] == T_PITCH:
                assert etok.program[prev] >= 0
        bass = [n.pitch for t in score.tracks if t.name == "bass" for n in t.notes]
        if bass:
            lo, hi = FAMILY_RANGE["bass"]
            assert lo <= min(bass) and max(bass) <= hi
        onsets = Counter((t.name, n.time) for t in score.tracks for n in t.notes)
        assert max(onsets.values()) <= 6
        assert programs


def test_prepare_ensemble_and_dataset(tmp_path):
    meta = generate_dataset(tmp_path / "raw", n=30, seed=4)
    cfg = load_yaml("configs/data_ensemble.yaml")
    cfg.update(metadata=str(meta), midi_root=str(tmp_path / "raw"), out_dir=str(tmp_path / "proc"),
               workers=1, val_fraction=0.2, min_notes_per_window=8)
    info = prepare(cfg)
    assert info["windows"]["train"] > 0 and info["tokenizer"]["mode"] == "ensemble"
    tok = TuneTokenizer(info["tokenizer"])
    ds = TokenDataset(tmp_path / "proc", "train", tok, {"transpose_range": [-2, 2], "transpose_prob": 1.0,
                                                        "attr_dropout": 0.0, "uncond_prob": 1.0})
    seq = ds[0].numpy()
    assert all(tok.tokens[i].endswith("_NULL") for i in seq[1:tok.prompt_len - 1])
    attrs = tok.attrs_from_prompt(ds.raw(0)[:tok.prompt_len].tolist())
    assert attrs.instruments and attrs.drums is not None and attrs.bars == 8


def test_ensemble_metrics_and_render(etok, excerpt, tmp_path):
    score = excerpt.to_score()
    a = MusicAttributes(instruments=["piano", *excerpt.families], drums=True, meter="3/4", bars=8)
    m = music_metrics(score, a, 8, ensemble=True)
    assert m["instrument_recall"] == pytest.approx(len(excerpt.families) / (len(excerpt.families) + 1))
    assert m["instrument_precision"] == 1.0 and m["drums_match"] is True
    assert m["n_tracks"] == len(excerpt.families) and m["polyphony"] >= 1.0
    assert render_wav(score, tmp_path / "e.wav", soundfont=None) == "numpy"
    assert (tmp_path / "e.wav").stat().st_size > 10000
