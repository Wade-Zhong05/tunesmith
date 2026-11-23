"""Attributes, tokenizer, melody reduction, key finding, dataset augmentation.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tunesmith.attributes import (MusicAttributes, from_midicaps, normalize_key, program_to_family,
                                  tempo_bin, transpose_key)
from tunesmith.data.dataset import IGNORE, TokenDataset, collate
from tunesmith.music.keyfinding import detect_key, relative_key
from tunesmith.music.melody import Excerpt, skyline
from tunesmith.tokenizer import T_PITCH, T_POSITION


@pytest.mark.parametrize("text,expected", [
    ("F# minor", "F# minor"), ("f#m", "F# minor"), ("Gb major", "F# major"), ("Bb", "Bb major"),
    ("A# minor", "Bb minor"), ("C sharp major", "C# major"), ("Eb", "Eb major"), ("H major", None), ("", None),
])
def test_normalize_key(text, expected):
    assert normalize_key(text) == expected


def test_attribute_helpers():
    assert tempo_bin(59) == "very_slow" and tempo_bin(135) == "fast" and tempo_bin(250) == "presto"
    assert transpose_key("A minor", 3) == "C minor"
    assert relative_key("C major") == "A minor" and relative_key("A minor") == "C major"
    assert program_to_family(0) == "piano" and program_to_family(73) == "flute" and program_to_family(127) is None
    a = MusicAttributes.lenient({"key": "nonsense", "tempo_bpm": "95", "meter": "5/4", "mood": "SAD", "bars": 12})
    assert a.key is None and a.tempo_bpm == 95 and a.meter == "other" and a.mood == "sad" and a.bars == 8


def test_from_midicaps_row():
    row = {"key": "F# minor", "tempo": "135", "time_signature": "4/4", "instrument_numbers_sorted": "[128, 48, 56]",
           "mood": "['melodic', 'film', 'epic']", "genre": ["electronic", "classical"]}
    a = from_midicaps(row)
    assert (a.key, a.tempo_bpm, a.meter, a.instrument, a.mood, a.genre) == (
        "F# minor", 135, "4/4", "strings", "epic", "electronic")


def test_unusual_meter_becomes_other():
    assert from_midicaps({"time_signature": "5/4"}).meter == "other"
    assert MusicAttributes(meter="other").meter == "other"
    assert MusicAttributes.lenient({"meter": "7/8"}).meter == "other"


def test_control_tokens_round_trip(tok):
    a = MusicAttributes(key="D major", tempo_bpm=70, meter="6/8", instrument="flute", mood="calm", bars=16)
    ids = tok.prompt_ids(a)
    assert len(ids) == tok.prompt_len
    back = tok.attrs_from_prompt(ids)
    assert (back.key, back.meter, back.instrument, back.mood, back.bars) == ("D major", "6/8", "flute", "calm", 16)
    assert tok.attrs_from_prompt(tok.null_prompt_ids()) == MusicAttributes()


@pytest.mark.parametrize("num,den", [(4, 4), (3, 4), (2, 4), (6, 8)])
def test_tokenizer_round_trip(tok, num, den):
    tpq = 480
    step = tpq // 2
    n = int(4 * num * 4 / den) * 2
    notes = [(i * step, step - 20, 60 + (i * 5) % 12, 80) for i in range(n)]
    ex = Excerpt(tpq, num, den, 90.0, 0, notes, 4)
    attrs = MusicAttributes(key="C major", meter=f"{num}/{den}", bars=4)
    ids = tok.encode(ex.to_score(), attrs)
    assert ids[0] == tok.bos_id and ids[tok.prompt_len - 1] == tok.sep_id and ids[-1] == tok.eos_id
    positions = [int(tok.position[i]) for i in ids if tok.types[i] == T_POSITION]
    assert max(positions) < tok.positions_per_bar(num, den)
    score = tok.decode_for(ids, attrs)
    got = [(round(x.time / score.tpq * 2), x.pitch) for x in score.tracks[0].notes]
    assert got == [(i, 60 + (i * 5) % 12) for i in range(n)]
    assert (score.time_signatures[0].numerator, score.time_signatures[0].denominator) == (num, den)


def test_transpose_table(tok):
    t = tok.transpose_table(2)
    assert tok.tokens[t[tok.token_to_id["Pitch_60"]]] == "Pitch_62"
    assert t[tok.token_to_id["Pitch_108"]] == -1
    assert tok.tokens[t[tok.token_to_id["KEY_Bb_major"]]] == "KEY_C_major"
    assert t[tok.token_to_id["KEY_NULL"]] == tok.token_to_id["KEY_NULL"]
    assert t[tok.token_to_id["Bar_None"]] == tok.token_to_id["Bar_None"]


def test_skyline_is_monophonic():
    notes = [(0, 480, 60, 80), (0, 480, 67, 80), (240, 480, 64, 80), (480, 240, 72, 80)]
    out = skyline(notes)
    assert [n[2] for n in out] == [67, 64, 72]
    assert all(a[0] + a[1] <= b[0] for a, b in zip(out, out[1:]))


def test_detect_key():
    c_major = [(i, 1, p, 1) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 67, 64, 60])]
    assert detect_key(c_major)[0] == "C major"
    a_minor = [(i, 2 if p % 12 == 9 else 1, p, 1) for i, p in enumerate([57, 59, 60, 62, 64, 65, 68, 69, 64, 60, 57])]
    assert detect_key(a_minor)[0] == "A minor"
    assert detect_key([]) == (None, 0.0)


def test_dataset_and_collate(tok, synth_data):
    _, processed = synth_data
    ds = TokenDataset(processed, "train", tok, {"transpose_range": [-5, 6], "transpose_prob": 1.0,
                                                 "attr_dropout": 0.0, "uncond_prob": 0.0})
    assert len(ds) > 10
    raw = ds.raw(0)
    key_before = tok.tokens[raw[1]]
    for _ in range(5):
        aug = ds[0].numpy()
        shift = [int(tok.pitch[b]) - int(tok.pitch[a]) for a, b in zip(raw, aug) if tok.types[a] == T_PITCH]
        if shift and shift[0] != 0:
            assert len(set(shift)) == 1  # every pitch moved by the same interval
            k = shift[0]
            if key_before != "KEY_NULL":
                from tunesmith.attributes import transpose_key as tk
                assert tok.tokens[aug[1]] == "KEY_" + tk(key_before[4:].replace("_", " "), k).replace(" ", "_")
    x, y = collate([ds[0], ds[1]], tok.pad_id, tok.prompt_len)
    assert x.shape == y.shape and x.dtype == torch.long
    assert (y[:, : tok.prompt_len - 1] == IGNORE).all()
    assert (y[:, tok.prompt_len - 1] != IGNORE).all()  # first music token is predicted


def test_attribute_dropout(tok, synth_data):
    _, processed = synth_data
    ds = TokenDataset(processed, "train", tok, {"transpose_range": [0, 0], "transpose_prob": 0.0,
                                                 "attr_dropout": 0.0, "uncond_prob": 1.0})
    seq = ds[0].numpy()
    assert all(tok.tokens[i].endswith("_NULL") for i in seq[1:tok.prompt_len - 1])
    assert np.array_equal(seq[tok.prompt_len - 1:], ds.raw(0)[tok.prompt_len - 1:])


def test_length_bucket_sampler():
    from tunesmith.train.train_composer import LengthBucketSampler

    lengths = np.random.default_rng(0).integers(10, 1000, size=1000)
    s = LengthBucketSampler(lengths, batch_size=8, bucket=10, seed=0)
    batches = list(iter(s))
    flat = [i for b in batches for i in b]
    assert len(batches) == len(s) == 125 and sorted(flat) == list(range(1000))
    padded = sum(len(b) * max(lengths[b]) for b in batches)
    assert lengths.sum() / padded > 0.8  # far less padding than random batches
    assert list(iter(s)) != batches       # a new order every epoch


def test_dataset_pickles_without_token_data(tok, synth_data):
    import pickle

    _, processed = synth_data
    ds = TokenDataset(processed, "train", tok, None)
    blob = pickle.dumps(ds)
    assert ds.__getstate__()["tokens"] is None  # workers reopen the memmap instead of copying the file
    clone = pickle.loads(blob)
    assert np.array_equal(clone.raw(0), ds.raw(0))
