"""Diffusion composer: piano-roll encoding, U-Net, DDIM sampling with masks and inpainting,
dataset, training and editing.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from symusic import Note, Score, Tempo, TimeSignature, Track

from tunesmith.attributes import INSTRUMENTS, MusicAttributes
from tunesmith.data.prepare import prepare
from tunesmith.data.synthetic import generate_dataset
from tunesmith.diffusion.composer import DiffusionComposer
from tunesmith.diffusion.data import RollDataset, transpose_roll
from tunesmith.diffusion.roll import (RollSpec, allowed_mask, band_for, notes_to_roll, roll_to_notes, roll_to_score,
                                      score_to_roll, tokens_to_roll)
from tunesmith.diffusion.schedule import Diffusion
from tunesmith.diffusion.train import train
from tunesmith.diffusion.unet import RollUNet, UNetConfig
from tunesmith.paths import load_yaml
from tunesmith.tokenizer import TuneTokenizer

SPEC = RollSpec()
STEP = 120  # ticks per sixteenth at 480 ticks per quarter

PIANO = [(0, 4, 60), (0, 4, 64), (0, 4, 67), (8, 8, 62), (16, 2, 72), (18, 2, 72)]  # repeated note at 16 and 18
BASS = [(0, 8, 36), (8, 8, 43), (124, 4, 40)]                                       # last note ends with bar 8
DRUMS = [(0, 36), (2, 42), (4, 38), (8, 36), (12, 38)]


@pytest.fixture(scope="module")
def etok() -> TuneTokenizer:
    return TuneTokenizer({"mode": "ensemble"})


def make_score() -> Score:
    score = Score(480)
    for fam, notes in (("piano", PIANO), ("bass", BASS)):
        track = Track(name=fam, program=INSTRUMENTS[fam], is_drum=False)
        for f, n, p in notes:
            track.notes.append(Note(f * STEP, n * STEP, p, 80))
        score.tracks.append(track)
    drums = Track(name="drums", program=0, is_drum=True)
    for f, p in DRUMS:
        drums.notes.append(Note(f * STEP, STEP, p, 90))
    score.tracks.append(drums)
    score.time_signatures.append(TimeSignature(0, 4, 4))
    score.tempos.append(Tempo(0, 120.0))
    return score


def tiny_model(seed: int = 0) -> RollUNet:
    torch.manual_seed(seed)
    model = RollUNet(UNetConfig(in_channels=SPEC.channels, base=16, mults=[1, 2], n_res=1, attn_levels=[1],
                                n_heads=2, dropout=0.0, cond_vocab=504, n_fields=11, field_dim=8))
    with torch.no_grad():  # zero-initialised output layers would hide the conditioning
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return model.eval()


def test_notes_roll_round_trip():
    roll = notes_to_roll([PIANO, BASS], DRUMS, SPEC)
    assert roll.shape == (9, 128, 96) and roll.dtype == np.uint8
    parts, drums = roll_to_notes(roll.astype(np.float32), SPEC)
    assert parts[0] == sorted(PIANO) and parts[1] == sorted(BASS) and parts[2] == parts[3] == []
    assert drums == sorted(DRUMS)


def test_tokens_to_roll_matches_notes(etok):
    attrs = MusicAttributes(instruments=["piano", "bass"], drums=True, meter="4/4", bars=8)
    ids = etok.encode(make_score(), attrs)
    roll = tokens_to_roll(ids, etok, SPEC)
    assert roll is not None
    assert np.array_equal(roll, notes_to_roll([PIANO, BASS], DRUMS, SPEC))
    from_score, families, has_drums = score_to_roll(make_score(), SPEC)
    assert families == ["piano", "bass"] and has_drums and np.array_equal(from_score, roll)
    three_four = etok.encode(make_score(), attrs)
    three_four[etok.prompt_len + 1] = etok.timesig_id("3/4")
    assert tokens_to_roll(three_four, etok, SPEC) is None


def test_roll_to_score():
    roll = notes_to_roll([PIANO, BASS], DRUMS, SPEC).astype(np.float32)
    score = roll_to_score(roll, ["piano", "bass"], SPEC, qpm=90)
    assert [t.name for t in score.tracks] == ["piano", "bass", "drums"]
    assert [int(t.program) for t in score.tracks[:2]] == [INSTRUMENTS["piano"], INSTRUMENTS["bass"]]
    assert score.tracks[2].is_drum and len(score.tracks[2].notes) == len(DRUMS)
    piano = sorted((n.time // STEP, n.duration // STEP, n.pitch) for n in score.tracks[0].notes)
    assert piano == sorted(PIANO)
    assert score.tempos[0].qpm == pytest.approx(90) and score.time_signatures[0].numerator == 4


def test_transpose_and_masks():
    roll = notes_to_roll([PIANO, BASS], DRUMS, SPEC)
    up = transpose_roll(roll, 2, SPEC)
    parts, drums = roll_to_notes(up.astype(np.float32), SPEC)
    assert parts[0] == sorted((f, n, p + 2) for f, n, p in PIANO) and drums == sorted(DRUMS)
    high = notes_to_roll([[(0, 1, 114)]], [], SPEC)
    assert transpose_roll(high, 3, SPEC) is None
    mask = allowed_mask(["piano", "bass"], False, SPEC)
    assert mask.shape == (9, 1, 96)
    assert mask[0].any() and mask[2].any() and not mask[4:].any()  # slots 3-4 and drums off
    assert not mask[2, 0, SPEC.row(70)] and mask[2, 0, SPEC.row(40)]  # bass range
    assert allowed_mask(["piano"], None, SPEC)[SPEC.drum_channel].any()
    assert band_for(MusicAttributes(genre="jazz")) == ["piano", "sax", "bass"]
    assert band_for(MusicAttributes(instruments=["flute", "harp"])) == ["flute", "harp"]


def test_unet_shapes_and_conditioning():
    model = tiny_model()
    x = torch.randn(2, *SPEC.shape)
    t = torch.tensor([5, 500])
    a = torch.randint(0, 504, (2, 11))
    out = model(x, t, a)
    assert out.shape == x.shape
    b = a.clone()
    b[:, 3] = (b[:, 3] + 1) % 504
    assert not torch.allclose(out, model(x, t, b))


def test_sampling_respects_mask_and_inpainting():
    model, diff = tiny_model(), Diffusion(50)
    cond = torch.randint(0, 504, (2, 11))
    allowed = torch.from_numpy(allowed_mask(["piano"], False, SPEC))
    known = torch.from_numpy(notes_to_roll([PIANO], [], SPEC)).float()[None].repeat(2, 1, 1, 1) * 2 - 1
    keep = torch.zeros(2, *SPEC.shape, dtype=torch.bool)
    keep[:, :, 16:32] = True  # keep bar 2
    x0 = diff.sample(model, cond, cond.clone(), (2, *SPEC.shape), steps=5, cfg_scale=2.0, allowed=allowed,
                     known=known, known_mask=keep, generator=torch.Generator().manual_seed(0))
    assert x0.shape == (2, *SPEC.shape) and x0.min() >= -1 and x0.max() <= 1
    free = ~keep & ~allowed.expand_as(keep)
    assert (x0[free] == -1).all()                  # forbidden cells stay off
    assert torch.equal(x0[keep], known[keep])      # kept cells are exactly the given roll
    gen = lambda: torch.Generator().manual_seed(3)  # noqa: E731
    again = [diff.sample(model, cond, None, (2, *SPEC.shape), steps=3, generator=gen()) for _ in range(2)]
    assert torch.equal(*again)


def test_dataset_training_and_editing(tmp_path):
    meta = generate_dataset(tmp_path / "raw", n=40, seed=5)
    cfg = load_yaml("configs/data_ensemble.yaml")
    cfg.update(metadata=str(meta), midi_root=str(tmp_path / "raw"), out_dir=str(tmp_path / "proc"),
               workers=1, val_fraction=0.25, min_notes_per_window=8)
    info = prepare(cfg)
    tok = TuneTokenizer(info["tokenizer"])
    ds = RollDataset(tmp_path / "proc", "train", tok, SPEC,
                     {"transpose_range": [-2, 2], "transpose_prob": 1.0, "attr_dropout": 0.0, "uncond_prob": 1.0})
    assert len(ds) > 0
    roll, cond = ds[0]
    assert roll.shape == SPEC.shape and roll.dtype == torch.uint8 and int(roll.sum()) > 0
    assert all(tok.tokens[i].endswith("_NULL") for i in cond.tolist())

    tcfg = load_yaml("configs/diffusion_tiny.yaml")
    tcfg.update(data_dir=str(tmp_path / "proc"), out_dir=str(tmp_path / "model"))
    summary = train(tcfg, device_name="cpu")
    assert summary["steps"] == 4 and (tmp_path / "model" / "best.pt").exists()
    assert list((tmp_path / "model" / "samples").glob("*.mid"))

    composer = DiffusionComposer.load(tmp_path / "model" / "best.pt", "cpu")
    [score] = composer.compose([MusicAttributes(instruments=["piano", "bass"], drums=False, tempo_bpm=100)], seed=1)
    assert {t.name for t in score.tracks} <= {"piano", "bass"}
    source = make_score()
    edited = composer.edit(source, bars=[2], seed=1)
    before, fam_before, _ = score_to_roll(source, SPEC)
    after, fam_after, _ = score_to_roll(edited, SPEC)
    keep = np.r_[0:32, 48:128]  # every bar except bar 3
    for fam in ("piano", "bass"):  # slots are ordered by note count, so look parts up by name
        on_before = before[2 * fam_before.index(fam)][keep]
        on_after = after[2 * fam_after.index(fam)][keep]
        assert np.array_equal(on_before, on_after)
    assert np.array_equal(before[SPEC.drum_channel][keep], after[SPEC.drum_channel][keep])
