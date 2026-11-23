"""Multi-track piano rolls: the representation the diffusion composer generates.

An ensemble window (8 bars of 4/4) becomes a tensor of shape (channels, frames, pitches):

    frames   = bars x 4 beats x 4 steps   (sixteenth-note grid; 128 for 8 bars)
    pitches  = MIDI pitch - pitch_lo     (96 rows: 20..115 covers the tokenizer's 21..109
                                          and the drum notes 27..88)
    channels = 2 per instrument slot + 1 for drums
               slot s: channel 2s   = onset   (1 where a note starts)
                       channel 2s+1 = sustain (1 on the later frames the note still sounds)
               drums:  channel 2*n_slots = onset only (drum hits have no length)

Instrument slot s holds the family named by the prompt's INST{s+1} token, so the
condition tells the model which instrument each channel group plays. Onset and sustain
are separate so that a long note and repeated short notes on the same pitch differ.

Values are {0, 1}; the model works on 2x - 1 in [-1, 1]. Decoding thresholds at 0.5.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
from symusic import Note, Score, Tempo, TimeSignature, Track

from ..attributes import INSTRUMENTS, MusicAttributes
from ..model.generate import DEFAULT_RANGE, FAMILY_RANGE  # playable range per family, as in the token grammar
from ..music.ensemble import cap_polyphony, group_families
from ..tokenizer import (PROGRAM_TO_FAMILY, T_BAR, T_DURATION, T_EOS, T_PITCH, T_PITCHDRUM, T_POSITION,
                         T_PROGRAM, T_TIMESIG, TuneTokenizer, apply_dynamics)

TPQ = 480
DRUM_RANGE = (27, 88)   # the tokenizer's PitchDrum range


@dataclass(frozen=True)
class RollSpec:
    bars: int = 8
    beats_per_bar: int = 4
    steps_per_beat: int = 4
    n_slots: int = 4
    pitch_lo: int = 20
    n_pitch: int = 96

    @property
    def frames(self) -> int:
        return self.bars * self.beats_per_bar * self.steps_per_beat

    @property
    def channels(self) -> int:
        return 2 * self.n_slots + 1

    @property
    def drum_channel(self) -> int:
        return 2 * self.n_slots

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.channels, self.frames, self.n_pitch)

    def row(self, pitch: int) -> int:
        """Pitch row, or -1 outside the roll."""
        r = int(pitch) - self.pitch_lo
        return r if 0 <= r < self.n_pitch else -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "RollSpec":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------- notes <-> roll

PitchedNote = tuple[int, int, int]   # (start step, length in steps, MIDI pitch)
DrumHit = tuple[int, int]            # (step, MIDI drum note)


def notes_to_roll(parts: list[list[PitchedNote]], drums: list[DrumHit], spec: RollSpec) -> np.ndarray:
    """`parts[s]` are the notes of instrument slot s. Notes outside the roll are dropped and
    lengths are cut at the end of the roll."""
    roll = np.zeros(spec.shape, dtype=np.uint8)
    for s, notes in enumerate(parts[: spec.n_slots]):
        for start, length, pitch in notes:
            r = spec.row(pitch)
            if r < 0 or not 0 <= start < spec.frames:
                continue
            end = min(spec.frames, start + max(1, int(length)))
            roll[2 * s + 1, start + 1:end, r] = 1
            roll[2 * s, start, r] = 1
    for step, pitch in drums:
        r = spec.row(pitch)
        if r >= 0 and 0 <= step < spec.frames:
            roll[spec.drum_channel, step, r] = 1
    # a later onset on the same pitch ends the sustain of the previous note
    for s in range(spec.n_slots):
        roll[2 * s + 1][roll[2 * s] == 1] = 0
    return roll


def roll_to_notes(roll: np.ndarray, spec: RollSpec, threshold: float = 0.5
                  ) -> tuple[list[list[PitchedNote]], list[DrumHit]]:
    """Inverse of notes_to_roll for values in [0, 1]: a note starts at every onset cell and
    lasts while the following frames have sustain and no new onset. Sustain without an
    onset before it is ignored."""
    on_all = roll > threshold
    parts: list[list[PitchedNote]] = []
    for s in range(spec.n_slots):
        on, hold = on_all[2 * s], on_all[2 * s + 1]
        notes: list[PitchedNote] = []
        for f, r in zip(*np.nonzero(on)):
            end = f + 1
            while end < spec.frames and hold[end, r] and not on[end, r]:
                end += 1
            notes.append((int(f), int(end - f), int(r) + spec.pitch_lo))
        parts.append(sorted(notes))
    drums = [(int(f), int(r) + spec.pitch_lo) for f, r in zip(*np.nonzero(on_all[spec.drum_channel]))]
    return parts, sorted(drums)


# ---------------------------------------------------------------- tokens -> roll (training data)

class _Tables:
    """Python-list copies of the tokenizer tables (fast scalar lookups in the data loader)."""

    def __init__(self, tok: TuneTokenizer):
        self.types = tok.types.tolist()
        self.pitch = tok.pitch.tolist()
        self.drum_pitch = tok.drum_pitch.tolist()
        self.position = tok.position.tolist()
        self.program = tok.program.tolist()
        self.beats = [0.0] * tok.vocab_size
        for i in np.nonzero(tok.types == T_DURATION)[0]:
            a, b, c = tok.tokens[i].split("_", 1)[1].split(".")
            self.beats[i] = int(a) + int(b) / int(c)
        self.res = tok.res
        self.ts_44 = tok.timesig_id("4/4")


_TABLES: dict[int, _Tables] = {}


def _tables(tok: TuneTokenizer) -> _Tables:
    key = id(tok)
    if key not in _TABLES:
        _TABLES[key] = _Tables(tok)
    return _TABLES[key]


def tokens_to_roll(ids: list[int], tok: TuneTokenizer, spec: RollSpec) -> Optional[np.ndarray]:
    """Roll for one prepared ensemble window (prompt + REMI tokens). Instrument slots follow
    the prompt's INST1..INST4 order. Returns None unless the window is in 4/4."""
    tb = _tables(tok)
    attrs = tok.attrs_from_prompt(list(ids[: tok.prompt_len]))
    slot_of = {INSTRUMENTS[f]: s for s, f in enumerate((attrs.instruments or [])[: spec.n_slots])}
    parts: list[list[PitchedNote]] = [[] for _ in range(spec.n_slots)]
    drums: list[DrumHit] = []
    steps_per_bar = spec.beats_per_bar * spec.steps_per_beat
    bar, step, program, pending = -1, 0, None, None
    for i in ids[tok.prompt_len:]:
        t = tb.types[i]
        if t == T_BAR:
            bar += 1
            step = bar * steps_per_bar
            if bar >= spec.bars:
                break
        elif t == T_TIMESIG:
            if i != tb.ts_44:
                return None
        elif t == T_POSITION:
            step = bar * steps_per_bar + tb.position[i] * spec.steps_per_beat // tb.res
        elif t == T_PROGRAM:
            program = tb.program[i]
        elif t == T_PITCH:
            pending = tb.pitch[i]
        elif t == T_PITCHDRUM:
            drums.append((step, tb.drum_pitch[i]))
            pending = None
        elif t == T_DURATION:
            if pending is not None and program in slot_of:
                length = max(1, round(tb.beats[i] * spec.steps_per_beat))
                parts[slot_of[program]].append((step, length, pending))
            pending = None
        elif t == T_EOS:
            break
    return notes_to_roll(parts, drums, spec)


# ---------------------------------------------------------------- roll <-> score

def roll_to_score(roll: np.ndarray, families: list[str], spec: RollSpec, qpm: float = 100.0,
                  threshold: float = 0.5, max_poly: int = 6) -> Score:
    """Multi-track Score from a roll (values in [0, 1]); slot s is played by families[s].
    Velocities follow the ensemble playback dynamics (tokenizer.apply_dynamics)."""
    parts, drums = roll_to_notes(roll, spec, threshold)
    step = TPQ // spec.steps_per_beat
    score = Score(TPQ)
    for s, fam in enumerate(families[: spec.n_slots]):
        notes = [(f * step, n * step, p, 80) for f, n, p in parts[s]]
        notes = cap_polyphony(notes, max_poly, keep_low=fam == "bass")
        if not notes:
            continue
        track = Track(name=fam, program=INSTRUMENTS[fam], is_drum=False)
        for t, d, p, v in sorted(notes):
            track.notes.append(Note(int(t), int(d), int(p), int(v)))
        score.tracks.append(track)
    if drums:
        track = Track(name="drums", program=0, is_drum=True)
        for f, p in drums:
            track.notes.append(Note(int(f * step), int(step), int(p), 90))
        score.tracks.append(track)
    score.time_signatures.append(TimeSignature(0, spec.beats_per_bar, 4))
    score.tempos.append(Tempo(0, float(qpm)))
    apply_dynamics(score)
    return score


def score_to_roll(score: Score, spec: RollSpec, max_poly: int = 6) -> tuple[np.ndarray, list[str], bool]:
    """Roll for any multi-track MIDI score (for editing an existing piece): families are
    grouped as in the training data, the `n_slots` with most notes are kept, times are
    quantised to the sixteenth-note grid. Returns (roll, families by slot, has drums)."""
    fams, drum_notes = group_families(score, max_poly)
    families = [f for f, ns in sorted(fams.items(), key=lambda kv: -len(kv[1])) if ns][: spec.n_slots]
    q = score.tpq / spec.steps_per_beat

    def step(t: int) -> int:
        return int(round(t / q))

    parts = [[(step(t), max(1, step(t + d) - step(t)), p) for t, d, p, _ in fams[f]] for f in families]
    drums = [(step(t), p) for t, _, p, _ in drum_notes]
    return notes_to_roll(parts, drums, spec), families, bool(drum_notes)


# ---------------------------------------------------------------- generation helpers

# Instruments for requests that name none, by style (the composer needs to know what each
# channel group will be played by).
DEFAULT_BANDS = {
    "rock": ["electric_guitar", "bass"],
    "pop": ["piano", "electric_guitar", "bass"],
    "jazz": ["piano", "sax", "bass"],
    "electronic": ["synth_lead", "synth_pad", "bass"],
    "hiphop": ["electric_piano", "bass"],
    "classical": ["strings", "piano"],
    "soundtrack": ["strings", "brass", "piano"],
    "ambient": ["synth_pad", "piano"],
    "folk": ["guitar", "flute", "bass"],
    "world": ["ethnic", "guitar", "bass"],
}
DEFAULT_BAND = ["piano", "strings", "bass"]


def band_for(attrs: MusicAttributes, n_slots: int = 4) -> list[str]:
    """The instrument families (one per slot) a request will be written for."""
    if attrs.instruments:
        return list(attrs.instruments)[:n_slots]
    if attrs.instrument:
        return [attrs.instrument]
    return DEFAULT_BANDS.get(attrs.genre or "", DEFAULT_BAND)[:n_slots]


def allowed_mask(families: list[str], drums: Optional[bool], spec: RollSpec) -> np.ndarray:
    """Cells the composer may switch on, shape (channels, 1, pitches): channels of unused
    slots are off, every family stays within its playable range, drums only if not refused."""
    mask = np.zeros((spec.channels, 1, spec.n_pitch), dtype=bool)
    for s, fam in enumerate(families[: spec.n_slots]):
        lo, hi = FAMILY_RANGE.get(fam, DEFAULT_RANGE)
        rows = [spec.row(p) for p in range(lo, hi + 1) if spec.row(p) >= 0]
        mask[2 * s, 0, rows] = True
        mask[2 * s + 1, 0, rows] = True
    if drums is not False:
        rows = [spec.row(p) for p in range(DRUM_RANGE[0], DRUM_RANGE[1] + 1) if spec.row(p) >= 0]
        mask[spec.drum_channel, 0, rows] = True
    return mask
