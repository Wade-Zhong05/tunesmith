"""Tokenizer: control-token prompt + REMI music tokens (miditok) in one vocabulary.

Two modes (`config["mode"]`, default "melody"):

melody   - one monophonic part:
    BOS KEY_A_minor METER_3/4 TEMPO_slow INSTRUMENT_piano MOOD_sad GENRE_NULL BARS_16 SEP
    Bar TimeSig_3/4 Position_0 Pitch_69 Velocity_79 Duration_1.0.8 Position_8 ... EOS

ensemble - up to four instrument families plus drums in one interleaved stream (REMI with
           Program tokens, no velocity tokens):
    BOS KEY_C_major METER_4/4 TEMPO_fast INST1_piano INST2_strings INST3_bass INST4_NULL
        DRUMS_yes MOOD_happy GENRE_pop BARS_8 SEP
    Bar TimeSig_4/4 Position_0 Program_0 Pitch_60 Duration_2.0.8 Program_0 Pitch_64 ...
        Program_33 Pitch_36 ... Program_-1 PitchDrum_36 ... Position_8 ... EOS

Ids 0..offset-1 are our special and control tokens; REMI token i has id offset + i.
Melody-mode vocabularies are unchanged from the first version, so existing data and
checkpoints keep working.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble mode
added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from miditok import REMI, TokenizerConfig
from symusic import Score, Tempo

from .attributes import (INSTRUMENTS, MusicAttributes, control_fields, control_vocab, program_to_family,
                         tempo_from_bin, transpose_key)

SPECIALS = ("PAD", "BOS", "SEP", "EOS")

# token type codes
(T_PAD, T_BOS, T_SEP, T_EOS, T_CTRL, T_BAR, T_TIMESIG, T_POSITION, T_PITCH, T_VELOCITY, T_DURATION, T_OTHER,
 T_PROGRAM, T_PITCHDRUM) = range(14)
_REMI_TYPES = {"PAD": T_PAD, "Bar": T_BAR, "TimeSig": T_TIMESIG, "Position": T_POSITION, "Pitch": T_PITCH,
               "Velocity": T_VELOCITY, "Duration": T_DURATION, "Program": T_PROGRAM, "PitchDrum": T_PITCHDRUM}

DEFAULT_CONFIG: dict[str, Any] = {
    "pitch_range": [21, 109],
    "beat_res": {"0_4": 8, "4_12": 4},
    "num_velocities": 8,
}

DRUM_PROGRAM = -1
ENSEMBLE_PROGRAMS = sorted(set(INSTRUMENTS.values()))  # one representative GM program per family
PROGRAM_TO_FAMILY = {p: f for f, p in INSTRUMENTS.items()}

# Playback dynamics for ensemble mode (no velocity tokens): base velocity per family,
# plus an accent on the first beat of each bar and a smaller one on every beat.
_BASE_VELOCITY = {"drums": 92, "bass": 88, "strings": 70, "synth_pad": 66, "choir": 70, "organ": 72}
_DEFAULT_VELOCITY = 80


class TuneTokenizer:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = cfg
        self.mode = str(cfg.get("mode", "melody"))
        self.fields = control_fields(self.mode)
        beat_res = {tuple(int(x) for x in str(k).split("_")): int(v) for k, v in cfg["beat_res"].items()}
        self.res = max(beat_res.values())
        common = dict(pitch_range=tuple(cfg["pitch_range"]), beat_res=beat_res,
                      num_velocities=int(cfg["num_velocities"]), special_tokens=["PAD"], use_chords=False,
                      use_rests=False, use_tempos=False, use_time_signatures=True)
        if self.mode == "melody":
            self.remi = REMI(TokenizerConfig(**common, use_programs=False))
        else:
            self.remi = REMI(TokenizerConfig(**common, use_programs=True, one_token_stream_for_programs=True,
                                             programs=ENSEMBLE_PROGRAMS + [DRUM_PROGRAM],
                                             use_pitchdrum_tokens=True, use_velocities=False))
        cv = control_vocab(self.mode)
        self.control = list(SPECIALS) + [t for f in self.fields for t in cv[f]]
        self.offset = len(self.control)
        remi_vocab: dict[str, int] = dict(self.remi.vocab)
        self.tokens: list[str] = self.control + [""] * len(remi_vocab)
        for tok, i in remi_vocab.items():
            self.tokens[self.offset + i] = tok
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

        self.pad_id, self.bos_id, self.sep_id, self.eos_id = (self.token_to_id[s] for s in SPECIALS)
        self.prompt_len = 2 + len(self.fields)

        V = self.vocab_size
        self.types = np.full(V, T_OTHER, dtype=np.int8)
        self.pitch = np.full(V, -1, dtype=np.int16)        # Pitch tokens
        self.drum_pitch = np.full(V, -1, dtype=np.int16)   # PitchDrum tokens
        self.position = np.full(V, -1, dtype=np.int16)
        self.program = np.full(V, -999, dtype=np.int16)    # Program tokens (-1 = drums)
        self.timesig: dict[int, tuple[int, int]] = {}
        self.ctrl_field: dict[int, str] = {}
        for i, tok in enumerate(self.tokens):
            if i < len(SPECIALS):
                self.types[i] = (T_PAD, T_BOS, T_SEP, T_EOS)[i]
                continue
            if i < self.offset:
                self.types[i] = T_CTRL
                self.ctrl_field[i] = tok.split("_", 1)[0].lower()
                continue
            kind, _, value = tok.partition("_")
            self.types[i] = _REMI_TYPES.get(kind, T_OTHER)
            if kind == "Pitch":
                self.pitch[i] = int(value)
            elif kind == "PitchDrum":
                self.drum_pitch[i] = int(value)
            elif kind == "Position":
                self.position[i] = int(value)
            elif kind == "Program":
                self.program[i] = int(value)
            elif kind == "TimeSig":
                num, den = value.split("/")
                self.timesig[i] = (int(num), int(den))

    @property
    def ensemble(self) -> bool:
        return self.mode == "ensemble"

    # ------------------------------------------------------------ prompt

    def prompt_ids(self, attrs: MusicAttributes) -> list[int]:
        return [self.bos_id] + [self.token_to_id[t] for t in attrs.control_tokens(self.mode)] + [self.sep_id]

    def null_prompt_ids(self) -> list[int]:
        return self.prompt_ids(MusicAttributes())

    def slot(self, field: str) -> int:
        """Index of a field's control token inside the prompt."""
        return 1 + self.fields.index(field)

    def null_id(self, field: str) -> int:
        return self.token_to_id[f"{field.upper()}_NULL"]

    def attrs_from_prompt(self, ids: list[int]) -> MusicAttributes:
        data: dict[str, Any] = {}
        instruments: list[str] = []
        for i in ids[1:self.prompt_len - 1]:
            field, _, value = self.tokens[i].partition("_")
            field = field.lower()
            if value == "NULL":
                continue
            if field == "key":
                data["key"] = value.replace("_", " ")
            elif field == "tempo":
                data["tempo_bpm"] = tempo_from_bin(value)
            elif field == "bars":
                data["bars"] = int(value)
            elif field.startswith("inst") and field[4:].isdigit():
                instruments.append(value)
            elif field == "drums":
                data["drums"] = value == "yes"
            else:
                data[field] = value
        if instruments:
            data["instruments"] = instruments
        return MusicAttributes.lenient(data)

    # ------------------------------------------------------------ music

    def encode(self, score: Score, attrs: MusicAttributes) -> list[int]:
        seqs = self.remi.encode(score)
        seq = seqs[0] if isinstance(seqs, list) else seqs
        music = [self.offset + int(i) for i in seq.ids]
        return self.prompt_ids(attrs) + music + [self.eos_id]

    def music_ids(self, ids: list[int]) -> list[int]:
        """REMI ids (without offset) after SEP, up to EOS."""
        start = ids.index(self.sep_id) + 1 if self.sep_id in ids else 0
        out = []
        for i in ids[start:]:
            if i == self.eos_id:
                break
            if i >= self.offset:
                out.append(int(i) - self.offset)
        return out

    def count_bars(self, ids: list[int]) -> int:
        return int(sum(1 for i in ids if self.types[i] == T_BAR))

    def decode(self, ids: list[int], program: int = 0, qpm: float = 120.0) -> Score:
        music = self.music_ids(list(ids))
        if self.ensemble:
            score = self.remi.decode(music) if music else Score(self.remi.time_division)
            for track in score.tracks:
                track.name = "drums" if track.is_drum else (PROGRAM_TO_FAMILY.get(int(track.program))
                                                            or program_to_family(int(track.program)) or "track")
        else:
            score = self.remi.decode([music], programs=[(int(program), False)])
            for track in score.tracks:
                track.program = int(program)
        score.tempos.clear()
        score.tempos.append(Tempo(0, float(qpm)))
        clip_to_bars(score, self.count_bars(ids))
        if self.ensemble:
            apply_dynamics(score)
        return score

    def decode_for(self, ids: list[int], attrs: MusicAttributes) -> Score:
        """Decode with the playback program (melody mode) and tempo implied by the attributes."""
        program = INSTRUMENTS.get(attrs.instrument or "piano", 0)
        qpm = float(attrs.tempo_bpm) if attrs.tempo_bpm else 100.0
        return self.decode(ids, program=program, qpm=qpm)

    # ------------------------------------------------------------ augmentation and grammar helpers

    def transpose_table(self, semitones: int) -> np.ndarray:
        """id -> id after shifting every Pitch token; -1 where the result leaves the range.

        PitchDrum tokens (drum sounds) are not transposed."""
        table = np.arange(self.vocab_size, dtype=np.int64)
        for i in np.nonzero(self.types == T_PITCH)[0]:
            target = f"Pitch_{int(self.pitch[i]) + semitones}"
            table[i] = self.token_to_id.get(target, -1)
        for i, field in self.ctrl_field.items():
            if field == "key" and not self.tokens[i].endswith("_NULL"):
                key = self.tokens[i][len("KEY_"):].replace("_", " ")
                table[i] = self.token_to_id["KEY_" + transpose_key(key, semitones).replace(" ", "_")]
        return table

    def positions_per_bar(self, numerator: int, denominator: int) -> int:
        # miditok counts beats in units of the time-signature denominator (6/8 has 6 beats)
        return int(numerator * self.res)

    def timesig_id(self, meter: str) -> Optional[int]:
        return self.token_to_id.get(f"TimeSig_{meter}")

    def program_id(self, family: Optional[str]) -> Optional[int]:
        """Program token for an instrument family (None = drums)."""
        prog = DRUM_PROGRAM if family is None else INSTRUMENTS.get(family)
        return self.token_to_id.get(f"Program_{prog}") if prog is not None else None

    def config_dict(self) -> dict[str, Any]:
        return {**self.config, "vocab_size": self.vocab_size}


def clip_to_bars(score: Score, n_bars: int) -> None:
    """Cut notes that ring past the last generated bar line (in place)."""
    if n_bars <= 0 or not len(score.time_signatures):
        return
    ts = score.time_signatures[0]
    limit = int(round(score.tpq * 4 * ts.numerator / ts.denominator)) * n_bars
    for track in score.tracks:
        kept = []
        for n in track.notes:
            if n.time >= limit:
                continue
            if n.time + n.duration > limit:
                n.duration = limit - n.time
            kept.append(n)
        track.notes.clear()
        for n in kept:
            track.notes.append(n)


def apply_dynamics(score: Score) -> None:
    """Velocities for ensemble output (it has no velocity tokens): per-family base level,
    +12 on the first beat of a bar, +5 on other beats (in place)."""
    ts = score.time_signatures[0] if len(score.time_signatures) else None
    num, den = (int(ts.numerator), int(ts.denominator)) if ts else (4, 4)
    beat = int(score.tpq * 4 / den)
    bar = beat * num
    for track in score.tracks:
        base = _BASE_VELOCITY.get("drums" if track.is_drum else track.name, _DEFAULT_VELOCITY)
        for n in track.notes:
            accent = 12 if n.time % bar == 0 else (5 if n.time % beat == 0 else 0)
            n.velocity = int(min(127, base + accent))
