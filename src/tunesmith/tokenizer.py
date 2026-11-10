"""Tokenizer: control-token prompt + REMI music tokens (miditok) in one vocabulary.

A training sequence looks like

    BOS KEY_A_minor METER_3/4 TEMPO_slow INST_piano MOOD_sad GENRE_NULL BARS_16 SEP
    Bar TimeSig_3/4 Position_0 Pitch_69 Velocity_79 Duration_1.0.8 Position_8 ... EOS

Ids 0..offset-1 are our special and control tokens; REMI token i has id offset + i.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from miditok import REMI, TokenizerConfig
from symusic import Score, Tempo

from .attributes import CONTROL_FIELDS, INSTRUMENTS, MusicAttributes, control_vocab, transpose_key

SPECIALS = ("PAD", "BOS", "SEP", "EOS")

# token type codes
T_PAD, T_BOS, T_SEP, T_EOS, T_CTRL, T_BAR, T_TIMESIG, T_POSITION, T_PITCH, T_VELOCITY, T_DURATION, T_OTHER = range(12)
_REMI_TYPES = {"PAD": T_PAD, "Bar": T_BAR, "TimeSig": T_TIMESIG, "Position": T_POSITION,
               "Pitch": T_PITCH, "Velocity": T_VELOCITY, "Duration": T_DURATION}

DEFAULT_CONFIG: dict[str, Any] = {
    "pitch_range": [21, 109],
    "beat_res": {"0_4": 8, "4_12": 4},
    "num_velocities": 8,
}


class TuneTokenizer:
    def __init__(self, config: Optional[dict[str, Any]] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.config = cfg
        beat_res = {tuple(int(x) for x in str(k).split("_")): int(v) for k, v in cfg["beat_res"].items()}
        self.res = max(beat_res.values())
        self.remi = REMI(TokenizerConfig(
            pitch_range=tuple(cfg["pitch_range"]),
            beat_res=beat_res,
            num_velocities=int(cfg["num_velocities"]),
            special_tokens=["PAD"],
            use_chords=False,
            use_rests=False,
            use_tempos=False,
            use_time_signatures=True,
            use_programs=False,
        ))
        cv = control_vocab()
        self.control = list(SPECIALS) + [t for f in CONTROL_FIELDS for t in cv[f]]
        self.offset = len(self.control)
        remi_vocab: dict[str, int] = dict(self.remi.vocab)
        self.tokens: list[str] = self.control + [""] * len(remi_vocab)
        for tok, i in remi_vocab.items():
            self.tokens[self.offset + i] = tok
        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.vocab_size = len(self.tokens)

        self.pad_id, self.bos_id, self.sep_id, self.eos_id = (self.token_to_id[s] for s in SPECIALS)
        self.prompt_len = 2 + len(CONTROL_FIELDS)

        V = self.vocab_size
        self.types = np.full(V, T_OTHER, dtype=np.int8)
        self.pitch = np.full(V, -1, dtype=np.int16)
        self.position = np.full(V, -1, dtype=np.int16)
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
            elif kind == "Position":
                self.position[i] = int(value)
            elif kind == "TimeSig":
                num, den = value.split("/")
                self.timesig[i] = (int(num), int(den))

    # ------------------------------------------------------------ prompt

    def prompt_ids(self, attrs: MusicAttributes) -> list[int]:
        return [self.bos_id] + [self.token_to_id[t] for t in attrs.control_tokens()] + [self.sep_id]

    def null_prompt_ids(self) -> list[int]:
        return self.prompt_ids(MusicAttributes())

    def slot(self, field: str) -> int:
        """Index of a field's control token inside the prompt."""
        return 1 + CONTROL_FIELDS.index(field)

    def null_id(self, field: str) -> int:
        return self.token_to_id[f"{field.upper()}_NULL"]

    def attrs_from_prompt(self, ids: list[int]) -> MusicAttributes:
        data: dict[str, Any] = {}
        for i in ids[1:self.prompt_len - 1]:
            field, _, value = self.tokens[i].partition("_")
            field = field.lower()
            if value == "NULL":
                continue
            if field == "key":
                data["key"] = value.replace("_", " ")
            elif field == "tempo":
                from .attributes import tempo_from_bin
                data["tempo_bpm"] = tempo_from_bin(value)
            elif field == "bars":
                data["bars"] = int(value)
            else:
                data[field] = value
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
        score = self.remi.decode([music], programs=[(int(program), False)])
        score.tempos.clear()
        score.tempos.append(Tempo(0, float(qpm)))
        for track in score.tracks:
            track.program = int(program)
        clip_to_bars(score, self.count_bars(ids))
        return score

    def decode_for(self, ids: list[int], attrs: MusicAttributes) -> Score:
        """Decode with the playback program and tempo implied by the attributes."""
        program = INSTRUMENTS.get(attrs.instrument or "piano", 0)
        qpm = float(attrs.tempo_bpm) if attrs.tempo_bpm else 100.0
        return self.decode(ids, program=program, qpm=qpm)

    # ------------------------------------------------------------ augmentation helpers

    def transpose_table(self, semitones: int) -> np.ndarray:
        """id -> id after shifting every Pitch token; -1 where the result leaves the range."""
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
