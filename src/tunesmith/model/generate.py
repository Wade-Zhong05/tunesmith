"""Generation: REMI grammar constraints, key bias, classifier-free guidance, sampling loop.

The grammar guarantees every generated sequence decodes to valid MIDI:

melody:   SEP -> Bar -> TimeSig -> (Position -> Pitch -> Velocity -> Duration)* -> Bar ... -> EOS
ensemble: SEP -> Bar -> TimeSig -> (Position -> (Program -> Pitch|PitchDrum -> Duration)+)* -> Bar ... -> EOS

Positions strictly increase inside a bar and stay below the bar length, the time
signature matches the requested meter, and generation stops after the requested number
of bars. In ensemble mode only the requested instrument families (and drums, if wanted)
may appear, each family plays inside its own pitch range, at most `max_poly` notes per
family share an onset, and EOS waits until every requested family has played (as long
as the last bar still has room).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; ensemble grammar
added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..attributes import INSTRUMENTS, MAX_INSTRUMENTS, MusicAttributes, key_pitch_classes
from ..tokenizer import (DRUM_PROGRAM, PROGRAM_TO_FAMILY, T_BAR, T_DURATION, T_EOS, T_PITCH, T_PITCHDRUM,
                         T_POSITION, T_PROGRAM, T_SEP, T_TIMESIG, T_VELOCITY, TuneTokenizer)
from . import sampling_filters_ref
from .sampling_filters import sample_next

log = logging.getLogger(__name__)


@dataclass
class GrammarState:
    target_bars: Optional[int]
    timesig_lock: Optional[int]
    polyphonic: bool = False
    rare_meters: bool = False
    last: int = T_SEP
    bars: int = 0
    pos: int = -1
    ppb: int = 32
    done: bool = False
    history: list[int] = field(default_factory=list)
    # rests: a bar may only end without notes if fewer than max_empty bars before it were empty,
    # and never before the first note (None: no limit, the behaviour of the raw model)
    max_empty: Optional[int] = 1
    bar_notes: int = 0
    empty_run: int = 0
    any_note: bool = False
    # ensemble mode
    programs: Optional[torch.Tensor] = None           # allowed Program tokens
    required: set[int] = field(default_factory=set)   # programs that must play at least once
    seen: set[int] = field(default_factory=set)
    program: int = 0                                   # program of the current note
    free: bool = False                                 # no families requested: cap at MAX_INSTRUMENTS
    onset_counts: dict[int, int] = field(default_factory=dict)


# Playable range per family (MIDI pitch), used to keep each part idiomatic in ensemble mode.
FAMILY_RANGE = {
    "bass": (28, 60), "cello": (36, 76), "violin": (55, 100), "flute": (60, 98), "trumpet": (54, 84),
    "brass": (40, 84), "sax": (46, 84), "woodwind": (50, 90), "choir": (48, 84), "guitar": (40, 88),
    "electric_guitar": (40, 88), "mallet": (53, 96), "synth_lead": (48, 96),
}
DEFAULT_RANGE = (36, 96)


class RemiGrammar:
    def __init__(self, tok: TuneTokenizer, pitch_range: tuple[int, int] = (36, 96)):
        self.tok = tok
        V = tok.vocab_size
        types = torch.from_numpy(tok.types.astype(np.int64))
        self.by_type = {t: types == t for t in (T_BAR, T_TIMESIG, T_POSITION, T_PITCH, T_VELOCITY, T_DURATION)}
        pitch = torch.from_numpy(tok.pitch.astype(np.int64))
        self.by_type[T_PITCH] = self.by_type[T_PITCH] & (pitch >= pitch_range[0]) & (pitch <= pitch_range[1])
        self.position_value = torch.from_numpy(tok.position.astype(np.int64))
        self.eos = torch.zeros(V, dtype=torch.bool)
        self.eos[tok.eos_id] = True
        self.pad = torch.zeros(V, dtype=torch.bool)
        self.pad[tok.pad_id] = True
        self.common_ts = torch.zeros(V, dtype=torch.bool)
        for meter in ("4/4", "3/4", "2/4", "6/8"):
            tid = tok.timesig_id(meter)
            if tid is not None:
                self.common_ts[tid] = True
        self.ensemble = tok.ensemble
        self.max_poly = 6
        if self.ensemble:
            self.by_type[T_PROGRAM] = types == T_PROGRAM
            self.by_type[T_PITCHDRUM] = types == T_PITCHDRUM
            raw_pitch = torch.from_numpy(tok.pitch.astype(np.int64))
            all_pitch = types == T_PITCH
            # per-program masks of the notes that program may play
            self.notes_for: dict[int, torch.Tensor] = {}
            for i in np.nonzero(tok.types == T_PROGRAM)[0]:
                prog = int(tok.program[i])
                if prog == DRUM_PROGRAM:
                    self.notes_for[prog] = self.by_type[T_PITCHDRUM]
                else:
                    lo, hi = FAMILY_RANGE.get(PROGRAM_TO_FAMILY.get(prog, ""), DEFAULT_RANGE)
                    self.notes_for[prog] = all_pitch & (raw_pitch >= lo) & (raw_pitch <= hi)
            self.pitched_programs = self.by_type[T_PROGRAM] & torch.from_numpy(tok.program.astype(np.int64) >= 0)

    def new_state(self, attrs: MusicAttributes, polyphonic: bool = False,
                  max_empty: Optional[int] = 1) -> GrammarState:
        lock = self.tok.timesig_id(attrs.meter) if attrs.meter and attrs.meter != "other" else None
        st = GrammarState(target_bars=attrs.bars, timesig_lock=lock, polyphonic=polyphonic,
                          rare_meters=attrs.meter == "other", max_empty=max_empty)
        if self.ensemble:
            allowed = torch.zeros_like(self.pad)
            families = attrs.instruments or ([attrs.instrument] if attrs.instrument else [])
            for fam in families:
                tid = self.tok.program_id(fam)
                if tid is not None:
                    allowed[tid] = True
                    st.required.add(INSTRUMENTS[fam])
            if not families:
                allowed |= self.pitched_programs
                st.free = True
            drum_tid = self.tok.program_id(None)
            if drum_tid is not None and attrs.drums is not False:
                allowed[drum_tid] = True
                if attrs.drums:
                    st.required.add(DRUM_PROGRAM)
            st.programs = allowed
        return st

    def _programs(self, st: GrammarState) -> torch.Tensor:
        m = st.programs.clone()
        if st.free and len(st.seen - {DRUM_PROGRAM}) >= MAX_INSTRUMENTS:
            # an unspecified ensemble may use at most MAX_INSTRUMENTS pitched families
            keep = torch.zeros_like(m)
            for prog in st.seen | {DRUM_PROGRAM}:
                tid = self.tok.program_id(None if prog == DRUM_PROGRAM else PROGRAM_TO_FAMILY.get(prog))
                if tid is not None:
                    keep[tid] = True
            m &= keep
        for prog, count in st.onset_counts.items():
            if count >= self.max_poly:
                tid = self.tok.program_id(None if prog == DRUM_PROGRAM else PROGRAM_TO_FAMILY.get(prog))
                if tid is not None:
                    m[tid] = False
        return m

    def _allowed_ensemble(self, st: GrammarState) -> torch.Tensor:
        can_bar = st.target_bars is None or st.bars < st.target_bars
        if st.last == T_POSITION:
            return self._programs(st)
        if st.last == T_PROGRAM:
            return self.notes_for.get(st.program, self.eos)
        if st.last in (T_PITCH, T_PITCHDRUM):
            return self.by_type[T_DURATION]
        if st.last == T_DURATION:
            positions = self._positions(st)
            m = positions | self._programs(st)
            if can_bar:
                m = m | self.by_type[T_BAR]
            finished = st.target_bars is None or st.bars >= st.target_bars
            if finished and (st.required <= st.seen or not bool(positions.any())):
                m = m | self.eos
            if not bool(m.any()):
                m = self.eos.clone()
            return m
        return self.eos

    def _positions(self, st: GrammarState) -> torch.Tensor:
        return self.by_type[T_POSITION] & (self.position_value > st.pos) & (self.position_value < st.ppb)

    @staticmethod
    def _may_rest(st: GrammarState) -> bool:
        """May the current bar, which has no notes yet, end empty?"""
        if st.max_empty is None:
            return True
        return st.any_note and st.empty_run < st.max_empty

    def allowed(self, st: GrammarState) -> torch.Tensor:
        if st.done:
            return self.pad
        can_bar = st.target_bars is None or st.bars < st.target_bars
        if st.last == T_SEP:
            return self.by_type[T_BAR]
        if self.ensemble and st.last not in (T_BAR, T_TIMESIG):
            return self._allowed_ensemble(st)
        if st.last == T_BAR:
            if st.timesig_lock is not None:
                m = torch.zeros_like(self.pad)
                m[st.timesig_lock] = True
                return m
            # unspecified meter: one of the four common ones; "other": anything
            return self.by_type[T_TIMESIG] if st.rare_meters else self.common_ts
        if st.last == T_TIMESIG:
            m = self._positions(st)
            if can_bar and self._may_rest(st):
                m = m | self.by_type[T_BAR]
            return m
        if st.last == T_POSITION:
            return self.by_type[T_PITCH]
        if st.last == T_PITCH:
            return self.by_type[T_VELOCITY]
        if st.last == T_VELOCITY:
            return self.by_type[T_DURATION]
        if st.last == T_DURATION:
            m = self._positions(st)
            if st.polyphonic:
                m = m | self.by_type[T_PITCH]
            if can_bar:
                m = m | self.by_type[T_BAR]
            if st.target_bars is None or st.bars >= st.target_bars:
                m = m | self.eos
            if not bool(m.any()):
                m = self.eos.clone()
            return m
        return self.eos

    def update(self, st: GrammarState, token_id: int) -> None:
        if st.done:
            return
        t = int(self.tok.types[token_id])
        st.history.append(token_id)
        if t == T_EOS:
            st.done = True
        elif t == T_BAR:
            if st.bars > 0:
                st.empty_run = st.empty_run + 1 if st.bar_notes == 0 else 0
            st.bars += 1
            st.bar_notes = 0
            st.pos = -1
            st.onset_counts = {}
        elif t == T_TIMESIG:
            num, den = self.tok.timesig[token_id]
            st.ppb = self.tok.positions_per_bar(num, den)
            st.timesig_lock = token_id
        elif t == T_POSITION:
            st.pos = int(self.tok.position[token_id])
            st.onset_counts = {}
        elif t == T_PROGRAM:
            st.program = int(self.tok.program[token_id])
            st.onset_counts[st.program] = st.onset_counts.get(st.program, 0) + 1
        elif t in (T_PITCH, T_PITCHDRUM):
            st.bar_notes += 1
            st.any_note = True
            if self.ensemble:
                st.seen.add(st.program)
        st.last = t


def key_bias_vector(tok: TuneTokenizer, key: Optional[str], strength: float) -> torch.Tensor:
    """Additive logit bias: -strength for Pitch tokens outside the key's scale."""
    bias = torch.zeros(tok.vocab_size)
    if not key or strength <= 0:
        return bias
    pcs = key_pitch_classes(key)
    for i in np.nonzero(tok.types == T_PITCH)[0]:
        if int(tok.pitch[i]) % 12 not in pcs:
            bias[int(i)] = -float(strength)
    return bias


_warned_filters = False


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
            generator: Optional[torch.Generator]) -> torch.Tensor:
    """The student's sample_next; the reference filters stand in while it is unfinished."""
    global _warned_filters
    try:
        return sample_next(logits, temperature, top_k, top_p, generator)
    except NotImplementedError as e:
        if not _warned_filters:
            log.warning("student sampling filters not implemented yet (%s); using the reference filters", e)
            _warned_filters = True
        return sampling_filters_ref.sample_next(logits, temperature, top_k, top_p, generator)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tok: TuneTokenizer,
    attrs_list: list[MusicAttributes],
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    cfg_scale: float = 1.0,
    key_bias: float = 0.0,
    max_len: Optional[int] = None,
    grammar: bool = True,
    polyphonic: bool = False,
    seed: Optional[int] = None,
    pitch_range: tuple[int, int] = (36, 96),
    max_empty_bars: Optional[int] = 1,
) -> list[list[int]]:
    """Generate one sequence per attribute set. Returns full id lists (prompt + music + EOS).

    With the grammar on, the piece starts with notes in bar 1 and never has more than
    `max_empty_bars` silent bars in a row (None lifts the limit).
    """
    model.eval()
    device = next(model.parameters()).device
    B = len(attrs_list)
    gram = RemiGrammar(tok, pitch_range)
    states = [gram.new_state(a, polyphonic, max_empty_bars) for a in attrs_list]
    prompts = [tok.prompt_ids(a) for a in attrs_list]
    use_cfg = cfg_scale != 1.0
    null_prompt = tok.null_prompt_ids()
    batch_prompts = prompts + ([null_prompt] * B if use_cfg else [])

    model_max = int(getattr(getattr(model, "cfg", None), "max_seq_len", 1024))
    max_len = min(max_len or model_max, model_max)
    bias = torch.stack([key_bias_vector(tok, a.key, key_bias) for a in attrs_list]) if key_bias > 0 else None
    gen = torch.Generator().manual_seed(int(seed)) if seed is not None else None

    seqs = [list(p) for p in prompts]
    cur_len = len(prompts[0])
    logits, cache = model(torch.tensor(batch_prompts, device=device), use_cache=True)
    last = logits[:, -1, :]

    while True:
        step_logits = last.float().cpu()
        if use_cfg:
            step_logits = step_logits[B:] + cfg_scale * (step_logits[:B] - step_logits[B:])
        if bias is not None:
            step_logits = step_logits + bias
        at_limit = cur_len + 1 >= max_len
        if grammar or at_limit:
            mask = torch.stack([
                (gram.eos if (at_limit and not st.done) else gram.allowed(st)) if grammar
                else (gram.eos if not st.done else gram.pad)
                for st in states
            ])
            step_logits = step_logits.masked_fill(~mask, float("-inf"))
        nxt = _sample(step_logits, temperature, top_k, top_p, gen)
        for b, st in enumerate(states):
            tid = tok.pad_id if st.done else int(nxt[b])
            nxt[b] = tid
            if not st.done:
                seqs[b].append(tid)
                if grammar:
                    gram.update(st, tid)
                elif tid == tok.eos_id:
                    st.done = True
        cur_len += 1
        if all(st.done for st in states) or cur_len >= max_len:
            break
        step = torch.cat([nxt, nxt]) if use_cfg else nxt
        if cache is not None:
            logits, cache = model(step.view(-1, 1).to(device), past_kv=cache, use_cache=True)
        else:
            rows = [s + [tok.pad_id] * (cur_len - len(s)) for s in seqs]
            if use_cfg:
                rows = rows + [null_prompt + r[len(null_prompt):] for r in rows]
            logits, _ = model(torch.tensor(rows, device=device))
        last = logits[:, -1, :]
    out = []
    for s in seqs:
        if tok.eos_id not in s:
            s = s + [tok.eos_id]
        out.append(s[: s.index(tok.eos_id) + 1])
    return out
