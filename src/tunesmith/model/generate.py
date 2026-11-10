"""Generation: REMI grammar constraints, key bias, classifier-free guidance, sampling loop.

The grammar guarantees every generated sequence decodes to valid MIDI:
SEP -> Bar -> TimeSig -> (Position -> Pitch -> Velocity -> Duration)* -> Bar ... -> EOS,
positions strictly increase inside a bar and stay below the bar length, the time
signature matches the requested meter, and generation stops after the requested
number of bars.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from ..attributes import MusicAttributes, key_pitch_classes
from ..tokenizer import (T_BAR, T_DURATION, T_EOS, T_PITCH, T_POSITION, T_SEP, T_TIMESIG,
                         T_VELOCITY, TuneTokenizer)
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

    def new_state(self, attrs: MusicAttributes, polyphonic: bool = False) -> GrammarState:
        lock = self.tok.timesig_id(attrs.meter) if attrs.meter and attrs.meter != "other" else None
        return GrammarState(target_bars=attrs.bars, timesig_lock=lock, polyphonic=polyphonic,
                            rare_meters=attrs.meter == "other")

    def _positions(self, st: GrammarState) -> torch.Tensor:
        return self.by_type[T_POSITION] & (self.position_value > st.pos) & (self.position_value < st.ppb)

    def allowed(self, st: GrammarState) -> torch.Tensor:
        if st.done:
            return self.pad
        can_bar = st.target_bars is None or st.bars < st.target_bars
        if st.last == T_SEP:
            return self.by_type[T_BAR]
        if st.last == T_BAR:
            if st.timesig_lock is not None:
                m = torch.zeros_like(self.pad)
                m[st.timesig_lock] = True
                return m
            # unspecified meter: one of the four common ones; "other": anything
            return self.by_type[T_TIMESIG] if st.rare_meters else self.common_ts
        if st.last == T_TIMESIG:
            m = self._positions(st)
            if can_bar:
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
            st.bars += 1
            st.pos = -1
        elif t == T_TIMESIG:
            num, den = self.tok.timesig[token_id]
            st.ppb = self.tok.positions_per_bar(num, den)
            st.timesig_lock = token_id
        elif t == T_POSITION:
            st.pos = int(self.tok.position[token_id])
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
    """sample_next, falling back to plain sampling while the student TODOs are unfinished."""
    global _warned_filters
    try:
        return sample_next(logits, temperature, top_k, top_p, generator)
    except NotImplementedError as e:
        if not _warned_filters:
            log.warning("sampling filter not implemented yet (%s); using plain sampling", e)
            _warned_filters = True
        return sample_next(logits, 1.0, 0, 1.0, generator)


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
) -> list[list[int]]:
    """Generate one sequence per attribute set. Returns full id lists (prompt + music + EOS)."""
    model.eval()
    device = next(model.parameters()).device
    B = len(attrs_list)
    gram = RemiGrammar(tok, pitch_range)
    states = [gram.new_state(a, polyphonic) for a in attrs_list]
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
