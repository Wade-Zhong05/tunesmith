"""Tiny additive synthesizer (numpy only): the offline fallback when no SoundFont is present.

Each instrument family maps to a preset of harmonic amplitudes and an ADSR envelope.
It sounds like a simple keyboard, which is enough to audition melodies everywhere.
`render_drums` synthesises a basic kit (sine-sweep kick, noise snare, hi-hats, toms, cymbals)
for General MIDI drum notes, so ensemble pieces can be auditioned too.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10; drums and
multi-track mixing added 2026-11-22. See ATTRIBUTION.md.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..attributes import program_to_family

# harmonics, attack, decay, sustain level, release (seconds)
PRESETS = {
    "keys": ([1.0, 0.5, 0.3, 0.15, 0.08, 0.04], 0.005, 0.8, 0.2, 0.25),
    "pluck": ([1.0, 0.7, 0.35, 0.2, 0.1], 0.003, 0.5, 0.05, 0.2),
    "sustain": ([1.0, 0.45, 0.25, 0.15, 0.08], 0.06, 0.15, 0.8, 0.2),
    "square": ([1.0, 0.0, 0.33, 0.0, 0.2, 0.0, 0.14], 0.01, 0.1, 0.7, 0.1),
}
FAMILY_PRESET = {
    "piano": "keys", "electric_piano": "keys", "mallet": "pluck", "guitar": "pluck", "electric_guitar": "pluck",
    "harp": "pluck", "ethnic": "pluck", "bass": "pluck", "synth_lead": "square",
}


def preset_for_program(program: int) -> str:
    return FAMILY_PRESET.get(program_to_family(program) or "piano", "sustain")


def render_notes(notes_s: list[tuple[float, float, int, int]], sample_rate: int = 44100,
                 preset: str = "keys") -> np.ndarray:
    """notes_s: (start_seconds, duration_seconds, pitch, velocity) -> mono float32 audio."""
    harmonics, attack, decay, sustain, release = PRESETS[preset]
    if not notes_s:
        return np.zeros(sample_rate // 2, dtype=np.float32)
    end = max(s + d for s, d, _, _ in notes_s) + release + 0.1
    out = np.zeros(int(end * sample_rate) + 1, dtype=np.float32)
    for start, dur, pitch, vel in notes_s:
        n_on = max(1, int(dur * sample_rate))
        n_rel = int(release * sample_rate)
        t = np.arange(n_on + n_rel) / sample_rate
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        wave_ = np.zeros_like(t)
        for k, a in enumerate(harmonics, start=1):
            if a and k * f0 < sample_rate / 2:
                wave_ += a * np.sin(2 * np.pi * k * f0 * t)
        env = np.ones_like(t)
        na, nd = max(1, int(attack * sample_rate)), max(1, int(decay * sample_rate))
        env[:na] = np.linspace(0, 1, na)
        dec_end = min(n_on, na + nd)
        if dec_end > na:
            env[na:dec_end] = np.linspace(1, sustain, dec_end - na)
        env[dec_end:n_on] = sustain if preset in ("sustain", "square") else np.linspace(
            env[dec_end - 1] if dec_end > 0 else sustain, sustain * 0.3, max(0, n_on - dec_end))
        level = env[n_on - 1] if n_on > 0 else 0.0
        env[n_on:] = np.linspace(level, 0, len(t) - n_on)
        i0 = int(start * sample_rate)
        seg = (wave_ * env * (vel / 127.0) * 0.25).astype(np.float32)
        out[i0:i0 + len(seg)] += seg[: len(out) - i0]
    peak = float(np.abs(out).max())
    return out / peak * 0.9 if peak > 0 else out


def _drum_hit(pitch: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """One drum sound for a General MIDI percussion note."""
    if pitch in (35, 36):  # kick: falling sine
        n = int(0.35 * sample_rate)
        t = np.arange(n) / sample_rate
        freq = 50 + 90 * np.exp(-t * 30)
        return np.sin(2 * np.pi * np.cumsum(freq) / sample_rate) * np.exp(-t * 9)
    if pitch in (38, 40, 37, 39):  # snare / clap: noise + body
        n = int(0.2 * sample_rate)
        t = np.arange(n) / sample_rate
        return 0.6 * rng.standard_normal(n) * np.exp(-t * 22) + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 30)
    if pitch in (42, 44, 46):  # hi-hats: short (closed) or longer (open) high noise
        n = int((0.25 if pitch == 46 else 0.06) * sample_rate)
        noise = rng.standard_normal(n)
        noise = np.diff(noise, prepend=0.0)  # crude high-pass
        return 0.35 * noise * np.exp(-np.arange(n) / sample_rate * (12 if pitch == 46 else 60))
    if pitch in (49, 51, 52, 55, 57, 59):  # cymbals
        n = int(0.8 * sample_rate)
        noise = np.diff(rng.standard_normal(n), prepend=0.0)
        return 0.25 * noise * np.exp(-np.arange(n) / sample_rate * 4)
    # toms and everything else: pitched decaying sine
    n = int(0.3 * sample_rate)
    t = np.arange(n) / sample_rate
    f = 80 + 6 * (pitch - 41)
    return 0.7 * np.sin(2 * np.pi * f * t) * np.exp(-t * 12)


def render_drums(hits_s: list[tuple[float, int, int]], sample_rate: int = 44100, seed: int = 0) -> np.ndarray:
    """hits_s: (start_seconds, pitch, velocity) -> mono float32 audio (not normalised)."""
    if not hits_s:
        return np.zeros(1, dtype=np.float32)
    rng = np.random.default_rng(seed)
    end = max(s for s, _, _ in hits_s) + 1.0
    out = np.zeros(int(end * sample_rate) + 1, dtype=np.float32)
    for start, pitch, vel in hits_s:
        hit = (_drum_hit(int(pitch), sample_rate, rng) * (vel / 127.0) * 0.5).astype(np.float32)
        i0 = int(start * sample_rate)
        out[i0:i0 + len(hit)] += hit[: len(out) - i0]
    return out


def mix(tracks: list[np.ndarray], gains: list[float] | None = None) -> np.ndarray:
    """Sum mono tracks of different lengths and normalise the peak to 0.9."""
    tracks = [t for t in tracks if len(t)]
    if not tracks:
        return np.zeros(1, dtype=np.float32)
    gains = gains or [1.0] * len(tracks)
    out = np.zeros(max(len(t) for t in tracks), dtype=np.float32)
    for t, g in zip(tracks, gains):
        out[: len(t)] += g * t
    peak = float(np.abs(out).max())
    return out / peak * 0.9 if peak > 0 else out


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 44100) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:  # (channels, samples) -> interleaved
        channels = audio.shape[0]
        data = audio.T.reshape(-1)
    else:
        channels, data = 1, audio
    pcm = (np.clip(data, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr
