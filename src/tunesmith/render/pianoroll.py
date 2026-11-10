"""Piano-roll data for the web UI and PNG plots for reports.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from symusic import Score


def notes_for_ui(score: Score) -> dict[str, Any]:
    """Notes in quarter-note beats, plus the bar length, for drawing on a canvas."""
    tpq = score.tpq
    ts = score.time_signatures[0] if len(score.time_signatures) else None
    num, den = (int(ts.numerator), int(ts.denominator)) if ts else (4, 4)
    notes = [{"pitch": int(n.pitch), "start": n.time / tpq, "dur": n.duration / tpq, "vel": int(n.velocity)}
             for tr in score.tracks if not tr.is_drum for n in tr.notes]
    total = max((n["start"] + n["dur"] for n in notes), default=0.0)
    beats_per_bar = num * 4 / den
    return {
        "notes": notes,
        "beats_per_bar": beats_per_bar,
        "meter": f"{num}/{den}",
        "total_beats": total,
        "bars": int(-(-total // beats_per_bar)) if total else 0,
        "qpm": float(score.tempos[0].qpm) if len(score.tempos) else 120.0,
    }


def plot_pianoroll(score: Score, path: str | Path, title: Optional[str] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = notes_for_ui(score)
    fig, ax = plt.subplots(figsize=(10, 3))
    for n in data["notes"]:
        ax.add_patch(plt.Rectangle((n["start"], n["pitch"] - 0.4), n["dur"], 0.8,
                                   color=plt.cm.viridis(n["vel"] / 127)))
    pitches = [n["pitch"] for n in data["notes"]] or [60]
    ax.set_xlim(0, max(1.0, data["total_beats"]))
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    bar = data["beats_per_bar"]
    x = 0.0
    while x <= data["total_beats"]:
        ax.axvline(x, color="0.8", lw=0.6, zorder=0)
        x += bar
    ax.set_xlabel("beats")
    ax.set_ylabel("MIDI pitch")
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
