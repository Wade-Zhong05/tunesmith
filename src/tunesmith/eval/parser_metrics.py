"""Parser metrics: per-field accuracy on fields the gold label specifies, hallucination rate
on fields it leaves empty, exact match, and the share of outputs that were valid JSON.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from typing import Optional

from ..attributes import CONTROL_FIELDS, MusicAttributes


def field_accuracy(preds: list[Optional[MusicAttributes]], golds: list[MusicAttributes]) -> dict[str, float]:
    n = len(golds)
    out: dict[str, float] = {"n": n, "valid_rate": sum(p is not None for p in preds) / max(1, n)}
    accs, exact = [], 0
    for field in CONTROL_FIELDS:
        support = correct = halluc = empty = 0
        for p, g in zip(preds, golds):
            gv = g.field_value(field)
            pv = p.field_value(field) if p is not None else None
            if gv is None:
                empty += 1
                halluc += pv is not None
            else:
                support += 1
                correct += pv == gv
        out[f"{field}_support"] = support
        if support:
            out[f"{field}_accuracy"] = correct / support
            accs.append(correct / support)
        if empty:
            out[f"{field}_hallucination"] = halluc / empty
    for p, g in zip(preds, golds):
        if p is not None and all(p.field_value(f) == g.field_value(f) for f in CONTROL_FIELDS):
            exact += 1
    out["mean_field_accuracy"] = sum(accs) / len(accs) if accs else 0.0
    out["exact_match"] = exact / max(1, n)
    return out
