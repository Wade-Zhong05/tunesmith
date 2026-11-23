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
    out.update(ensemble_accuracy(preds, golds))
    return out


def ensemble_accuracy(preds: list[Optional[MusicAttributes]], golds: list[MusicAttributes]) -> dict[str, float]:
    """Instrument-list F1 (order ignored) where the gold names instruments; drums accuracy where
    the gold says yes/no, and drums hallucination where it says nothing."""
    out: dict[str, float] = {}
    f1s, drum_ok, drum_support, drum_halluc, drum_empty = [], 0, 0, 0, 0
    for p, g in zip(preds, golds):
        if g.instruments:
            got = set(p.instruments or ([p.instrument] if p and p.instrument else [])) if p else set()
            want = set(g.instruments)
            hit = len(got & want)
            prec, rec = (hit / len(got) if got else 0.0), hit / len(want)
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        pd = p.drums if p is not None else None
        if g.drums is None:
            drum_empty += 1
            drum_halluc += pd is not None
        else:
            drum_support += 1
            drum_ok += pd == g.drums
    out["instruments_support"] = len(f1s)
    if f1s:
        out["instruments_f1"] = sum(f1s) / len(f1s)
    out["drums_support"] = drum_support
    if drum_support:
        out["drums_accuracy"] = drum_ok / drum_support
    if drum_empty:
        out["drums_hallucination"] = drum_halluc / drum_empty
    return out
