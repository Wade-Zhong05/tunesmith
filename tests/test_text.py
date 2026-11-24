"""Rule parser, prompt helpers, parser dataset construction and metrics.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import random

import pytest
import yaml

from tunesmith.attributes import MusicAttributes
from tunesmith.eval.parser_metrics import field_accuracy
from tunesmith.paths import ROOT, load_yaml, read_jsonl
from tunesmith.understand.make_dataset import build, clean_target, template_rows
from tunesmith.understand.prompts import build_messages, extract_json, target_json
from tunesmith.understand.rules import load_keywords, mentions, normalize_text, parse

CASES = yaml.safe_load(open(ROOT / "tests" / "fixtures" / "rule_cases.yaml", encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[str(i) for i in range(len(CASES))])
def test_rule_parser(case):
    got = parse(case["text"]).model_dump()
    for field, value in case["expect"].items():
        assert got[field] == value, (field, got)


def test_extract_json_tolerant():
    assert extract_json('Sure! {"key": "C major", "bars": null}') == {"key": "C major", "bars": None}
    assert extract_json("{'mood': 'sad', 'bars': 16}") == {"mood": "sad", "bars": 16}
    assert extract_json("no json here") is None


def test_prompt_modes():
    zero, few, lora = (build_messages("a sad song", m) for m in ("zero", "few", "lora"))
    assert zero[-1]["content"] == "a sad song" and len(few) > len(zero) and len(lora) == 2
    assert target_json({"key": "C major"}).startswith('{"key": "C major", "tempo_bpm": null')


def test_clean_target_only_keeps_mentioned_fields():
    caption = "A melodic electronic composition featuring a string ensemble and trumpet, in F# minor, at an Allegro tempo."
    row = {"key": "F# minor", "tempo": "135", "time_signature": "4/4", "instrument_numbers_sorted": [48, 56],
           "mood": ["film", "epic"], "genre": ["electronic"]}
    t = clean_target(caption, row, load_keywords())
    assert t["key"] == "F# minor" and t["tempo_bpm"] == 135 and t["genre"] == "electronic"
    assert t["meter"] is None  # "4/4" is not in the caption
    assert t["mood"] is None   # "epic" is not mentioned either
    assert t["instrument"] == "strings"
    assert mentions("in 3/4 time", "meter", "3/4")


def test_template_rows_are_consistent():
    rows = template_rows(200, random.Random(0), load_keywords(), load_yaml("configs/parser_templates.yaml")["templates"])
    agree = sum(parse(r["text"]).field_value(f) == MusicAttributes.lenient(r["target"]).field_value(f)
                for r in rows for f in ("key", "instrument", "mood") if r["target"].get(f))
    total = sum(1 for r in rows for f in ("key", "instrument", "mood") if r["target"].get(f))
    assert total > 50 and agree / total > 0.9  # templates say what their targets say


def test_build_parser_dataset(tmp_path, synth_data):
    raw, _ = synth_data
    stats = build(str(raw / "metadata.jsonl"), str(tmp_path), {"train": 40, "val": 10, "test": 10},
                  {"train": 20, "val": 4, "test": 4}, val_fraction=0.2)
    assert stats["train"]["rows"] > 20
    row = next(read_jsonl(tmp_path / "train.jsonl"))
    assert set(row) == {"text", "target", "source"}


def test_field_accuracy():
    gold = [MusicAttributes(key="C major", mood="sad"), MusicAttributes(tempo_bpm=130)]
    pred = [MusicAttributes(key="C major", mood="happy", genre="pop"), None]
    m = field_accuracy(pred, gold)
    assert m["key_accuracy"] == 1.0 and m["mood_accuracy"] == 0.0 and m["tempo_accuracy"] == 0.0
    assert m["genre_hallucination"] == 0.5 and m["valid_rate"] == 0.5 and m["exact_match"] == 0.0


def test_rule_parser_repairs_stretched_and_misspelt_keywords():
    a = parse("a very very saaddd and very sloooow song with violin piano and guita")
    assert a.mood == "sad" and a.tempo_bpm == 52 and a.instruments == ["violin", "piano", "guitar"]
    assert normalize_text("happpy jazzzz with pinao and violn") == "happy jazz with piano and violin"
    # word forms and substituted letters are left alone: they are other words, not typos
    assert normalize_text("guitars voice space dancer horse") == "guitars voice space dancer horse"
