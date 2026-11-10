"""Prompts for the LLM parser (zero-shot, few-shot, and the format used for LoRA training).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import ast
import json
import re
from functools import lru_cache
from typing import Any, Optional

from ..attributes import BARS, GENRES, INSTRUMENTS, KEYS, METERS, MOODS
from ..paths import CONFIGS, load_yaml

SYSTEM_PROMPT = (
    "You convert a description of a piece of instrumental music into JSON "
    "with exactly these keys: key, tempo_bpm, meter, instrument, mood, genre, bars.\n"
    "Use null for anything the description does not say or clearly imply. Allowed values:\n"
    f"- key: one of {', '.join(KEYS)}\n"
    "- tempo_bpm: integer beats per minute (slow about 70, moderate about 90, fast about 130)\n"
    f"- meter: one of {', '.join(METERS)}\n"
    f"- instrument: one of {', '.join(INSTRUMENTS)}\n"
    f"- mood: one of {', '.join(MOODS)}\n"
    f"- genre: one of {', '.join(GENRES)}\n"
    f"- bars: one of {', '.join(map(str, BARS))}\n"
    "Answer with the JSON object only."
)

# Shorter instruction used for (and after) LoRA fine-tuning: the schema is learned.
LORA_SYSTEM_PROMPT = "Extract music attributes as JSON (key, tempo_bpm, meter, instrument, mood, genre, bars)."

FIELDS = ("key", "tempo_bpm", "meter", "instrument", "mood", "genre", "bars")


def target_json(target: dict[str, Any]) -> str:
    """Canonical compact JSON for a target (all fields, fixed order)."""
    return json.dumps({f: target.get(f) for f in FIELDS}, ensure_ascii=False, separators=(", ", ": "))


@lru_cache(maxsize=2)
def fewshot_examples(path: Optional[str] = None) -> list[dict[str, Any]]:
    return list(load_yaml(path or CONFIGS / "parser_fewshot.yaml").get("examples", []))


def build_messages(text: str, mode: str = "zero") -> list[dict[str, str]]:
    if mode == "lora":
        return [{"role": "system", "content": LORA_SYSTEM_PROMPT}, {"role": "user", "content": text}]
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if mode == "few":
        for ex in fewshot_examples():
            msgs.append({"role": "user", "content": ex["text"]})
            msgs.append({"role": "assistant", "content": target_json(ex["target"])})
    elif mode != "zero":
        raise ValueError(f"unknown prompt mode {mode!r}")
    msgs.append({"role": "user", "content": text})
    return msgs


def extract_json(output: str) -> Optional[dict[str, Any]]:
    """First {...} object in a model output; tolerant of single quotes and Python literals."""
    m = re.search(r"\{.*?\}", output, re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        try:
            obj = ast.literal_eval(blob.replace("null", "None").replace("true", "True").replace("false", "False"))
        except (ValueError, SyntaxError):
            return None
    return obj if isinstance(obj, dict) else None
