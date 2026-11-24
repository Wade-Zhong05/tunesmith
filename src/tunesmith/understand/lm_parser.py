"""LLM parser: a small instruct model (default Qwen2.5-0.5B-Instruct) maps text -> attributes.

Modes:
  zero  - schema in the system prompt, no examples (prompting baseline)
  few   - schema + 5 worked examples from configs/parser_fewshot.yaml (in-context learning)
  lora  - short instruction + our LoRA adapter (fine-tuned with train_parser.py)

Invalid JSON or invalid field values never crash: bad fields are dropped (`lenient`),
and `valid_json` in the result records whether the raw output parsed at all.

The text first goes through the rule parser's spelling repair (`rules.normalize_text`):
stretched or slightly misspelt keywords ("sloooow", "guita") are fixed, other words are
left alone. The repair changes no text in the parser test set.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ..attributes import MusicAttributes
from ..model import pick_device
from ..paths import resolve
from .prompts import build_messages, extract_json
from .rules import normalize_text


@dataclass
class ParseResult:
    attrs: MusicAttributes
    raw: str
    valid_json: bool


def model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class LMParser:
    def __init__(self, base: str | Path, adapter: Optional[str | Path] = None, mode: str = "lora",
                 device: str = "auto", max_new_tokens: int = 96):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_path = resolve(base)
        source = str(base_path) if base_path.exists() else str(base)  # local folder or hub id
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(source)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(source, dtype=model_dtype(self.device)).to(self.device)
        if mode == "lora":
            adapter_path = resolve(adapter) if adapter else None
            if adapter_path and adapter_path.exists():
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, str(adapter_path)).to(self.device)
            else:
                mode = "few"
        self.mode = mode
        self.name = f"lm-{mode}"
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def _prompt(self, text: str) -> str:
        return self.tok.apply_chat_template(build_messages(text, self.mode), tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def parse_batch(self, texts: list[str], batch_size: int = 8) -> list[ParseResult]:
        out: list[ParseResult] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tok([self._prompt(normalize_text(t)) for t in chunk], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id)
            for row in gen[:, enc["input_ids"].shape[1]:]:
                raw = self.tok.decode(row, skip_special_tokens=True)
                obj = extract_json(raw)
                attrs = MusicAttributes.lenient(obj) if obj else MusicAttributes()
                out.append(ParseResult(attrs, raw, obj is not None))
        return out

    def parse(self, text: str) -> MusicAttributes:
        return self.parse_batch([text])[0].attrs
