"""LoRA supervised fine-tuning of the text -> attributes parser.

    pixi run download-parser-base
    pixi run make-parser-data
    pixi run -e cuda train-parser --config configs/parser_lora.yaml

Each example is a chat: system instruction + user description -> assistant JSON.
Loss is computed on the assistant answer only (prompt tokens are masked with -100).
Only LoRA adapter weights are trained; the base model stays frozen. Validation loss
and field accuracy (by greedy decoding a subset) are logged; the adapter with the best
validation field accuracy is saved to out_dir.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..attributes import MusicAttributes
from ..eval.parser_metrics import field_accuracy
from ..model import pick_device
from ..paths import load_yaml, read_jsonl, resolve, save_json
from ..train.train_composer import lr_lambda
from .lm_parser import model_dtype
from .prompts import build_messages, extract_json, target_json

IGNORE = -100


def encode_example(tok, text: str, target: dict[str, Any], max_len: int) -> dict[str, list[int]]:
    prompt = tok.apply_chat_template(build_messages(text, "lora"), tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(target_json(target), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (p_ids + a_ids)[:max_len]
    labels = ([IGNORE] * len(p_ids) + a_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


def pad_batch(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def decode_accuracy(model, tok, rows: list[dict], device, max_new_tokens: int = 96) -> dict[str, float]:
    model.eval()
    tok.padding_side = "left"
    preds, golds = [], []
    for i in range(0, len(rows), 8):
        chunk = rows[i:i + 8]
        prompts = [tok.apply_chat_template(build_messages(r["text"], "lora"), tokenize=False, add_generation_prompt=True)
                   for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for r, row in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            obj = extract_json(tok.decode(row, skip_special_tokens=True))
            preds.append(MusicAttributes.lenient(obj) if obj else None)
            golds.append(MusicAttributes.lenient(r["target"]))
    tok.padding_side = "right"
    model.train()
    return field_accuracy(preds, golds)


def train(cfg: dict[str, Any], device_name: str = "auto") -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_cfg, l_cfg = cfg["train"], cfg["lora"]
    random.seed(int(t_cfg["seed"]))
    torch.manual_seed(int(t_cfg["seed"]))
    device = pick_device(device_name)
    base = resolve(cfg["base_dir"])
    source = str(base) if base.exists() else cfg["base_model"]
    out_dir = resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(source, dtype=model_dtype(device)).to(device)
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=int(l_cfg["r"]), lora_alpha=int(l_cfg["alpha"]),
                                             lora_dropout=float(l_cfg["dropout"]), target_modules=list(l_cfg["target_modules"])))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {trainable / 1e6:.2f}M of {total / 1e6:.1f}M ({100 * trainable / total:.2f}%), device {device}")

    data_dir = resolve(cfg["data_dir"])
    train_rows = list(read_jsonl(data_dir / "train.jsonl"))
    val_rows = list(read_jsonl(data_dir / "val.jsonl"))
    if t_cfg.get("max_train_examples"):
        random.shuffle(train_rows)
        train_rows = train_rows[: int(t_cfg["max_train_examples"])]
    max_len = int(t_cfg["max_len"])
    train_enc = [encode_example(tok, r["text"], r["target"], max_len) for r in train_rows]
    val_enc = [encode_example(tok, r["text"], r["target"], max_len) for r in val_rows]
    col = partial(pad_batch, pad_id=tok.pad_token_id)
    bs = int(t_cfg["batch_size"])
    train_dl = DataLoader(train_enc, batch_size=bs, shuffle=True, collate_fn=col)
    val_dl = DataLoader(val_enc, batch_size=bs, shuffle=False, collate_fn=col)

    accum = int(t_cfg["grad_accum"])
    steps_total = max(1, math.ceil(len(train_dl) * int(t_cfg["epochs"]) / accum))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(t_cfg["lr"]),
                            weight_decay=float(t_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, warmup=int(t_cfg["warmup_steps"]),
                                                           total=steps_total, min_ratio=0.0))
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["step", "lr", "train_loss", "val_loss", "val_field_acc", "elapsed_s"])

    def log(**kw):
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([kw.get(k, "") for k in ("step", "lr", "train_loss", "val_loss", "val_field_acc", "elapsed_s")])

    @torch.no_grad()
    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        for b in val_dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(**b)
            k = int((b["labels"][:, 1:] != IGNORE).sum())
            tot, n = tot + float(out.loss) * k, n + k
        model.train()
        return tot / max(1, n)

    gen_rows = val_rows[: int(t_cfg["eval_gen_samples"])]
    best_acc, step, t0 = -1.0, 0, time.time()
    running, running_n = 0.0, 0
    model.train()
    for epoch in range(int(t_cfg["epochs"])):
        for i, b in enumerate(train_dl):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss / accum
            loss.backward()
            running, running_n = running + float(loss.detach()) * accum, running_n + 1
            if (i + 1) % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(t_cfg["grad_clip"]))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % int(t_cfg["log_every"]) == 0:
                print(f"epoch {epoch} step {step}/{steps_total} loss {running / running_n:.4f}", flush=True)
                log(step=step, lr=f"{sched.get_last_lr()[0]:.3g}", train_loss=f"{running / running_n:.5f}",
                    elapsed_s=f"{time.time() - t0:.0f}")
                running, running_n = 0.0, 0
            if step % int(t_cfg["eval_every"]) == 0 or step == steps_total:
                vl = val_loss()
                acc = decode_accuracy(model, tok, gen_rows, device)
                print(f"  val loss {vl:.4f}  field acc {acc['mean_field_accuracy']:.3f}", flush=True)
                log(step=step, val_loss=f"{vl:.5f}", val_field_acc=f"{acc['mean_field_accuracy']:.4f}",
                    elapsed_s=f"{time.time() - t0:.0f}")
                if acc["mean_field_accuracy"] > best_acc:
                    best_acc = acc["mean_field_accuracy"]
                    model.save_pretrained(str(out_dir))
                    tok.save_pretrained(str(out_dir))
                    save_json({"step": step, "val_loss": vl, **acc}, out_dir / "best_metrics.json")
    if best_acc < 0:  # never evaluated (tiny runs)
        model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
    summary = {"steps": step, "best_val_field_accuracy": best_acc, "minutes": (time.time() - t0) / 60,
               "trainable_params": trainable}
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/parser_lora.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train-examples", type=int)
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.max_train_examples:
        cfg["train"]["max_train_examples"] = args.max_train_examples
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    train(cfg, args.device)


if __name__ == "__main__":
    main()
