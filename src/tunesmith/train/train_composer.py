"""Train the composer (attributes -> melody) as a causal language model.

    pixi run -e cuda train-composer --config configs/composer_base.yaml   # Transformer, GPU
    pixi run -e cuda train-composer --config configs/composer_gru.yaml    # GRU baseline
    pixi run train-composer --config configs/composer_tiny.yaml           # CPU smoke run
    ... --resume                                                         # continue from last.pt

Writes to the config's out_dir: best.pt (lowest validation loss), last.pt (with
optimizer state, for resuming), log.csv, curves.png and the config used.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-10. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import IGNORE, TokenDataset, collate
from ..model import NotImplementedYet, build_model, pick_device, save_checkpoint
from ..paths import load_json, load_yaml, resolve
from ..tokenizer import TuneTokenizer


def lr_lambda(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warm-up, then cosine decay to min_ratio (multiplier of the base lr)."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def amp_dtype(setting: str, device: torch.device):
    if device.type != "cuda" or setting == "off":
        return None
    if setting == "bf16" or (setting == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=IGNORE)


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def forever(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


@torch.no_grad()
def evaluate(model, loader, device, dtype, max_batches: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            logits, _ = model(x)
        n = int((y != IGNORE).sum())
        total += float(lm_loss(logits, y)) * n
        count += n
    model.train()
    return total / max(1, count)


def plot_curves(log_path: Path, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    tr = [(int(r["step"]), float(r["train_loss"])) for r in rows if r["train_loss"]]
    va = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train loss", alpha=0.8)
    if va:
        ax.plot(*zip(*va), "o-", label="validation loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train(cfg: dict[str, Any], resume: bool = False, device_name: str = "auto") -> dict[str, Any]:
    t_cfg, a_cfg, m_cfg = cfg["train"], cfg.get("augment") or {}, cfg["model"]
    data_dir, out_dir = resolve(cfg["data_dir"]), resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(data_dir / "meta.json")
    tok = TuneTokenizer(meta["tokenizer"])
    if tok.vocab_size != meta["vocab_size"]:
        raise ValueError("tokenizer does not match the prepared data; re-run prepare-data")

    torch.manual_seed(int(t_cfg.get("seed", 0)))
    device = pick_device(device_name)
    dtype = amp_dtype(str(t_cfg.get("amp", "auto")), device)
    model = build_model(m_cfg, tok.vocab_size).to(device)
    n_params = model.num_parameters()
    print(f"model {m_cfg['type']}: {n_params / 1e6:.2f}M parameters, device {device}, amp {dtype}")

    train_ds = TokenDataset(data_dir, "train", tok, a_cfg)
    val_ds = TokenDataset(data_dir, "val", tok, None)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"empty split: train {len(train_ds)}, val {len(val_ds)}")
    col = partial(collate, pad_id=tok.pad_id, prompt_len=tok.prompt_len)
    nw = int(t_cfg.get("num_workers", 0))
    train_dl = DataLoader(train_ds, batch_size=int(t_cfg["batch_size"]), shuffle=True, drop_last=len(train_ds) > int(t_cfg["batch_size"]),
                          num_workers=nw, collate_fn=col, persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=int(t_cfg["batch_size"]), shuffle=False, num_workers=0, collate_fn=col)

    opt = torch.optim.AdamW(param_groups(model, float(t_cfg["weight_decay"])), lr=float(t_cfg["lr"]),
                            betas=tuple(t_cfg["betas"]), fused=device.type == "cuda")
    max_steps = int(t_cfg["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, warmup=int(t_cfg["warmup_steps"]),
                                                           total=max_steps, min_ratio=float(t_cfg["min_lr_ratio"])))
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    step, best_val, bad_evals = 0, float("inf"), 0
    log_path = out_dir / "log.csv"
    if resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, best_val, bad_evals = ck["step"], ck["best_val"], ck.get("bad_evals", 0)
        print(f"resumed from step {step} (best val {best_val:.4f})")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["step", "lr", "train_loss", "val_loss", "tokens_per_s", "elapsed_s"])
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "n_params": n_params}, f, sort_keys=False)

    def log_row(**kw):
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([kw.get(k, "") for k in ("step", "lr", "train_loss", "val_loss", "tokens_per_s", "elapsed_s")])

    def save(name: str, with_state: bool):
        extra = {"step": step, "best_val": best_val, "n_params": n_params}
        if with_state:
            extra.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), bad_evals=bad_evals)
        save_checkpoint(out_dir / name, model, m_cfg, tok, extra)

    accum = int(t_cfg.get("grad_accum", 1))
    it = forever(train_dl)
    model.train()
    t0, window_tokens, window_loss, window_n = time.time(), 0, 0.0, 0
    t_window = time.time()
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits, _ = model(x)
            loss = lm_loss(logits, y)
            scaler.scale(loss / accum).backward()
            window_tokens += int((y != IGNORE).sum())
            window_loss += float(loss.detach())
            window_n += 1
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t_cfg["grad_clip"]))
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1

        if step % int(t_cfg["log_every"]) == 0 or step == 1:
            dt = time.time() - t_window
            tps = window_tokens / max(1e-9, dt)
            elapsed = time.time() - t0
            eta = (max_steps - step) * elapsed / max(1, step)
            print(f"step {step}/{max_steps}  loss {window_loss / window_n:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{tps:,.0f} tok/s  eta {eta / 60:.1f} min", flush=True)
            log_row(step=step, lr=f"{sched.get_last_lr()[0]:.6g}", train_loss=f"{window_loss / window_n:.5f}",
                    tokens_per_s=f"{tps:.0f}", elapsed_s=f"{elapsed:.1f}")
            window_tokens, window_loss, window_n, t_window = 0, 0.0, 0, time.time()

        if step % int(t_cfg["eval_every"]) == 0 or step == max_steps:
            val = evaluate(model, val_dl, device, dtype, int(t_cfg["eval_batches"]))
            print(f"  val loss {val:.4f}  (ppl {math.exp(min(20, val)):.2f})", flush=True)
            log_row(step=step, val_loss=f"{val:.5f}", elapsed_s=f"{time.time() - t0:.1f}")
            if val < best_val:
                best_val, bad_evals = val, 0
                save("best.pt", with_state=False)
            else:
                bad_evals += 1
                if bad_evals >= int(t_cfg.get("early_stop_patience", 10**9)):
                    print(f"early stopping: no improvement for {bad_evals} evaluations")
                    break
        if step % int(t_cfg["save_every"]) == 0:
            save("last.pt", with_state=True)

    save("last.pt", with_state=True)
    plot_curves(log_path, out_dir / "curves.png")
    summary = {"steps": step, "best_val": best_val, "n_params": n_params, "minutes": (time.time() - t0) / 60}
    print(f"finished: {summary}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/composer_base.yaml")
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--model-type", choices=["transformer", "gru"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.max_steps:
        cfg["train"]["max_steps"] = args.max_steps
    if args.model_type:
        cfg["model"]["type"] = args.model_type
    try:
        train(cfg, resume=args.resume, device_name=args.device)
    except NotImplementedYet as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
