"""Train the diffusion composer on piano rolls of the prepared ensemble windows.

    pixi run -e cuda train-diffusion                                       # configs/diffusion_ensemble.yaml
    pixi run train-diffusion --config configs/diffusion_tiny.yaml          # CPU smoke run
    ... --resume                                                          # continue from last.pt

Writes to the config's out_dir:
- best.pt: EMA weights with the lowest validation loss (what the sampler loads);
- last.pt: raw and EMA weights with optimizer state, for resuming;
- log.csv, curves.png, the config used;
- samples/: a few pieces written at every evaluation, to listen to the model as it trains.

The validation loss uses fixed diffusion steps and noise, so values are comparable across
evaluations. It is a noise-prediction MSE, not comparable to the Transformer's
cross-entropy; the generated music is compared with the same metrics instead
(`pixi run evaluate --system <best.pt>`).

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ..attributes import MusicAttributes
from ..model import pick_device
from ..paths import load_json, load_yaml, resolve
from ..tokenizer import TuneTokenizer
from ..train.train_composer import amp_dtype, forever, lr_lambda
from .composer import DiffusionComposer, save_diffusion_checkpoint
from .data import RollDataset
from .roll import RollSpec
from .schedule import Diffusion
from .unet import RollUNet, UNetConfig

LOG_FIELDS = ("step", "lr", "train_loss", "val_loss", "samples_per_s", "elapsed_s")
# Requests written to samples/ at every evaluation.
PREVIEW = [
    MusicAttributes(instruments=["piano", "electric_guitar", "bass"], drums=True, genre="pop", mood="happy",
                    tempo_bpm=120, key="C major"),
    MusicAttributes(instruments=["strings", "piano"], drums=False, mood="calm", genre="classical",
                    tempo_bpm=80, key="A minor"),
]


def to_model_range(roll: torch.Tensor) -> torch.Tensor:
    return roll.float() * 2 - 1


@torch.no_grad()
def evaluate(model, diffusion: Diffusion, loader, device, dtype, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total, n = 0.0, 0
    for i, (roll, cond) in enumerate(loader):
        if i >= max_batches:
            break
        x0, cond = to_model_range(roll.to(device)), cond.to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = diffusion.loss(model, x0, cond, generator=gen)
        total += float(loss) * x0.size(0)
        n += x0.size(0)
    model.train(was_training)
    return total / max(1, n)


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
        ax.plot(*zip(*va), "o-", label="validation loss (EMA weights)")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("noise-prediction MSE (log scale)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train(cfg: dict[str, Any], resume: bool = False, device_name: str = "auto") -> dict[str, Any]:
    t_cfg, a_cfg = cfg["train"], cfg.get("augment") or {}
    data_dir, out_dir = resolve(cfg["data_dir"]), resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_json(data_dir / "meta.json")
    tok = TuneTokenizer(meta["tokenizer"])
    spec = RollSpec.from_dict(cfg.get("roll"))
    diffusion = Diffusion(int(cfg.get("diffusion", {}).get("timesteps", 1000)))

    seed = int(t_cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = pick_device(device_name)
    dtype = amp_dtype(str(t_cfg.get("amp", "auto")), device)
    ucfg = UNetConfig(**{**cfg.get("model", {}), "in_channels": spec.channels, "cond_vocab": tok.vocab_size,
                         "n_fields": len(tok.fields)})
    model = RollUNet(ucfg).to(device)
    if spec.frames % model.downsample_factor or spec.n_pitch % model.downsample_factor:
        raise ValueError(f"roll {spec.frames}x{spec.n_pitch} is not divisible by {model.downsample_factor}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = model.num_parameters()
    print(f"diffusion U-Net: {n_params / 1e6:.2f}M parameters, roll {spec.shape}, device {device}, amp {dtype}")

    train_ds = RollDataset(data_dir, "train", tok, spec, a_cfg, max_items=t_cfg.get("max_train_items"), seed=seed)
    val_ds = RollDataset(data_dir, "val", tok, spec, None, max_items=int(t_cfg.get("max_val_items", 2048)), seed=seed)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"no 4/4 windows: train {len(train_ds)}, val {len(val_ds)}")
    print(f"4/4 windows: train {len(train_ds):,}, val {len(val_ds):,}")
    bs, nw = int(t_cfg["batch_size"]), int(t_cfg.get("num_workers", 0))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) > bs, num_workers=nw,
                          persistent_workers=nw > 0, pin_memory=device.type == "cuda")
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=float(t_cfg["lr"]), weight_decay=float(t_cfg.get("weight_decay", 0.0)),
                            betas=tuple(t_cfg.get("betas", [0.9, 0.999])), fused=device.type == "cuda")
    max_steps = int(t_cfg["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, partial(lr_lambda, warmup=int(t_cfg["warmup_steps"]),
                                                           total=max_steps, min_ratio=float(t_cfg["min_lr_ratio"])))
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)
    ema_decay = float(t_cfg.get("ema_decay", 0.999))
    sample_cfg = cfg.get("sample", {})

    step, best_val = 0, float("inf")
    log_path = out_dir / "log.csv"
    if resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["raw_state"])
        ema.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed from step {step} (best val {best_val:.5f})")
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                rows = list(csv.reader(f))
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows[:1] + [r for r in rows[1:] if r and int(r[0]) <= step])
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_FIELDS)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "n_params": n_params}, f, sort_keys=False)

    def log_row(**kw):
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([kw.get(k, "") for k in LOG_FIELDS])

    def save(name: str, with_state: bool):
        extra = {"step": step, "best_val": best_val, "n_params": n_params, "sample": sample_cfg}
        if with_state:
            extra.update(raw_state=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict())
        save_diffusion_checkpoint(out_dir / name, ema, spec, tok, diffusion.T, extra)

    def write_samples():
        composer = DiffusionComposer(ema, tok, spec, diffusion, sample_cfg)
        scores = composer.compose(PREVIEW, seed=0)
        for k, score in enumerate(scores):
            path = out_dir / "samples" / f"step{step:06d}_{k}.mid"
            path.parent.mkdir(parents=True, exist_ok=True)
            score.dump_midi(str(path))
        print(f"  samples: {[sum(len(t.notes) for t in s.tracks) for s in scores]} notes", flush=True)

    accum = int(t_cfg.get("grad_accum", 1))
    it = forever(train_dl)
    model.train()
    t0, t_window, start_step = time.time(), time.time(), step
    window_loss, window_n, window_items = 0.0, 0, 0
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            roll, cond = next(it)
            x0, cond = to_model_range(roll.to(device, non_blocking=True)), cond.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                loss = diffusion.loss(model, x0, cond)
            scaler.scale(loss / accum).backward()
            window_loss += float(loss.detach())
            window_n += 1
            window_items += x0.size(0)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(t_cfg["grad_clip"]))
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        decay = min(ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)

        if step % int(t_cfg["log_every"]) == 0 or step == 1:
            dt = time.time() - t_window
            elapsed = time.time() - t0
            eta = (max_steps - step) * elapsed / max(1, step - start_step)
            print(f"step {step}/{max_steps}  loss {window_loss / window_n:.5f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{window_items / max(1e-9, dt):,.0f} rolls/s  eta {eta / 60:.1f} min", flush=True)
            log_row(step=step, lr=f"{sched.get_last_lr()[0]:.6g}", train_loss=f"{window_loss / window_n:.6f}",
                    samples_per_s=f"{window_items / max(1e-9, dt):.1f}", elapsed_s=f"{elapsed:.1f}")
            window_loss, window_n, window_items, t_window = 0.0, 0, 0, time.time()

        if step % int(t_cfg["eval_every"]) == 0 or step == max_steps:
            val = evaluate(ema, diffusion, val_dl, device, dtype, int(t_cfg["eval_batches"]))
            print(f"  val loss (EMA) {val:.5f}", flush=True)
            log_row(step=step, val_loss=f"{val:.6f}", elapsed_s=f"{time.time() - t0:.1f}")
            if val < best_val:
                best_val = val
                save("best.pt", with_state=False)
            if t_cfg.get("write_samples", True):
                write_samples()
        if step % int(t_cfg["save_every"]) == 0:
            save("last.pt", with_state=True)

    save("last.pt", with_state=True)
    plot_curves(log_path, out_dir / "curves.png")
    summary = {"steps": step, "best_val": best_val, "n_params": n_params, "minutes": (time.time() - t0) / 60}
    print(f"finished: {summary}")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/diffusion_ensemble.yaml")
    ap.add_argument("--data-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-steps", type=int)
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
    train(cfg, resume=args.resume, device_name=args.device)


if __name__ == "__main__":
    main()
