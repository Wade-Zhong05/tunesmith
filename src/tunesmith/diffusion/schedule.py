"""Gaussian diffusion: noising for training, DDIM sampling with guidance, masks and inpainting.

Forward process (Ho et al., 2020), cosine noise schedule (Nichol & Dhariwal, 2021):

    x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)

The U-Net is trained to predict eps (mean squared error). Sampling uses DDIM (Song et al.,
2021) with a few dozen steps. At every step:

1. guidance: eps = eps_uncond + s * (eps_cond - eps_uncond)  (Ho & Salimans, 2022);
2. the clean-roll estimate x_0 is clipped to [-1, 1];
3. constraints: cells outside the allowed mask (unused instrument slots, pitches outside an
   instrument's range, refused drums) are set to "off";
4. inpainting: cells the caller wants to keep are replaced by the given roll, which is how
   a single bar can be regenerated or an accompaniment written for a fixed melody
   (the "replacement" method; Song et al., 2021, appendix);
5. eps is recomputed from the edited x_0, so the DDIM update stays consistent.

AI attribution: drafted with Claude Code (Claude Opus 5.5), 2026-11-23. See ATTRIBUTION.md.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alphas_cumprod(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class Diffusion:
    def __init__(self, timesteps: int = 1000):
        self.T = int(timesteps)
        self.abar = cosine_alphas_cumprod(self.T)

    def _abar(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.abar.to(x.device)[t].view(-1, *([1] * (x.dim() - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self._abar(t, x0)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def loss(self, model: nn.Module, x0: torch.Tensor, cond: torch.Tensor,
             generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Noise-prediction MSE at uniformly drawn steps. With a generator the steps and the
        noise are reproducible (used for the validation loss)."""
        B = x0.size(0)
        if generator is None:
            t = torch.randint(0, self.T, (B,), device=x0.device)
            noise = torch.randn_like(x0)
        else:
            t = torch.randint(0, self.T, (B,), generator=generator).to(x0.device)
            noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        pred = model(self.q_sample(x0, t, noise), t, cond)
        return F.mse_loss(pred.float(), noise)

    @torch.no_grad()
    def sample(self, model: nn.Module, cond: torch.Tensor, null_cond: Optional[torch.Tensor], shape: tuple,
               steps: int = 50, cfg_scale: float = 1.0, allowed: Optional[torch.Tensor] = None,
               known: Optional[torch.Tensor] = None, known_mask: Optional[torch.Tensor] = None,
               eta: float = 0.0, generator: Optional[torch.Generator] = None,
               autocast: Optional[torch.dtype] = None,
               callback: Optional[Callable[[int, torch.Tensor], None]] = None) -> torch.Tensor:
        """DDIM sampling. Returns x_0 in [-1, 1], shape `shape` (B, C, F, P).

        cond, null_cond: (B, n_fields) control ids; null_cond is used for guidance.
        allowed:     bool, broadcastable to `shape`; False cells are forced off (-1).
        known:       roll in [-1, 1] to keep where known_mask is True (inpainting).
        """
        device = cond.device
        model.eval()

        def randn(s):
            return torch.randn(s, generator=generator).to(device)

        x = randn(shape)
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        use_cfg = cfg_scale != 1.0 and null_cond is not None
        abar = self.abar.to(device)
        x0 = x
        for i, t in enumerate(ts):
            a_t = abar[t]
            a_prev = abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            if known is not None and known_mask is not None:
                x = torch.where(known_mask, a_t.sqrt() * known + (1 - a_t).sqrt() * randn(shape), x)
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            with torch.autocast(device.type, dtype=autocast, enabled=autocast is not None):
                if use_cfg:
                    both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([cond, null_cond]))
                    e_c, e_u = both.float().chunk(2)
                    eps = e_u + cfg_scale * (e_c - e_u)
                else:
                    eps = model(x, tt, cond).float()
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            if allowed is not None:
                x0 = torch.where(allowed, x0, torch.full_like(x0, -1.0))
            if known is not None and known_mask is not None:
                x0 = torch.where(known_mask, known, x0)
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            if i + 1 < len(ts):
                sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).clamp(min=0).sqrt()
                x = a_prev.sqrt() * x0 + (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
                if eta > 0:
                    x = x + sigma * randn(shape)
            if callback is not None:
                callback(i, x0)
        return x0
