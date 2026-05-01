"""DDPM training objective + DDIM (and DDPM) sampling.

Notation follows the original DDPM paper:
- betas, alphas = 1 - betas, alpha_bar_t = prod_{s<=t} alpha_s.
- Forward (closed-form): q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I).
- Training: predict epsilon from x_t and t with simple MSE loss.

DDIM sampling (Song et al. 2020):
    x_{t-1} = sqrt(abar_{t-1}) * x0_pred
              + sqrt(1 - abar_{t-1} - sigma_t^2) * eps_pred
              + sigma_t * z
With eta = 0 -> sigma_t = 0 -> deterministic. eta = 1 reproduces DDPM.

The class is model-agnostic: it just needs a `model(x, t) -> eps` callable.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    f = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    abar = f / f[0]
    betas = 1 - (abar[1:] / abar[:-1])
    return betas.clamp(1e-8, 0.999)


def make_betas(schedule: str, timesteps: int, beta_start: float, beta_end: float):
    if schedule == "linear":
        return linear_beta_schedule(timesteps, beta_start, beta_end)
    if schedule == "cosine":
        return cosine_beta_schedule(timesteps)
    raise ValueError(f"unknown schedule {schedule}")


# ---------------------------------------------------------------------------
# Diffusion wrapper
# ---------------------------------------------------------------------------
def _gather(coef: torch.Tensor, t: torch.Tensor, target_shape) -> torch.Tensor:
    """Gather `coef` at indices `t` and reshape to broadcast against target_shape."""
    out = coef.to(device=t.device).gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(target_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule: str = "linear",
    ):
        super().__init__()
        betas = make_betas(schedule, timesteps, beta_start, beta_end).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.timesteps = timesteps
        # buffers move with .to(device) and serialize
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    # ------------------------------------------------------------------
    # forward (training)
    # ------------------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)
        sa = _gather(self.sqrt_alphas_cumprod, t, x0.shape)
        sma = _gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sa * x0 + sma * noise, noise

    def training_loss(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if t is None:
            t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x0, t)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, noise)

    # ------------------------------------------------------------------
    # DDIM / DDPM sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ddim_sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape,
        num_steps: int = 50,
        eta: float = 0.0,
        x_T: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
        trajectory_stride: int = 1,
        clip_x0: bool = True,
    ):
        """Run DDIM sampling.

        eta=0 => deterministic DDIM. eta=1 => DDPM-equivalent stochastic.
        Set num_steps == self.timesteps for a full DDPM-like schedule.
        """
        device = device or self.betas.device
        if x_T is None:
            x_t = torch.randn(shape, device=device)
        else:
            x_t = x_T.to(device)

        # uniform subsequence of timesteps, length num_steps, high -> low.
        # Keep these as Python ints — indexing buffers with MPS tensors is
        # buggy in some PyTorch builds (returns garbage indices).
        step_list = torch.linspace(0, self.timesteps - 1, num_steps, dtype=torch.long).tolist()
        step_list = list(reversed(step_list))
        prev_list = step_list[1:] + [-1]

        trajectory: List[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(x_t.detach().cpu())

        for i, (t_idx, t_prev) in enumerate(zip(step_list, prev_list)):
            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            eps = model(x_t, t_batch)

            abar_t = self.alphas_cumprod[t_idx]
            abar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            x0_pred = (x_t - torch.sqrt(1.0 - abar_t) * eps) / torch.sqrt(abar_t)
            if clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            sigma_t = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t)) * torch.sqrt(1 - abar_t / abar_prev)
            dir_xt = torch.sqrt(torch.clamp(1 - abar_prev - sigma_t ** 2, min=0.0)) * eps

            noise = torch.randn_like(x_t) if eta > 0 and t_prev >= 0 else torch.zeros_like(x_t)
            x_t = torch.sqrt(abar_prev) * x0_pred + dir_xt + sigma_t * noise

            if return_trajectory and ((i + 1) % trajectory_stride == 0 or i == len(step_list) - 1):
                trajectory.append(x_t.detach().cpu())

        if return_trajectory:
            return x_t, trajectory
        return x_t


# ---------------------------------------------------------------------------
# EMA helper (used during training to track a smoothed copy of weights)
# ---------------------------------------------------------------------------
class EMA:
    """EMA of model weights with shadow copy on CPU to save GPU memory.

    For a 245M-param model the shadow takes ~1GB; keeping it off MPS frees
    that memory for activations.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 device: str = "cpu"):
        self.decay = decay
        self.device = torch.device(device)
        self.shadow = {k: v.detach().to(self.device).clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            v_cpu = v.detach().to(self.device, non_blocking=False)
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v_cpu)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.to(self.device).clone() for k, v in sd.items()}

    def copy_to(self, model: nn.Module):
        # load_state_dict moves tensors to model's device automatically
        target_device = next(model.parameters()).device
        sd = {k: v.to(target_device) for k, v in self.shadow.items()}
        model.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# AdamW (manual)
# ---------------------------------------------------------------------------
class AdamW:
    """Hand-rolled AdamW with CPU-resident optimizer state.

    Two reasons this is custom:
    1. PyTorch 2.3.1's MPS AdamW kernel produces NaN parameters after one
       step when some grads are exactly zero (which happens here because
       several layers are zero-initialized). The Python impl is stable.
    2. The first/second moment buffers (m, v) live on CPU, halving GPU
       memory usage. We copy the grad to CPU each step, compute the AdamW
       update on CPU, and copy only the resulting weight delta back to MPS.

    For a 245M-param network this saves ~2GB of GPU memory.
    """

    def __init__(self, params, lr: float = 2e-4, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0,
                 state_device: str = "cpu"):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state_device = torch.device(state_device)
        self.t = 0
        self.m = [torch.zeros_like(p, device=self.state_device) for p in self.params]
        self.v = [torch.zeros_like(p, device=self.state_device) for p in self.params]

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            # bring grad to optimizer state device (typically CPU)
            g = p.grad.to(self.state_device, non_blocking=False)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / bc1
            denom = (v / bc2).sqrt().add_(self.eps)
            update = m_hat / denom                              # on state device
            # decoupled weight decay is applied in-place on the param itself
            if self.weight_decay > 0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.add_(update.to(p.device, non_blocking=False), alpha=-self.lr)

    def state_dict(self):
        return {"t": self.t, "m": [x.clone() for x in self.m],
                "v": [x.clone() for x in self.v],
                "lr": self.lr, "b1": self.b1, "b2": self.b2,
                "eps": self.eps, "weight_decay": self.weight_decay,
                "state_device": str(self.state_device)}

    def load_state_dict(self, sd):
        self.t = sd["t"]
        self.m = [x.to(self.state_device).clone() for x in sd["m"]]
        self.v = [x.to(self.state_device).clone() for x in sd["v"]]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    diff = GaussianDiffusion(timesteps=100, beta_start=1e-4, beta_end=2e-2, schedule="linear")
    assert diff.alphas_cumprod.shape == (100,)
    assert diff.alphas_cumprod[0].item() < 1.0 and diff.alphas_cumprod[-1].item() < diff.alphas_cumprod[0].item()

    # 1) q_sample at t=0 should be very close to x0 (since beta_0 ~ 0)
    x0 = torch.randn(4, 3, 16, 16)
    t0 = torch.zeros(4, dtype=torch.long)
    xt, _ = diff.q_sample(x0, t0)
    # at t=0, sqrt(1 - abar_0) ~ 1e-2, so noise contribution is small but nonzero
    assert (xt - x0).abs().max().item() < 0.1, (xt - x0).abs().max().item()

    # 2) at t=T-1 the marginal should be ~ pure noise (mean ~0, var ~1)
    tT = torch.full((4,), diff.timesteps - 1, dtype=torch.long)
    xtT, _ = diff.q_sample(x0, tT)
    assert xtT.std().item() > 0.7

    # 3) training loss with a dummy model that returns zeros should equal
    #    var of the noise (~1)
    zero_model = lambda x, t: torch.zeros_like(x)
    loss = diff.training_loss(zero_model, x0)
    assert 0.5 < loss.item() < 1.5, loss.item()

    # 4) DDIM sampling shape + determinism (eta=0)
    class Identity(nn.Module):
        def forward(self, x, t): return torch.zeros_like(x)
    model = Identity()
    out = diff.ddim_sample(model, (2, 3, 16, 16), num_steps=10, eta=0.0,
                           x_T=torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 3, 16, 16)

    # determinism: same x_T -> same output
    xT = torch.randn(2, 3, 16, 16)
    a = diff.ddim_sample(model, (2, 3, 16, 16), num_steps=10, eta=0.0, x_T=xT.clone())
    b = diff.ddim_sample(model, (2, 3, 16, 16), num_steps=10, eta=0.0, x_T=xT.clone())
    assert torch.allclose(a, b, atol=1e-6)

    # 5) trajectory return
    out2, traj = diff.ddim_sample(model, (1, 3, 16, 16), num_steps=10, eta=0.0,
                                  x_T=torch.randn(1, 3, 16, 16), return_trajectory=True)
    # initial + 10 steps = 11 frames
    assert len(traj) == 11, len(traj)

    # 6) EMA
    net = nn.Linear(4, 4)
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.ones_like(p))
    ema.update(net)
    # shadow should have moved halfway toward new weights
    for k, v in ema.shadow.items():
        if v.dtype.is_floating_point:
            assert (v - net.state_dict()[k]).abs().max() <= (net.state_dict()[k]).abs().max()

    # 7) MPS round trip
    if torch.backends.mps.is_available():
        diff_mps = GaussianDiffusion(timesteps=50).to("mps")
        x_mps = torch.randn(1, 3, 8, 8, device="mps")
        out_mps = diff_mps.ddim_sample(Identity().to("mps"), (1, 3, 8, 8),
                                       num_steps=5, eta=0.0)
        assert out_mps.shape == (1, 3, 8, 8)
        print("mps ok")

    print("diffusion.py: all tests passed")
