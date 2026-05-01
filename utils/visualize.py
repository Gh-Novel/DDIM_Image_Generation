"""Visualization helpers: image grids, denoising-trajectory GIFs, and
latent-interpolation grids.

All functions accept tensors in the [-1, 1] range (model output convention)
unless otherwise stated, and write/return uint8 arrays in [0, 255].
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Small primitives
# ---------------------------------------------------------------------------
def to_uint8(x: torch.Tensor) -> np.ndarray:
    """Tensor in [-1, 1] (B,3,H,W) or (3,H,W) -> uint8 numpy (H,W,3) or (B,H,W,3)."""
    x = x.detach().to(torch.float32).cpu()
    x = (x.clamp(-1.0, 1.0) + 1.0) * 127.5
    x = x.round().clamp(0, 255).to(torch.uint8)
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).numpy()                # (B,H,W,3)
    if x.ndim == 3:
        return x.permute(1, 2, 0).numpy()                   # (H,W,3)
    raise ValueError(f"unsupported shape {x.shape}")


def make_grid(images: torch.Tensor, nrow: Optional[int] = None, pad: int = 2,
              pad_value: float = 1.0) -> np.ndarray:
    """Lay a batch of images out as a grid. Inputs in [-1, 1].

    Returns uint8 (H, W, 3).
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B,C,H,W), got {images.shape}")
    B, C, H, W = images.shape
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(B)))
    ncol = int(math.ceil(B / nrow))

    grid_h = ncol * H + (ncol + 1) * pad
    grid_w = nrow * W + (nrow + 1) * pad

    grid = torch.full((C, grid_h, grid_w), pad_value, dtype=images.dtype)
    for i in range(B):
        r, c = divmod(i, nrow)
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        grid[:, y:y + H, x:x + W] = images[i]

    return to_uint8(grid)


def save_image_grid(images: torch.Tensor, path: str, nrow: Optional[int] = None) -> str:
    arr = make_grid(images, nrow=nrow)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(arr).save(path)
    return path


# ---------------------------------------------------------------------------
# Denoising trajectory GIF
# ---------------------------------------------------------------------------
def trajectory_to_gif(
    trajectory: Sequence[torch.Tensor],
    path: str,
    fps: int = 10,
    nrow: Optional[int] = None,
) -> str:
    """Save a list of tensors (each (B,C,H,W) in [-1,1]) as an animated GIF.

    Each frame is laid out as a grid of all batch items.
    """
    import imageio.v2 as imageio       # local import; heavy dep

    frames = []
    for x in trajectory:
        if x.ndim == 3:
            x = x.unsqueeze(0)
        frames.append(make_grid(x, nrow=nrow))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(path, frames, format="GIF", duration=duration, loop=0)
    return path


# ---------------------------------------------------------------------------
# Latent interpolation
# ---------------------------------------------------------------------------
def slerp(z1: torch.Tensor, z2: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two same-shape latents.

    Falls back to lerp if vectors are nearly colinear (avoids div-by-zero).
    """
    flat1 = z1.flatten(start_dim=0)
    flat2 = z2.flatten(start_dim=0)
    dot = (flat1 * flat2).sum() / (flat1.norm() * flat2.norm() + 1e-12)
    dot = dot.clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-6:
        return (1 - t) * z1 + t * z2
    a = torch.sin((1 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    return a * z1 + b * z2


def interpolate_latents(z1: torch.Tensor, z2: torch.Tensor, num_steps: int = 8,
                        method: str = "slerp") -> torch.Tensor:
    """Return a tensor of shape (num_steps, *z1.shape) of interpolated latents."""
    ts = torch.linspace(0.0, 1.0, num_steps)
    out = []
    for t in ts:
        if method == "slerp":
            out.append(slerp(z1, z2, t.item()))
        elif method == "lerp":
            out.append((1 - t) * z1 + t * z2)
        else:
            raise ValueError(method)
    return torch.stack(out, dim=0)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    torch.manual_seed(0)
    imgs = torch.randn(8, 3, 32, 32).clamp(-1, 1)

    grid = make_grid(imgs, nrow=4)
    assert grid.dtype == np.uint8 and grid.ndim == 3 and grid.shape[2] == 3

    with tempfile.TemporaryDirectory() as td:
        p1 = save_image_grid(imgs, os.path.join(td, "g.png"))
        assert os.path.exists(p1)

        traj = [torch.randn(4, 3, 16, 16).clamp(-1, 1) for _ in range(6)]
        p2 = trajectory_to_gif(traj, os.path.join(td, "t.gif"), fps=8, nrow=2)
        assert os.path.exists(p2) and os.path.getsize(p2) > 0

    z1 = torch.randn(1, 3, 16, 16)
    z2 = torch.randn(1, 3, 16, 16)
    interps = interpolate_latents(z1, z2, num_steps=5, method="slerp")
    assert interps.shape == (5, 1, 3, 16, 16)
    # endpoints recovered
    assert torch.allclose(interps[0], z1, atol=1e-5)
    assert torch.allclose(interps[-1], z2, atol=1e-5)

    print("visualize.py: all tests passed")
