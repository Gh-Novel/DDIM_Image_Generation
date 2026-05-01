"""Inference: load a checkpoint and generate samples / trajectory / interp grid.

Usage:
    # 16 random faces with DDIM 50 steps
    python3 sample.py --ckpt checkpoints/stage-256_best.pt --num 16 --steps 50

    # save denoising trajectory as a GIF
    python3 sample.py --ckpt checkpoints/stage-256_best.pt --trajectory \
        --num 4 --steps 50 --out samples/traj.gif

    # interpolate between two random latents (8 frames, slerp)
    python3 sample.py --ckpt checkpoints/stage-256_best.pt --interpolate 8 \
        --out samples/interp.png

    # DDPM-1000 vs DDIM-50 side-by-side
    python3 sample.py --ckpt ... --compare-ddpm --num 4
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from config import Config
from models.unet import UNet
from models.diffusion import GaussianDiffusion, EMA
from utils.visualize import (save_image_grid, trajectory_to_gif,
                             interpolate_latents, make_grid)
from PIL import Image


# ---------------------------------------------------------------------------
def load_run(ckpt_path: str, device: torch.device, prefer_ema: bool = True):
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload["config"]
    cfg = Config(**cfg_dict)
    model = UNet(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        time_embed_dim=cfg.time_embed_dim,
        dropout=cfg.dropout,
    ).to(device)
    if prefer_ema and payload.get("ema") is not None:
        model.load_state_dict(payload["ema"], strict=True)
        print("[sample] loaded EMA weights")
    else:
        model.load_state_dict(payload["model"], strict=True)
        print("[sample] loaded raw weights")
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=cfg.timesteps, beta_start=cfg.beta_start,
        beta_end=cfg.beta_end, schedule=cfg.beta_schedule,
    ).to(device)
    return cfg, model, diffusion


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--device", type=str, default=None)
    # mode flags
    p.add_argument("--trajectory", action="store_true",
                   help="save denoising trajectory as a GIF")
    p.add_argument("--interpolate", type=int, default=0,
                   help="number of interpolation frames between two latents")
    p.add_argument("--compare-ddpm", action="store_true",
                   help="generate DDIM-N vs DDPM-T side-by-side comparison")
    return p.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    cfg, model, diffusion = load_run(args.ckpt, device, prefer_ema=not args.no_ema)
    print(f"[sample] image_size={cfg.image_size} run={cfg.run_name} device={device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    shape = (args.num, cfg.in_channels, cfg.image_size, cfg.image_size)
    out_dir = cfg.sample_dir
    os.makedirs(out_dir, exist_ok=True)

    # ---- interpolation -------------------------------------------------
    if args.interpolate > 0:
        n = args.interpolate
        z1 = torch.randn(1, *shape[1:], device=device)
        z2 = torch.randn(1, *shape[1:], device=device)
        latents = interpolate_latents(z1.cpu(), z2.cpu(), num_steps=n).squeeze(1).to(device)
        # latents shape: (n, C, H, W). One sampling pass per frame.
        with torch.no_grad():
            samples = diffusion.ddim_sample(
                model, (n, *shape[1:]), num_steps=args.steps, eta=args.eta,
                x_T=latents, device=device,
            )
        out = args.out or os.path.join(out_dir, f"interp_{n}.png")
        save_image_grid(samples.cpu(), out, nrow=n)
        print(f"[sample] interpolation saved -> {out}")
        return

    # ---- trajectory GIF ------------------------------------------------
    if args.trajectory:
        x_T = torch.randn(shape, device=device)
        with torch.no_grad():
            _, traj = diffusion.ddim_sample(
                model, shape, num_steps=args.steps, eta=args.eta,
                x_T=x_T, device=device,
                return_trajectory=True, trajectory_stride=1,
            )
        out = args.out or os.path.join(out_dir, f"traj_{args.steps}.gif")
        trajectory_to_gif(traj, out, fps=10)
        print(f"[sample] trajectory saved -> {out}")
        return

    # ---- DDIM vs DDPM comparison --------------------------------------
    if args.compare_ddpm:
        x_T = torch.randn(shape, device=device)
        with torch.no_grad():
            ddim = diffusion.ddim_sample(model, shape, num_steps=args.steps,
                                         eta=args.eta, x_T=x_T.clone(), device=device)
            ddpm = diffusion.ddim_sample(model, shape, num_steps=cfg.timesteps,
                                         eta=1.0, x_T=x_T.clone(), device=device)
        # stack as 2 rows
        side = torch.cat([ddim.cpu(), ddpm.cpu()], dim=0)
        out = args.out or os.path.join(out_dir, f"compare_ddim{args.steps}_vs_ddpm.png")
        save_image_grid(side, out, nrow=args.num)
        print(f"[sample] comparison saved -> {out}  (top: DDIM-{args.steps}, bottom: DDPM-{cfg.timesteps})")
        return

    # ---- default: simple grid -----------------------------------------
    with torch.no_grad():
        samples = diffusion.ddim_sample(
            model, shape, num_steps=args.steps, eta=args.eta, device=device,
        )
    out = args.out or os.path.join(out_dir, f"samples_n{args.num}_s{args.steps}.png")
    save_image_grid(samples.cpu(), out)
    print(f"[sample] grid saved -> {out}")


if __name__ == "__main__":
    main()
