"""DDPM training loop with W&B logging, EMA, auto-resume from checkpoint.

Usage:
    python3 train.py --image-size 64 --epochs 50
    python3 train.py --image-size 256 --epochs 200 --resume

The checkpoint policy: write `latest.pt` every epoch and `best.pt` when
the running epoch loss improves. Auto-resume looks for `latest.pt` under
the configured ckpt_dir and loads model + optimizer + EMA + epoch + step.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from typing import Optional

# Apple Silicon: a few ops still fall back to CPU. Enable that by default.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config, get_default_config
from models.unet import UNet
from models.diffusion import GaussianDiffusion, EMA, AdamW
from utils.dataset import make_dataloader, denormalize
from utils.visualize import make_grid, save_image_grid


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> UNet:
    return UNet(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        time_embed_dim=cfg.time_embed_dim,
        dropout=cfg.dropout,
    )


def build_diffusion(cfg: Config) -> GaussianDiffusion:
    return GaussianDiffusion(
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        schedule=cfg.beta_schedule,
    )


def latest_ckpt_path(cfg: Config) -> str:
    return os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_latest.pt")


def best_ckpt_path(cfg: Config) -> str:
    return os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")


def save_checkpoint(path: str, *, model, optimizer, ema, epoch, step, best_loss, cfg: Config):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "step": step,
        "best_loss": best_loss,
        "config": cfg.to_dict(),
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, *, model, optimizer, ema, device):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return payload


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train(cfg: Config, args):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    print(f"[train] device={device} run={cfg.run_name} image={cfg.image_size}")

    # ---- data --------------------------------------------------------
    loader = make_dataloader(
        root=cfg.data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        augment=True,
        limit=args.limit,
    )
    print(f"[train] dataset images={len(loader.dataset)} batches/epoch={len(loader)}")

    # ---- model + diffusion ------------------------------------------
    model = build_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, decay=cfg.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params={n_params/1e6:.1f}M")

    # ---- resume ------------------------------------------------------
    start_epoch = 0
    global_step = 0
    best_loss = math.inf
    ckpt = latest_ckpt_path(cfg)
    if args.resume and os.path.isfile(ckpt):
        payload = load_checkpoint(ckpt, model=model, optimizer=optimizer, ema=ema, device=device)
        start_epoch = payload.get("epoch", 0) + 1
        global_step = payload.get("step", 0)
        best_loss = payload.get("best_loss", math.inf)
        print(f"[train] resumed from {ckpt} at epoch={start_epoch} step={global_step}")

    # ---- W&B ---------------------------------------------------------
    use_wandb = cfg.use_wandb and not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=cfg.to_dict(),
                resume="allow",
            )
        except Exception as e:                                     # noqa: BLE001
            print(f"[train] wandb disabled ({e})")
            use_wandb = False

    # ---- graceful shutdown so we always save latest ------------------
    interrupted = {"flag": False}
    def _on_sig(signum, frame):
        interrupted["flag"] = True
        print("\n[train] caught signal, finishing batch then saving checkpoint...")
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    # ---- training ----------------------------------------------------
    sample_shape = (min(16, cfg.batch_size), cfg.in_channels, cfg.image_size, cfg.image_size)
    fixed_noise = torch.randn(sample_shape, device=device)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            batch = batch.to(device, non_blocking=True)
            loss = diffusion.training_loss(model, batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            global_step += 1
            loss_v = loss.item()
            epoch_loss += loss_v
            epoch_count += 1
            pbar.set_postfix(loss=f"{loss_v:.4f}", step=global_step)

            if use_wandb and global_step % cfg.log_every == 0:
                import wandb
                wandb.log({"loss": loss_v, "epoch": epoch}, step=global_step)

            if interrupted["flag"]:
                break

        avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"[train] epoch {epoch} avg_loss={avg_loss:.4f}")
        if use_wandb:
            import wandb
            wandb.log({"epoch_loss": avg_loss, "epoch": epoch}, step=global_step)

        # ---- sample grid at every Nth epoch -------------------------
        if (epoch + 1) % cfg.sample_every_epochs == 0 or epoch == 0:
            model.eval()
            ema_model = build_model(cfg).to(device)
            ema.copy_to(ema_model)
            ema_model.eval()
            with torch.no_grad():
                samples = diffusion.ddim_sample(
                    ema_model, sample_shape, num_steps=cfg.ddim_steps, eta=cfg.ddim_eta,
                    x_T=fixed_noise.clone(), device=device,
                )
            sample_path = os.path.join(cfg.sample_dir,
                                       f"{cfg.run_name}_epoch{epoch:04d}.png")
            save_image_grid(samples.cpu(), sample_path, nrow=4)
            if use_wandb:
                import wandb
                wandb.log({"samples": wandb.Image(sample_path)}, step=global_step)

        # ---- checkpoint --------------------------------------------
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or interrupted["flag"]:
            save_checkpoint(
                latest_ckpt_path(cfg),
                model=model, optimizer=optimizer, ema=ema,
                epoch=epoch, step=global_step,
                best_loss=best_loss, cfg=cfg,
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    best_ckpt_path(cfg),
                    model=model, optimizer=optimizer, ema=ema,
                    epoch=epoch, step=global_step,
                    best_loss=best_loss, cfg=cfg,
                )

        if interrupted["flag"]:
            print("[train] saved and exiting")
            break

    if wandb_run is not None:
        wandb_run.finish()


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-size", type=int, default=64, choices=[64, 128, 256])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="cap dataset size (smoke tests)")
    p.add_argument("--resume", action="store_true",
                   help="auto-load <run>_latest.pt if present")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {}
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.lr is not None: overrides["lr"] = args.lr
    if args.num_workers is not None: overrides["num_workers"] = args.num_workers
    if args.run_name is not None: overrides["run_name"] = args.run_name
    cfg = Config.for_stage(args.image_size, **overrides)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)
    train(cfg, args)


if __name__ == "__main__":
    main()
