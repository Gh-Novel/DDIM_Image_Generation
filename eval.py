"""FID evaluation: generated samples vs the training distribution.

Computes Frechet Inception Distance with torchmetrics (Inception-V3, 2048-d
features). Real statistics are computed once and reused across every DDIM
step count, so comparing samplers is cheap.

Usage:
    # FID at several DDIM step counts (the speed/quality tradeoff table)
    python3 eval.py --ckpt checkpoints/stage-64_best.pt --steps 10 20 50

    # quick smoke run
    python3 eval.py --ckpt checkpoints/stage-64_best.pt --num-samples 256 \
        --num-real 256 --steps 20

Note: FID is sample-size dependent — always compare runs that used the same
--num-samples. 2048+ generated samples is a reasonable minimum for the
2048-d feature space; papers typically report FID-10k or FID-50k.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

from sample import load_run
from utils.dataset import CelebAHQ, denormalize


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float -> [0, 255] uint8, as the FID metric expects."""
    return (denormalize(x) * 255.0).round().to(torch.uint8)


def feed_real_images(fid, data_dir: str, image_size: int, num_real: int,
                     batch_size: int, device: torch.device):
    ds = CelebAHQ(data_dir, image_size=image_size, augment=False, limit=num_real)
    print(f"[eval] real statistics from {len(ds)} images in {data_dir}")
    for start in tqdm(range(0, len(ds), batch_size), desc="real", dynamic_ncols=True):
        batch = torch.stack([ds[i] for i in range(start, min(start + batch_size, len(ds)))])
        fid.update(to_uint8(batch).to(device), real=True)


def generate_and_feed(fid, model, diffusion, *, num_samples: int, steps: int,
                      eta: float, batch_size: int, image_size: int,
                      in_channels: int, model_device: torch.device,
                      fid_device: torch.device, seed: int) -> float:
    g = torch.Generator(device="cpu").manual_seed(seed)
    done = 0
    t0 = time.time()
    pbar = tqdm(total=num_samples, desc=f"ddim-{steps}", dynamic_ncols=True)
    while done < num_samples:
        n = min(batch_size, num_samples - done)
        shape = (n, in_channels, image_size, image_size)
        x_T = torch.randn(*shape, generator=g).to(model_device)
        with torch.no_grad():
            out = diffusion.ddim_sample(model, shape, num_steps=steps, eta=eta,
                                        x_T=x_T, device=model_device)
        fid.update(to_uint8(out.cpu()).to(fid_device), real=False)
        done += n
        pbar.update(n)
    pbar.close()
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data-dir", type=str, default=None,
                   help="real images dir (default: from checkpoint config)")
    p.add_argument("--steps", type=int, nargs="+", default=[10, 20, 50],
                   help="DDIM step counts to evaluate")
    p.add_argument("--num-samples", type=int, default=2048,
                   help="generated samples per step count")
    p.add_argument("--num-real", type=int, default=5000,
                   help="real images for reference statistics")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true", help="use raw weights")
    p.add_argument("--fid-device", type=str, default="cpu",
                   help="device for the Inception network (cpu is safest on MPS)")
    args = p.parse_args()

    from torchmetrics.image.fid import FrechetInceptionDistance

    if torch.backends.mps.is_available():
        model_device = torch.device("mps")
    elif torch.cuda.is_available():
        model_device = torch.device("cuda")
    else:
        model_device = torch.device("cpu")
    fid_device = torch.device(args.fid_device)

    cfg, model, diffusion = load_run(args.ckpt, model_device, prefer_ema=not args.no_ema)
    data_dir = args.data_dir or cfg.data_dir
    print(f"[eval] model={model_device} inception={fid_device} "
          f"image_size={cfg.image_size} eta={args.eta}")

    # reset_real_features=False lets us reset() between step counts while
    # keeping the (expensive) real statistics.
    fid = FrechetInceptionDistance(feature=2048, normalize=False,
                                   reset_real_features=False).to(fid_device)
    feed_real_images(fid, data_dir, cfg.image_size, args.num_real,
                     args.batch_size, fid_device)

    results = []
    for steps in args.steps:
        elapsed = generate_and_feed(
            fid, model, diffusion, num_samples=args.num_samples, steps=steps,
            eta=args.eta, batch_size=args.batch_size, image_size=cfg.image_size,
            in_channels=cfg.in_channels, model_device=model_device,
            fid_device=fid_device, seed=args.seed,
        )
        score = fid.compute().item()
        per_img = elapsed / args.num_samples
        results.append((steps, score, per_img))
        print(f"[eval] DDIM-{steps}: FID={score:.2f}  ({per_img*1000:.0f} ms/image)")
        fid.reset()  # keeps real stats, drops fake stats

    print(f"\nFID vs {args.num_real} real images, "
          f"{args.num_samples} generated samples each:\n")
    print("| Sampler | FID ↓ | Time/image |")
    print("|---|---|---|")
    for steps, score, per_img in results:
        print(f"| DDIM-{steps} | {score:.2f} | {per_img*1000:.0f} ms |")


if __name__ == "__main__":
    main()
