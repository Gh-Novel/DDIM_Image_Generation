---
title: DDIM Face Generation
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# DDIM Face Generation

A **Denoising Diffusion Implicit Model (DDIM)** trained from scratch on 30,000 faces from the CelebA-HQ dataset. Built entirely in PyTorch — no pretrained components, no diffusers library.

## Demo features

- **Generate** — sample new human faces from pure Gaussian noise in 20 steps
- **Trajectory** — animated GIF showing the full denoising path (noise → face)  
- **Interpolate** — smooth slerp blend between two independently sampled faces
- **How it works** — full architecture and training details at the bottom of the page

## Technical details

| | |
|---|---|
| Architecture | U-Net with sinusoidal time embeddings + multi-head self-attention |
| Channels | [64, 128, 256, 256] |
| Parameters | 25.6M |
| Dataset | CelebA-HQ (30k faces, 64×64) |
| Training | 100 epochs, ~14 hours, Apple Silicon MPS |
| Sampler | DDIM — 20 steps vs DDPM 1000 steps (50× speedup) |
| Noise schedule | Linear β: 1×10⁻⁴ → 0.02, T=1000 |
| Inference weights | EMA (exponential moving average of training weights) |

## Built from scratch

Every component is hand-written:
`attention.py` · `unet.py` · `diffusion.py` · `dataset.py` · `train.py`

## Source code

[github.com/Gh-Novel/DDIM_Image_Generation](https://github.com/Gh-Novel/DDIM_Image_Generation)
