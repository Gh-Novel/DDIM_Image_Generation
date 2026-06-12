# 🧠 DDIM Face Generation

> A **Denoising Diffusion Implicit Model** trained from scratch on 30,000 faces — no pretrained weights, no diffusers library. Pure PyTorch.

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/NoobNovel/DDIM_Image_Generation)
[![CI](https://github.com/Gh-Novel/DDIM_Image_Generation/actions/workflows/ci.yml/badge.svg)](https://github.com/Gh-Novel/DDIM_Image_Generation/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 🖼️ Results — 100 Epochs on CelebA-HQ 64×64

<div align="center">
<img width="600" alt="Generated faces at 100 epochs" src="https://github.com/user-attachments/assets/bcd19a35-9a20-4e68-a252-a6140499e44f" />

*Faces generated from pure Gaussian noise — no post-processing*
</div>

---

## 🚀 Live Demo

<div align="center">

**[▶ Try it on Hugging Face Spaces](https://huggingface.co/spaces/novelkathor/DDIM_Image_Generation)**

<img width="900" alt="Gradio demo UI" src="https://github.com/user-attachments/assets/1caff3cd-37e4-4adf-b3fb-75f9a715db77" />

</div>

**Demo features:**
- **✨ Generate** — sample new faces from pure noise with adjustable DDIM steps
- **🎞️ Trajectory** — animated GIF showing the full denoising path (noise → face)
- **🔀 Interpolate** — spherical linear interpolation (slerp) between two faces
- **📖 How it works** — full architecture & training breakdown at the bottom of the page

---

## ⚙️ Technical Details

| | |
|---|---|
| **Architecture** | U-Net with sinusoidal time embeddings + multi-head self-attention |
| **Channels** | [64, 128, 256, 256] |
| **Parameters** | 25.6M |
| **Dataset** | CelebA-HQ — 30,000 aligned faces at 64×64 |
| **Training** | 100 epochs, ~40 hours, Apple Silicon MPS (no cloud GPU) |
| **Sampler** | DDIM — 20 steps vs DDPM 1000 steps **(50× speedup)** |
| **Noise schedule** | Linear β: 1×10⁻⁴ → 0.02, T = 1000 |
| **Inference weights** | EMA (exponential moving average of training weights) |

---

## 📊 Evaluation — FID

Frechet Inception Distance against the CelebA-HQ training distribution
(5,000 real images, 2,048 generated samples per row, EMA weights, deterministic
DDIM with η = 0). Computed with `eval.py` — torchmetrics Inception-V3, 2048-d
features. Timings are per image on Apple Silicon MPS, batch 64.

| Sampler | FID ↓ | Time/image |
|---|---|---|
| DDIM-10 | 58.66 | 212 ms |
| DDIM-20 | 37.40 | 273 ms |
| DDIM-50 | 22.63 | 901 ms |

The speed/quality tradeoff in one table: 50 DDIM steps reach FID 22.6 — a
**20× reduction in sampling steps** vs DDPM-1000 — while 20 steps still produce
recognizable faces 3× faster. Reproduce with:

```bash
python eval.py --ckpt checkpoints/stage-64_best.pt --steps 10 20 50
```

---

## 🏗️ Architecture

```
Input x_t (noisy image) + timestep t
            │
    ┌───────▼────────┐
    │  Time Embedding │  Sinusoidal → MLP → injected at every ResBlock
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │    U-Net       │  4 resolution levels
    │                │  Self-attention at 8×8 and 16×16
    │  Down → Mid    │  GroupNorm + SiLU throughout
    │       → Up     │  Zero-init output conv (identity at init)
    └───────┬────────┘
            │
    predicted ε (noise)
```

**Training objective:** `L = ||ε − ε_θ(√ᾱₜ x₀ + √(1−ᾱₜ) ε, t)||²`

---

## 📁 Project Structure

```
minidiffusion/
├── models/
│   ├── attention.py     # Multi-head self-attention (2D spatial)
│   ├── unet.py          # Full U-Net with time embeddings
│   └── diffusion.py     # DDPM training + DDIM sampling + EMA + AdamW
├── utils/
│   ├── dataset.py       # CelebA-HQ dataloader
│   └── visualize.py     # Trajectory GIF, interpolation grid
├── tests/               # pytest suite (run in CI on every push)
├── train.py             # Training loop — W&B logging, auto-resume
├── sample.py            # Inference — grid, trajectory, interpolation, compare
├── eval.py              # FID evaluation (torchmetrics Inception-V3)
├── app.py               # Gradio demo UI
└── config.py            # All hyperparameters
```

---

## 🔧 Built From Scratch

Every component is hand-written — no diffusers, no guided-diffusion, no pretrained encoders:

`attention.py` · `unet.py` · `diffusion.py` · `dataset.py` · `train.py`

Notable engineering decisions:
- **Custom CPU-resident AdamW** — fixes a MPS NaN bug in PyTorch 2.3.1 where zero-grad params corrupt optimizer state, while also saving ~2GB of GPU memory
- **EMA shadow on CPU** — keeps a smoothed copy of weights off the GPU, saving another ~1GB
- **MPS-safe DDIM indexing** — tensor indexing with MPS buffers returns garbage in some PyTorch builds; fixed by using Python ints throughout the sampling loop

---

## 🏃 Run Locally

```bash
git clone https://github.com/Gh-Novel/DDIM_Image_Generation.git
cd DDIM_Image_Generation
pip install -r requirements.txt

# Run the Gradio demo (uses bundled checkpoint)
python app.py

# Or generate samples directly
python sample.py --ckpt checkpoints/stage-64_best.pt --num 16 --steps 50

# Train from scratch on your own data
python train.py --image-size 64 --epochs 100 --run-name my-run

# Run the test suite
pip install -r requirements-dev.txt
pytest tests/ -q

# Compute FID at several DDIM step counts
python eval.py --ckpt checkpoints/stage-64_best.pt --steps 10 20 50
```
