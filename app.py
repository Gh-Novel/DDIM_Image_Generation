"""Gradio demo — DDIM Face Generation.

Single-page layout:
  - Top: title + generate controls + output
  - Middle: trajectory GIF + interpolation (collapsible)
  - Bottom: how it works / architecture description
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

from sample import load_run
from utils.visualize import interpolate_latents, trajectory_to_gif, make_grid


# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------
class State:
    def __init__(self, ckpt_path: str, prefer_ema: bool = True):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.cfg, self.model, self.diffusion = load_run(ckpt_path, self.device, prefer_ema)
        self.image_size = self.cfg.image_size
        self.in_channels = self.cfg.in_channels


STATE: Optional[State] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _seeded(seed: Optional[int]) -> torch.Generator:
    g = torch.Generator(device="cpu")
    if seed is not None and seed >= 0:
        g.manual_seed(int(seed))
    return g


def _grid_pil(samples: torch.Tensor, nrow: int) -> Image.Image:
    return Image.fromarray(make_grid(samples.cpu(), nrow=nrow))


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def cb_generate(num: int, steps: int, seed: float) -> Image.Image:
    s = STATE
    g = _seeded(int(seed))
    shape = (int(num), s.in_channels, s.image_size, s.image_size)
    x_T = torch.randn(*shape, generator=g).to(s.device)
    with torch.no_grad():
        out = s.diffusion.ddim_sample(s.model, shape, num_steps=int(steps),
                                      eta=0.0, x_T=x_T, device=s.device)
    nrow = int(np.ceil(np.sqrt(num)))
    return _grid_pil(out, nrow)


def cb_trajectory(steps: int, seed: float) -> str:
    s = STATE
    g = _seeded(int(seed))
    shape = (1, s.in_channels, s.image_size, s.image_size)
    x_T = torch.randn(*shape, generator=g).to(s.device)
    with torch.no_grad():
        _, traj = s.diffusion.ddim_sample(
            s.model, shape, num_steps=int(steps), eta=0.0,
            x_T=x_T, device=s.device,
            return_trajectory=True, trajectory_stride=1,
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    trajectory_to_gif(traj, tmp.name, fps=12)
    return tmp.name


def cb_interpolate(frames: int, steps: int, seed_a: float, seed_b: float) -> Image.Image:
    s = STATE
    shape_one = (1, s.in_channels, s.image_size, s.image_size)
    z1 = torch.randn(*shape_one, generator=_seeded(int(seed_a)))
    z2 = torch.randn(*shape_one, generator=_seeded(int(seed_b)))
    latents = interpolate_latents(z1, z2, num_steps=int(frames)).squeeze(1).to(s.device)
    with torch.no_grad():
        out = s.diffusion.ddim_sample(
            s.model, (int(frames), s.in_channels, s.image_size, s.image_size),
            num_steps=int(steps), eta=0.0, x_T=latents, device=s.device,
        )
    return _grid_pil(out, int(frames))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
TECH_MD = """
## How it works

This demo runs a **DDIM (Denoising Diffusion Implicit Model)** trained from scratch — no pretrained weights, no diffusers library.

### The core idea
A diffusion model learns to reverse a noise process. During training, we take a real face and progressively corrupt it with Gaussian noise over T=1000 steps until it's pure noise. The model (a U-Net) learns to predict the noise added at each step. At inference, we start from pure random noise and run the reverse process — but with DDIM we can skip most steps, getting a good result in just 20–50 steps instead of 1000.

### Architecture

```
Input (noise + timestep t)
        │
   ┌────▼────┐
   │  U-Net  │   Channels: [64, 128, 256, 256]
   │         │   Self-attention at 8×8 and 16×16 resolution
   │  Time   │   Sinusoidal time embedding → MLP → injected at every ResBlock
   │ Embed   │   GroupNorm + SiLU activations throughout
   └────┬────┘
        │
   predicted ε (noise)
```

The U-Net has:
- **4 resolution levels** with strided conv downsampling / nearest-neighbour upsampling
- **Residual blocks** with time-step conditioning (FiLM-style additive injection)
- **Multi-head self-attention** at the two lowest resolutions (8×8, 16×16)
- **EMA weights** used for inference — a running exponential average of training weights that produces cleaner samples

### Training
- **Dataset:** CelebA-HQ — 30,000 aligned face photographs at 256×256, resized to 64×64
- **Hardware:** Apple Mac Mini M-series (MPS backend), no cloud GPU
- **Duration:** ~100 epochs, ~14 hours total
- **Optimizer:** AdamW (CPU-resident state to avoid MPS memory pressure)
- **Loss:** simple MSE between predicted and actual noise — `L = ||ε - ε_θ(x_t, t)||²`
- **Noise schedule:** linear β from 1×10⁻⁴ → 0.02 over T=1000 steps

### Sampling modes
| Mode | What it shows |
|------|--------------|
| **Generate** | New faces sampled from pure Gaussian noise via DDIM |
| **Trajectory** | The full denoising path animated as a GIF — from noise to face |
| **Interpolate** | Spherical linear interpolation (slerp) between two noise vectors, showing a smooth transition between two generated faces |

### DDIM speedup
Standard DDPM requires T=1000 sequential network passes. DDIM uses a non-Markovian sampler that achieves comparable quality in 20–50 steps — a **20–50× speedup** with no retraining.

### Built entirely from scratch
Every component is hand-written in PyTorch:
`attention.py` · `unet.py` · `diffusion.py` · `dataset.py` · `train.py`
No Hugging Face Diffusers, no guided-diffusion, no pre-trained encoders.
"""


def build_ui():
    import gradio as gr

    s = STATE
    max_steps = min(s.cfg.timesteps, 100)   # cap at 100 for CPU

    with gr.Blocks(
        title="DDIM Face Generation",
        theme=gr.themes.Soft(),
        css=".output-image img { image-rendering: pixelated; }"
    ) as demo:

        gr.Markdown("""
# 🧠 DDIM Face Generation
**Denoising Diffusion Implicit Model trained from scratch on CelebA-HQ.**
Generates novel human faces by reversing a learned noise process — no pretrained weights used.
> ⏱️ Running on CPU — generation takes ~30–60 seconds. Use **seed ≥ 0** to reproduce results.
        """)

        # ── Generate ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Controls")
                num    = gr.Slider(1, 9, value=4, step=1, label="Number of faces")
                steps  = gr.Slider(10, max_steps, value=20, step=5,
                                   label="DDIM steps  (more = sharper, slower)")
                seed   = gr.Number(value=-1, label="Seed  (-1 = random each time)")
                gen_btn = gr.Button("✨ Generate Faces", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Output")
                gen_out = gr.Image(label="Generated faces", type="pil",
                                   show_label=False, height=400)

        gen_btn.click(cb_generate, [num, steps, seed], gen_out)

        gr.Markdown("---")

        # ── Trajectory & Interpolation (accordion) ────────────────────
        with gr.Accordion("🎞️ Denoising Trajectory  (noise → face GIF)", open=False):
            gr.Markdown("Watch a single face emerge from pure Gaussian noise step by step.")
            with gr.Row():
                t_steps = gr.Slider(10, max_steps, value=20, step=5, label="Steps")
                t_seed  = gr.Number(value=42, label="Seed")
                t_btn   = gr.Button("Animate", variant="secondary")
            t_out = gr.Image(label="Denoising trajectory", type="filepath")
            t_btn.click(cb_trajectory, [t_steps, t_seed], t_out)

        with gr.Accordion("🔀 Latent Interpolation  (face A → face B)", open=False):
            gr.Markdown(
                "Spherical linear interpolation (slerp) between two noise vectors — "
                "each column is a smooth blend between two independently sampled faces."
            )
            with gr.Row():
                i_frames = gr.Slider(4, 10, value=6, step=1, label="Frames")
                i_steps  = gr.Slider(10, max_steps, value=20, step=5, label="DDIM steps")
                i_seed_a = gr.Number(value=0,  label="Seed A")
                i_seed_b = gr.Number(value=7,  label="Seed B")
                i_btn    = gr.Button("Interpolate", variant="secondary")
            i_out = gr.Image(label="A ⟶ B interpolation", type="pil")
            i_btn.click(cb_interpolate, [i_frames, i_steps, i_seed_a, i_seed_b], i_out)

        gr.Markdown("---")

        # ── Tech description ──────────────────────────────────────────
        with gr.Accordion("📖 How it works — architecture, training & theory", open=False):
            gr.Markdown(TECH_MD)

        gr.Markdown(
            "<div style='text-align:center;color:#888;font-size:0.85em'>"
            "Built from scratch · PyTorch · CelebA-HQ · Apple Silicon · "
            "<a href='https://github.com/Gh-Novel/DDIM_Image_Generation' target='_blank'>GitHub</a>"
            "</div>"
        )

    return demo


# ---------------------------------------------------------------------------
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "stage-64_best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main():
    global STATE
    args = parse_args()
    STATE = State(args.ckpt, prefer_ema=not args.no_ema)
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",   # required for HF Spaces Docker
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
