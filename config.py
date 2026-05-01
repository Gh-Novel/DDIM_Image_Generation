"""Central hyperparameter config for DDIM face-generation project.

Resolution stages (64 -> 128 -> 256) share the same model architecture; each
stage just resizes input images and adjusts batch size. Use Config.for_stage()
to materialize a stage-specific config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Tuple

import torch


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    # ---- paths ---------------------------------------------------------
    project_root: str = "/Volumes/Projects/DDIM_image_Generation"
    data_dir: str = "/Volumes/Projects/DDIM_image_Generation/celeba_hq_256"
    ckpt_dir: str = "/Volumes/Projects/DDIM_image_Generation/minidiffusion/checkpoints"
    sample_dir: str = "/Volumes/Projects/DDIM_image_Generation/minidiffusion/samples"

    # ---- model ---------------------------------------------------------
    image_size: int = 64                       # current training stage
    in_channels: int = 3
    base_channels: int = 128
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)   # -> [128, 256, 512, 1024]
    num_res_blocks: int = 2
    # resolutions (in pixels) at which self-attention is applied. Two attn
    # blocks at the bottleneck (8x8) and one at 32x32 are encoded by listing
    # 8 twice and 32 once; the U-Net checks membership.
    attn_resolutions: Tuple[int, ...] = (8, 8, 32)
    dropout: float = 0.1
    time_embed_dim: int = 512                  # 4 * base_channels

    # ---- diffusion -----------------------------------------------------
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "linear"              # linear | cosine
    ddim_steps: int = 50
    ddim_eta: float = 0.0                      # 0 = deterministic DDIM

    # ---- training ------------------------------------------------------
    batch_size: int = 32                       # overridden per stage
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    epochs: int = 100
    log_every: int = 50                        # steps
    sample_every_epochs: int = 5               # log a sample grid to W&B
    ckpt_every_epochs: int = 1
    seed: int = 42

    # ---- runtime -------------------------------------------------------
    device: str = field(default_factory=_pick_device)
    mixed_precision: bool = False              # MPS autocast still flaky
    use_wandb: bool = True
    wandb_project: str = "minidiffusion-celebahq"
    run_name: str = "stage-64"

    # ---- helpers -------------------------------------------------------
    @classmethod
    def for_stage(cls, image_size: int, **overrides) -> "Config":
        """Return a config tuned for a given resolution stage.

        Channel counts are tuned for a 24GB Mac Mini (Apple Silicon, MPS).
        Smaller stages use a smaller backbone so the warm-up trains quickly;
        the 256-stage uses the full [128,256,512,1024] config from the spec.
        """
        if image_size == 64:
            # ~30M params — fast to iterate on, fits easily in MPS memory
            stage = dict(image_size=64, batch_size=32, run_name="stage-64",
                         base_channels=64, channel_mults=(1, 2, 4, 4),
                         num_res_blocks=2, attn_resolutions=(8, 8, 16),
                         time_embed_dim=256)
        elif image_size == 128:
            # ~80M params
            stage = dict(image_size=128, batch_size=16, run_name="stage-128",
                         base_channels=96, channel_mults=(1, 2, 4, 4),
                         num_res_blocks=2, attn_resolutions=(8, 8, 32),
                         time_embed_dim=384)
        elif image_size == 256:
            # ~245M params — the full spec, only run overnight
            stage = dict(image_size=256, batch_size=4, run_name="stage-256",
                         base_channels=128, channel_mults=(1, 2, 4, 8),
                         num_res_blocks=2, attn_resolutions=(8, 8, 32),
                         time_embed_dim=512)
        else:
            raise ValueError(f"Unsupported image_size {image_size}")
        stage.update(overrides)
        return cls(**stage)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_config() -> Config:
    cfg = Config()
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)
    return cfg


if __name__ == "__main__":
    cfg = get_default_config()
    print("device:", cfg.device)
    for stage in (64, 128, 256):
        s = Config.for_stage(stage)
        print(f"stage {stage}: bs={s.batch_size} attn={s.attn_resolutions} run={s.run_name}")
