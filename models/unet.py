"""U-Net with sinusoidal time embeddings, residual blocks, and self-attention.

Architecture follows the DDPM/improved-DDPM lineage:
- Sinusoidal time embed -> 2-layer MLP -> per-block FiLM-style addition.
- Residual blocks (GroupNorm -> SiLU -> Conv) with optional 1x1 skip.
- Down path: ResBlock(s) [+ Attn] then strided 3x3 conv.
- Bottleneck: ResBlock -> Attn -> ResBlock.
- Up path: skip-concat -> ResBlock(s) [+ Attn] then nearest upsample + 3x3 conv.

Channels at each stage are base_channels * channel_mults[i]. Self-attention
is applied at any stage whose spatial resolution is in attn_resolutions.
The attn_resolutions tuple may list a value twice to request two attn
blocks at that resolution (e.g. (8, 8, 32) -> 2 attn at 8x8, 1 at 32x32).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttention2d


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer or float timesteps
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------
def _norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, out_ch)

        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        # zero-init final conv so block starts as a near-identity-on-skip.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Up / Down samplers
# ---------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x): return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Tuple[int, ...] = (8, 8, 32),
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = tuple(channel_mults)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = tuple(attn_resolutions)

        # ---- time embedding ----------------------------------------------
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # ---- helper: how many attn blocks at this resolution -------------
        def attn_count_at(res: int) -> int:
            return sum(1 for r in attn_resolutions if r == res)

        # ---- input projection --------------------------------------------
        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ---- down path ---------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_samplers = nn.ModuleList()

        skip_channels = [base_channels]
        cur_ch = base_channels
        cur_res = image_size

        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attn = nn.ModuleList()
            n_attn = attn_count_at(cur_res)
            for j in range(num_res_blocks):
                level_blocks.append(ResBlock(cur_ch, out_ch, time_embed_dim, dropout))
                cur_ch = out_ch
                # attach attention to as many blocks as requested at this res
                if j < n_attn:
                    level_attn.append(SelfAttention2d(cur_ch, num_heads=num_heads))
                else:
                    level_attn.append(nn.Identity())
                skip_channels.append(cur_ch)
            self.down_blocks.append(level_blocks)
            self.down_attn.append(level_attn)

            if i < len(channel_mults) - 1:
                self.down_samplers.append(Downsample(cur_ch))
                skip_channels.append(cur_ch)
                cur_res //= 2
            else:
                self.down_samplers.append(nn.Identity())

        # ---- bottleneck --------------------------------------------------
        self.mid_block1 = ResBlock(cur_ch, cur_ch, time_embed_dim, dropout)
        self.mid_attn = SelfAttention2d(cur_ch, num_heads=num_heads)
        self.mid_block2 = ResBlock(cur_ch, cur_ch, time_embed_dim, dropout)

        # ---- up path -----------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_samplers = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            level_blocks = nn.ModuleList()
            level_attn = nn.ModuleList()
            n_attn = attn_count_at(cur_res)
            # one extra block on the up path to consume the same-level skip
            for j in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(ResBlock(cur_ch + skip_ch, out_ch, time_embed_dim, dropout))
                cur_ch = out_ch
                if j < n_attn:
                    level_attn.append(SelfAttention2d(cur_ch, num_heads=num_heads))
                else:
                    level_attn.append(nn.Identity())
            self.up_blocks.append(level_blocks)
            self.up_attn.append(level_attn)

            if i > 0:
                self.up_samplers.append(Upsample(cur_ch))
                cur_res *= 2
            else:
                self.up_samplers.append(nn.Identity())

        assert not skip_channels, f"unconsumed skips: {skip_channels}"

        # ---- output ------------------------------------------------------
        self.out_norm = _norm(cur_ch)
        self.out_conv = nn.Conv2d(cur_ch, in_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        h = self.in_conv(x)
        skips = [h]

        # down
        for level_blocks, level_attn, sampler in zip(self.down_blocks, self.down_attn, self.down_samplers):
            for block, attn in zip(level_blocks, level_attn):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            if not isinstance(sampler, nn.Identity):
                h = sampler(h)
                skips.append(h)

        # bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # up
        for level_blocks, level_attn, sampler in zip(self.up_blocks, self.up_attn, self.up_samplers):
            for block, attn in zip(level_blocks, level_attn):
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            if not isinstance(sampler, nn.Identity):
                h = sampler(h)

        assert not skips, f"unconsumed skips at end: {len(skips)}"
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) tiny config sanity
    net = UNet(
        image_size=32,
        in_channels=3,
        base_channels=16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=(8,),
        time_embed_dim=64,
    )
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = net(x, t)
    assert y.shape == x.shape, y.shape

    # 2) zero-init means the model outputs ~0 at start, so loss against random
    #    target produces gradients in body parameters.
    target = torch.randn_like(x)
    loss = F.mse_loss(y, target)
    loss.backward()
    grad_norms = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
    assert any(g > 0 for g in grad_norms)

    # 3) param count for full 256-stage config (don't run forward, just build)
    full = UNet(
        image_size=256,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(8, 8, 32),
        time_embed_dim=512,
    )
    params = sum(p.numel() for p in full.parameters())
    print(f"full 256-stage UNet params: {params/1e6:.1f}M")

    # 4) 64-stage forward on MPS
    if torch.backends.mps.is_available():
        net64 = UNet(
            image_size=64,
            base_channels=64,
            channel_mults=(1, 2, 4, 4),
            num_res_blocks=1,
            attn_resolutions=(8, 16),
            time_embed_dim=256,
        ).to("mps")
        x64 = torch.randn(1, 3, 64, 64, device="mps")
        t64 = torch.randint(0, 1000, (1,), device="mps")
        with torch.no_grad():
            y64 = net64(x64, t64)
        assert y64.shape == (1, 3, 64, 64)
        print("mps forward ok")

    print("unet.py: all tests passed")
