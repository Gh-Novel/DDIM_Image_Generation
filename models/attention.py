"""2D spatial multi-head self-attention block for U-Net feature maps.

Operates on tensors of shape (B, C, H, W). Reshapes to (B, H*W, C), applies
GroupNorm + multi-head self-attention with residual connection, and reshapes
back. Standard recipe used in DDPM/score-based models.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        groups = min(num_groups, channels)
        while channels % groups != 0:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        # Zero-init the output projection so the block starts as identity.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)                                    # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)                        # each (B, C, H, W)

        # (B, heads, head_dim, H*W)
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # attn weights: (B, heads, H*W, H*W)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)       # (B, heads, head_dim, N)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out


if __name__ == "__main__":
    torch.manual_seed(0)
    block = SelfAttention2d(channels=64, num_heads=4)
    x = torch.randn(2, 64, 8, 8)

    # 1) shape preserved
    y = block(x)
    assert y.shape == x.shape, y.shape
    # 2) zero-init means initial output equals input (identity at init)
    assert torch.allclose(y, x, atol=1e-6), "attn block should be identity at init"
    # 3) gradients flow into proj_out (qkv grad is 0 at init b/c proj_out
    #    is zero-initialized, which is intentional)
    y.sum().backward()
    assert block.proj_out.weight.grad is not None
    assert block.proj_out.weight.grad.abs().sum() > 0

    # 4) larger map
    block2 = SelfAttention2d(channels=128, num_heads=8)
    y2 = block2(torch.randn(1, 128, 32, 32))
    assert y2.shape == (1, 128, 32, 32)

    # 5) MPS run
    if torch.backends.mps.is_available():
        block_mps = SelfAttention2d(channels=64, num_heads=4).to("mps")
        ym = block_mps(torch.randn(1, 64, 16, 16, device="mps"))
        assert ym.shape == (1, 64, 16, 16)
        print("mps ok")

    print("attention.py: all tests passed")
