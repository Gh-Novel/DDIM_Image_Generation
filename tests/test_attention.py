"""Tests for the 2D multi-head self-attention block."""
import pytest
import torch

from models.attention import SelfAttention2d


def test_shape_preserved():
    torch.manual_seed(0)
    block = SelfAttention2d(channels=64, num_heads=4)
    x = torch.randn(2, 64, 8, 8)
    assert block(x).shape == x.shape


def test_identity_at_init():
    # proj_out is zero-initialized, so the block must be an exact identity
    # before any training step.
    torch.manual_seed(0)
    block = SelfAttention2d(channels=64, num_heads=4)
    x = torch.randn(2, 64, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_gradients_reach_proj_out():
    torch.manual_seed(0)
    block = SelfAttention2d(channels=32, num_heads=4)
    x = torch.randn(1, 32, 8, 8)
    block(x).sum().backward()
    assert block.proj_out.weight.grad is not None
    assert block.proj_out.weight.grad.abs().sum() > 0


def test_larger_feature_map():
    block = SelfAttention2d(channels=128, num_heads=8)
    y = block(torch.randn(1, 128, 32, 32))
    assert y.shape == (1, 128, 32, 32)


def test_rejects_indivisible_heads():
    with pytest.raises(ValueError):
        SelfAttention2d(channels=30, num_heads=4)
