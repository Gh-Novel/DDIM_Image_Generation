"""Tests for the U-Net backbone."""
import pytest
import torch
import torch.nn.functional as F

from models.unet import UNet, SinusoidalTimeEmbedding


def tiny_unet(**overrides) -> UNet:
    kwargs = dict(
        image_size=32,
        in_channels=3,
        base_channels=16,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        attn_resolutions=(8,),
        time_embed_dim=64,
    )
    kwargs.update(overrides)
    return UNet(**kwargs)


def test_forward_shape():
    torch.manual_seed(0)
    net = tiny_unet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    assert net(x, t).shape == x.shape


def test_output_zero_at_init():
    # out_conv is zero-initialized, so an untrained net predicts exactly 0,
    # which makes the first training steps well-behaved.
    torch.manual_seed(0)
    net = tiny_unet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    assert torch.allclose(net(x, t), torch.zeros_like(x))


def test_gradients_flow():
    torch.manual_seed(0)
    net = tiny_unet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    loss = F.mse_loss(net(x, t), torch.randn_like(x))
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
    assert any(g > 0 for g in grads)


@pytest.mark.parametrize("mults,blocks", [((1, 2), 1), ((1, 2, 4), 2), ((1, 2, 4, 4), 2)])
def test_skip_connections_balance(mults, blocks):
    # construction asserts all skips are consumed; forward asserts the same
    net = tiny_unet(channel_mults=mults, num_res_blocks=blocks, image_size=64)
    x = torch.randn(1, 3, 64, 64)
    t = torch.zeros(1, dtype=torch.long)
    assert net(x, t).shape == x.shape


def test_stage64_config_builds():
    # the actual stage-64 architecture from config.py (~30M params)
    net = tiny_unet(
        image_size=64, base_channels=64, channel_mults=(1, 2, 4, 4),
        num_res_blocks=2, attn_resolutions=(8, 8, 16), time_embed_dim=256,
    )
    params = sum(p.numel() for p in net.parameters())
    assert 20e6 < params < 50e6, f"unexpected param count {params/1e6:.1f}M"


def test_time_embedding_shape_and_parity():
    emb = SinusoidalTimeEmbedding(dim=64)
    t = torch.arange(4)
    assert emb(t).shape == (4, 64)
    with pytest.raises(ValueError):
        SinusoidalTimeEmbedding(dim=63)
