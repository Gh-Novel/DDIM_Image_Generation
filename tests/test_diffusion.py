"""Tests for schedules, forward process, DDIM sampling, EMA, and the
custom CPU-state AdamW (checked against torch.optim.AdamW).
"""
import pytest
import torch
import torch.nn as nn

from models.diffusion import (
    GaussianDiffusion, EMA, AdamW,
    linear_beta_schedule, cosine_beta_schedule, make_betas,
)


class ZeroModel(nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


@pytest.fixture
def diff():
    return GaussianDiffusion(timesteps=100, beta_start=1e-4, beta_end=2e-2,
                             schedule="linear")


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def test_linear_schedule_endpoints():
    betas = linear_beta_schedule(1000, 1e-4, 2e-2)
    assert betas.shape == (1000,)
    assert betas[0].item() == pytest.approx(1e-4)
    assert betas[-1].item() == pytest.approx(2e-2)


def test_cosine_schedule_bounds():
    betas = cosine_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert betas.min().item() > 0
    assert betas.max().item() <= 0.999


def test_make_betas_rejects_unknown():
    with pytest.raises(ValueError):
        make_betas("quadratic", 100, 1e-4, 2e-2)


def test_alphas_cumprod_monotone(diff):
    abar = diff.alphas_cumprod
    assert (abar[1:] <= abar[:-1]).all(), "abar must be non-increasing"
    assert abar[0].item() < 1.0


# ---------------------------------------------------------------------------
# Forward process q(x_t | x_0)
# ---------------------------------------------------------------------------
def test_q_sample_near_identity_at_t0(diff):
    torch.manual_seed(0)
    x0 = torch.randn(4, 3, 16, 16)
    t0 = torch.zeros(4, dtype=torch.long)
    xt, _ = diff.q_sample(x0, t0)
    assert (xt - x0).abs().max().item() < 0.1


def test_q_sample_near_noise_at_T(diff):
    torch.manual_seed(0)
    x0 = torch.randn(4, 3, 16, 16)
    tT = torch.full((4,), diff.timesteps - 1, dtype=torch.long)
    xtT, _ = diff.q_sample(x0, tT)
    assert xtT.std().item() > 0.7


def test_training_loss_of_zero_model_is_noise_var(diff):
    torch.manual_seed(0)
    x0 = torch.randn(8, 3, 16, 16)
    loss = diff.training_loss(ZeroModel(), x0)
    assert 0.5 < loss.item() < 1.5


# ---------------------------------------------------------------------------
# DDIM sampling
# ---------------------------------------------------------------------------
def test_ddim_shape(diff):
    out = diff.ddim_sample(ZeroModel(), (2, 3, 16, 16), num_steps=10, eta=0.0,
                           x_T=torch.randn(2, 3, 16, 16))
    assert out.shape == (2, 3, 16, 16)


def test_ddim_deterministic_at_eta0(diff):
    xT = torch.randn(2, 3, 16, 16)
    a = diff.ddim_sample(ZeroModel(), (2, 3, 16, 16), num_steps=10, eta=0.0, x_T=xT.clone())
    b = diff.ddim_sample(ZeroModel(), (2, 3, 16, 16), num_steps=10, eta=0.0, x_T=xT.clone())
    assert torch.allclose(a, b, atol=1e-6)


def test_ddim_stochastic_at_eta1(diff):
    xT = torch.randn(2, 3, 16, 16)
    torch.manual_seed(1)
    a = diff.ddim_sample(ZeroModel(), (2, 3, 16, 16), num_steps=10, eta=1.0, x_T=xT.clone())
    torch.manual_seed(2)
    b = diff.ddim_sample(ZeroModel(), (2, 3, 16, 16), num_steps=10, eta=1.0, x_T=xT.clone())
    assert not torch.allclose(a, b, atol=1e-4)


def test_ddim_trajectory_frames(diff):
    _, traj = diff.ddim_sample(ZeroModel(), (1, 3, 16, 16), num_steps=10, eta=0.0,
                               x_T=torch.randn(1, 3, 16, 16), return_trajectory=True)
    assert len(traj) == 11  # initial noise + one frame per step


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def test_ema_moves_halfway_at_decay_half():
    torch.manual_seed(0)
    net = nn.Linear(4, 4)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    ema = EMA(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.ones_like(p))
    ema.update(net)
    for k, v in ema.shadow.items():
        expected = 0.5 * before[k] + 0.5 * net.state_dict()[k]
        assert torch.allclose(v, expected, atol=1e-6)


def test_ema_copy_to_round_trip():
    torch.manual_seed(0)
    net = nn.Linear(4, 4)
    ema = EMA(net, decay=0.999)
    target = nn.Linear(4, 4)
    ema.copy_to(target)
    for a, b in zip(net.parameters(), target.parameters()):
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Custom AdamW vs torch.optim.AdamW
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_adamw_matches_torch_reference(weight_decay):
    torch.manual_seed(0)
    ref_net = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
    own_net = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 8))
    own_net.load_state_dict(ref_net.state_dict())

    lr = 1e-3
    ref_opt = torch.optim.AdamW(ref_net.parameters(), lr=lr, weight_decay=weight_decay)
    own_opt = AdamW(own_net.parameters(), lr=lr, weight_decay=weight_decay)

    x = torch.randn(32, 8)
    y = torch.randn(32, 8)
    for _ in range(5):
        ref_loss = nn.functional.mse_loss(ref_net(x), y)
        ref_opt.zero_grad(); ref_loss.backward(); ref_opt.step()

        own_loss = nn.functional.mse_loss(own_net(x), y)
        own_opt.zero_grad(); own_loss.backward(); own_opt.step()

    for (kr, vr), (ko, vo) in zip(ref_net.state_dict().items(),
                                  own_net.state_dict().items()):
        assert torch.allclose(vr, vo, atol=1e-5), f"param {kr} diverged"


def test_adamw_handles_zero_grads():
    # the original motivation for the custom optimizer: zero grads must not
    # produce NaNs (PyTorch 2.3.1 MPS AdamW bug)
    torch.manual_seed(0)
    net = nn.Linear(4, 4)
    opt = AdamW(net.parameters(), lr=1e-3)
    out = net(torch.randn(2, 4))
    # loss that ignores the bias -> bias grad is exactly zero
    loss = (out * 0).sum() + net.weight.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_adamw_state_dict_round_trip():
    torch.manual_seed(0)
    net = nn.Linear(4, 4)
    opt = AdamW(net.parameters(), lr=1e-3)
    nn.functional.mse_loss(net(torch.randn(2, 4)), torch.randn(2, 4)).backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = AdamW(net.parameters(), lr=1e-3)
    opt2.load_state_dict(sd)
    assert opt2.t == opt.t
    for m1, m2 in zip(opt.m, opt2.m):
        assert torch.allclose(m1, m2)
