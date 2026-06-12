"""Tests for the CelebA-HQ loader, using synthetic images in a tmp dir."""
import numpy as np
import pytest
import torch
from PIL import Image

from utils.dataset import CelebAHQ, make_dataloader, denormalize


@pytest.fixture
def image_dir(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(8):
        arr = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(tmp_path / f"{i:05d}.jpg")
    return str(tmp_path)


def test_item_shape_and_range(image_dir):
    ds = CelebAHQ(image_dir, image_size=64, augment=False)
    assert len(ds) == 8
    x = ds[0]
    assert x.shape == (3, 64, 64)
    assert -1.0 <= x.min().item() <= x.max().item() <= 1.0


def test_limit(image_dir):
    assert len(CelebAHQ(image_dir, image_size=64, limit=3)) == 3


def test_dataloader_batch(image_dir):
    loader = make_dataloader(image_dir, image_size=32, batch_size=4,
                             num_workers=0, limit=8)
    batch = next(iter(loader))
    assert batch.shape == (4, 3, 32, 32)


def test_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        CelebAHQ("/nonexistent/path", image_size=64)


def test_empty_dir_raises(tmp_path):
    with pytest.raises(RuntimeError):
        CelebAHQ(str(tmp_path), image_size=64)


def test_denormalize_maps_to_unit_range():
    x = torch.tensor([-1.0, 0.0, 1.0])
    out = denormalize(x)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))
    # out-of-range values are clamped first
    assert denormalize(torch.tensor([5.0])).item() == 1.0
