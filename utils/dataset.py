"""CelebA-HQ dataset loader.

Reads pre-cropped 256x256 JPGs from a flat directory, resizes to the target
stage resolution, applies horizontal flip augmentation, and normalizes to
[-1, 1] (the convention diffusion models work in).
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CelebAHQ(Dataset):
    EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self, root: str, image_size: int, augment: bool = True,
                 limit: Optional[int] = None):
        self.root = root
        if not os.path.isdir(root):
            raise FileNotFoundError(f"data dir not found: {root}")

        files: List[str] = []
        for ext in self.EXTS:
            files.extend(glob.glob(os.path.join(root, f"*{ext}")))
            files.extend(glob.glob(os.path.join(root, f"*{ext.upper()}")))
        files = sorted(set(files))
        if not files:
            raise RuntimeError(f"no images found in {root}")
        if limit is not None:
            files = files[:limit]
        self.files = files
        self.image_size = image_size

        ops = []
        if augment:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        # bilinear is the standard choice for downsampling photographs
        ops.append(transforms.Resize(image_size, antialias=True))
        ops.append(transforms.CenterCrop(image_size))
        ops.append(transforms.ToTensor())                  # [0, 1]
        ops.append(transforms.Normalize([0.5] * 3, [0.5] * 3))  # [-1, 1]
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


def make_dataloader(
    root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: bool = True,
    limit: Optional[int] = None,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = CelebAHQ(root=root, image_size=image_size, augment=augment, limit=limit)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] tensors back to [0, 1] for visualization/saving."""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_DIR = "/Volumes/Projects/DDIM_image_Generation/celeba_hq_256"

    ds = CelebAHQ(DATA_DIR, image_size=64, augment=True, limit=8)
    assert len(ds) == 8
    x = ds[0]
    assert x.shape == (3, 64, 64), x.shape
    assert -1.0 <= x.min().item() <= x.max().item() <= 1.0

    loader = make_dataloader(DATA_DIR, image_size=64, batch_size=4,
                             num_workers=0, limit=8)
    batch = next(iter(loader))
    assert batch.shape == (4, 3, 64, 64), batch.shape
    print(f"dataset ok: {len(CelebAHQ(DATA_DIR, image_size=64, augment=False))} images total")

    # test a 256 sample as well
    ds256 = CelebAHQ(DATA_DIR, image_size=256, augment=False, limit=2)
    assert ds256[0].shape == (3, 256, 256)
    print("dataset.py: all tests passed")
