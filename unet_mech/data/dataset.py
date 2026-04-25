from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _resampling(name: str):
    return getattr(getattr(Image, "Resampling", Image), name)


class MontgomeryDataset(Dataset):
    """
    PyTorch Dataset for Montgomery CXR images.

    * Images are opened with .convert("RGB") → 3 channels for ResNet18.
    * Normalisation uses ImageNet mean/std (applied in make_transforms).
    * Masks are single-channel binary float tensors.
    """

    def __init__(
        self,
        root: str,
        img_size: int,
        transform=None,
        mask_transform=None,
        joint_transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform
        self.mask_transform = mask_transform
        self.joint_transform = joint_transform

        img_dir = self.root / "CXR_png"
        left_dir = self.root / "ManualMask" / "leftMask"
        right_dir = self.root / "ManualMask" / "rightMask"

        self.samples: List[Tuple[Path, Path, Path]] = []
        for img_path in sorted(img_dir.glob("*.png")):
            left_mask = left_dir / img_path.name
            right_mask = right_dir / img_path.name
            if left_mask.exists() and right_mask.exists():
                self.samples.append((img_path, left_mask, right_mask))

        if not self.samples:
            raise FileNotFoundError(
                f"No paired images found under {img_dir}. "
                "Run download_montgomery() first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, left_path, right_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB").resize(
            (self.img_size, self.img_size), _resampling("BILINEAR")
        )

        left = np.array(
            Image.open(left_path).convert("L").resize(
                (self.img_size, self.img_size), _resampling("NEAREST")
            )
        )
        right = np.array(
            Image.open(right_path).convert("L").resize(
                (self.img_size, self.img_size), _resampling("NEAREST")
            )
        )
        mask_np = ((left > 127) | (right > 127)).astype(np.uint8)
        mask = Image.fromarray(mask_np * 255)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        mask = (mask > 0.5).float()
        return image, mask
