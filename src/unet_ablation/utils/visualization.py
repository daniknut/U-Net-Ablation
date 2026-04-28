"""Utilities for saving qualitative segmentation examples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image


def _palette(num_classes: int) -> list[int]:
    values: list[int] = []
    for class_index in range(max(num_classes, 1)):
        values.extend(
            [
                (37 * class_index) % 256,
                (67 * class_index) % 256,
                (97 * class_index) % 256,
            ]
        )
    return (values + [0] * 768)[:768]


def save_colorized_mask(
    mask: torch.Tensor,
    path: str | Path,
    num_classes: int,
    ignore_index: int | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    array = mask.detach().cpu().numpy().astype(np.uint8)
    if ignore_index is not None:
        array = np.where(mask.detach().cpu().numpy() == ignore_index, 0, array)

    image = Image.fromarray(array)
    image = image.convert("P")
    image.putpalette(_palette(num_classes))
    image.save(target)


def save_image_tensor(
    image: torch.Tensor,
    path: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(3, 1, 1)
    image = image.detach().cpu() * std_tensor + mean_tensor
    image = image.clamp(0.0, 1.0)
    array = (image.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(array).save(target)
