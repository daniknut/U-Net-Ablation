"""ADE20K dataset utilities."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from unet_ablation.utils import DataConfig


@dataclass(slots=True)
class SegmentationSample:
    """Paired image and segmentation mask paths."""

    image: Path
    mask: Path


def _supported_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def discover_split_samples(image_dir: Path, mask_dir: Path) -> list[SegmentationSample]:
    """Match images and masks by relative path and stem."""

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    samples: list[SegmentationSample] = []
    for image_path in sorted(path for path in image_dir.rglob("*") if path.is_file() and _supported_image(path)):
        relative_path = image_path.relative_to(image_dir)
        mask_path = mask_dir / relative_path.with_suffix(".png")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path}: expected {mask_path}")
        samples.append(SegmentationSample(image=image_path, mask=mask_path))
    return samples


def write_metadata_file(path: Path, root: Path, samples: Iterable[SegmentationSample]) -> None:
    """Write relative image and mask pairs as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            record = {
                "image": sample.image.relative_to(root).as_posix(),
                "mask": sample.mask.relative_to(root).as_posix(),
            }
            handle.write(json.dumps(record) + "\n")


def _read_metadata(path: Path, root: Path) -> list[SegmentationSample]:
    samples: list[SegmentationSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            samples.append(
                SegmentationSample(
                    image=root / record["image"],
                    mask=root / record["mask"],
                )
            )
    return samples


def load_samples(
    root: Path,
    metadata_file: Path | None,
    image_subdir: str,
    mask_subdir: str,
) -> list[SegmentationSample]:
    """Load a split from metadata when present, otherwise scan the raw folders."""

    if metadata_file is not None and metadata_file.exists():
        return _read_metadata(metadata_file, root)
    return discover_split_samples(root / image_subdir, root / mask_subdir)


class ADE20KDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple ADE20K loader with resize, optional flip, and tensor conversion."""

    def __init__(
        self,
        samples: list[SegmentationSample],
        image_size: tuple[int, int],
        ignore_index: int,
        label_offset: int = 0,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.label_offset = label_offset
        self.normalize_mean = torch.tensor(normalize_mean, dtype=torch.float32).view(3, 1, 1)
        self.normalize_std = torch.tensor(normalize_std, dtype=torch.float32).view(3, 1, 1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image).convert("RGB")
        mask = Image.open(sample.mask)

        if self.image_size:
            resize_size = (self.image_size[1], self.image_size[0])
            image = image.resize(resize_size, Image.Resampling.BILINEAR)
            mask = mask.resize(resize_size, Image.Resampling.NEAREST)

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)

        if self.label_offset:
            valid_mask = mask_array != self.ignore_index
            mask_array[valid_mask] = mask_array[valid_mask] - self.label_offset

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
        image_tensor = (image_tensor - self.normalize_mean) / self.normalize_std
        mask_tensor = torch.from_numpy(mask_array)

        return image_tensor, mask_tensor


def build_dataloaders(config: DataConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders from config."""

    root = Path(config.root)
    metadata_root = Path(config.metadata_dir)
    train_samples = load_samples(
        root=root,
        metadata_file=metadata_root / config.train_metadata,
        image_subdir=config.train_images_subdir,
        mask_subdir=config.train_masks_subdir,
    )
    val_samples = load_samples(
        root=root,
        metadata_file=metadata_root / config.val_metadata,
        image_subdir=config.val_images_subdir,
        mask_subdir=config.val_masks_subdir,
    )

    train_dataset = ADE20KDataset(
        samples=train_samples,
        image_size=config.image_size,
        ignore_index=config.ignore_index,
        label_offset=config.label_offset,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        augment=True,
    )
    val_dataset = ADE20KDataset(
        samples=val_samples,
        image_size=config.image_size,
        ignore_index=config.ignore_index,
        label_offset=config.label_offset,
        normalize_mean=config.normalize_mean,
        normalize_std=config.normalize_std,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
