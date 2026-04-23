"""Metadata helpers that do not depend on PyTorch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
