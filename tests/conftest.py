"""Test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def write_dummy_split(root: Path, split_name: str, samples: int = 2) -> None:
    image_dir = root / "images" / split_name
    mask_dir = root / "annotations" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(samples):
        image_array = np.full((48, 48, 3), fill_value=index * 32 + 64, dtype=np.uint8)
        mask_array = np.zeros((48, 48), dtype=np.uint8)
        mask_array[:, :16] = 1
        mask_array[:, 16:32] = 2

        Image.fromarray(image_array).save(image_dir / f"sample_{index}.jpg")
        Image.fromarray(mask_array).save(mask_dir / f"sample_{index}.png")


@pytest.fixture
def smoke_config_path(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "ade20k"
    write_dummy_split(dataset_root, "training")
    write_dummy_split(dataset_root, "validation")

    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "smoke",
                "data": {
                    "root": str(dataset_root),
                    "metadata_dir": str(dataset_root / "metadata"),
                    "image_size": [64, 64],
                    "batch_size": 2,
                    "num_workers": 0,
                    "num_classes": 3,
                    "ignore_index": 255,
                },
                "model": {
                    "architecture": "unet",
                    "variant": "highres_skip_only",
                    "num_classes": 3,
                    "encoder_channels": [16, 32, 64, 128],
                    "bottleneck_channels": 256,
                },
                "train": {
                    "device": "cpu",
                    "epochs": 2,
                    "seed": 0,
                    "output_dir": str(tmp_path / "runs"),
                    "checkpoint_dir": str(tmp_path / "checkpoints"),
                },
                "evaluation": {
                    "output_dir": str(tmp_path / "eval"),
                    "num_visualizations": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path
