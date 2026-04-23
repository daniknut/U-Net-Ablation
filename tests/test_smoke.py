from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn

from unet_ablation.data import build_dataloaders
from unet_ablation.engine import evaluate_model
from unet_ablation.models import build_unet
from unet_ablation.utils import load_experiment_config


def _write_dummy_split(root: Path, split_name: str) -> None:
    image_dir = root / "images" / split_name
    mask_dir = root / "annotations" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(2):
        image_array = np.full((48, 48, 3), fill_value=index * 32 + 64, dtype=np.uint8)
        mask_array = np.zeros((48, 48), dtype=np.uint8)
        mask_array[:, :16] = 1
        mask_array[:, 16:32] = 2

        Image.fromarray(image_array, mode="RGB").save(image_dir / f"sample_{index}.jpg")
        Image.fromarray(mask_array, mode="L").save(mask_dir / f"sample_{index}.png")


def test_config_loader_and_validation_smoke(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ade20k"
    _write_dummy_split(dataset_root, "training")
    _write_dummy_split(dataset_root, "validation")

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
                    "num_classes": 3,
                    "encoder_channels": [16, 32, 64, 128],
                    "bottleneck_channels": 256,
                    "skip_mask": [True, True, False, False],
                },
                "train": {
                    "device": "cpu",
                    "epochs": 1,
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

    config = load_experiment_config(config_path)
    _, val_loader = build_dataloaders(config.data)
    images, masks = next(iter(val_loader))

    assert images.shape == (2, 3, 64, 64)
    assert masks.shape == (2, 64, 64)

    model = build_unet(config.model)
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        num_classes=config.data.num_classes,
        ignore_index=config.data.ignore_index,
    )

    assert set(metrics) == {"loss", "mean_iou", "pixel_accuracy"}
