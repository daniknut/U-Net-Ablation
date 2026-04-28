from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from unet_ablation.data.ade20k import build_dataloaders
from unet_ablation.engine.trainer import evaluate_model
from unet_ablation.models.unet import build_model
from unet_ablation.utils.config import load_experiment_config


def test_config_loader_and_validation_smoke(smoke_config_path: Path) -> None:
    config = load_experiment_config(smoke_config_path)
    _, val_loader = build_dataloaders(config.data)
    images, masks = next(iter(val_loader))

    assert images.shape == (2, 3, 64, 64)
    assert masks.shape == (2, 64, 64)

    model = build_model(config.model)
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
