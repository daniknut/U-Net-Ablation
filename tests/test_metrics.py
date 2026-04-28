from __future__ import annotations

import pytest
import torch

from unet_ablation.metrics.segmentation import compute_segmentation_metrics, confusion_matrix


def test_segmentation_metrics_match_expected_values() -> None:
    predictions = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    targets = torch.tensor([[[0, 1], [0, 2]]], dtype=torch.long)

    metrics = compute_segmentation_metrics(
        predictions=predictions,
        targets=targets,
        num_classes=3,
    )

    assert metrics["mean_iou"] == pytest.approx(2.0 / 3.0)
    assert metrics["pixel_accuracy"] == pytest.approx(0.75)


def test_confusion_matrix_respects_ignore_index() -> None:
    predictions = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    targets = torch.tensor([[[0, 255], [0, 2]]], dtype=torch.long)

    matrix = confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=3,
        ignore_index=255,
    )

    expected = torch.tensor(
        [
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(matrix, expected)
