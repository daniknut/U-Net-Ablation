"""Segmentation evaluation metrics."""

from __future__ import annotations

import torch


def _prepare_predictions(predictions: torch.Tensor) -> torch.Tensor:
    if predictions.ndim == 4:
        return predictions.argmax(dim=1)
    if predictions.ndim == 3:
        return predictions
    raise ValueError("Predictions must be logits [N, C, H, W] or labels [N, H, W]")


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Build a confusion matrix for segmentation masks."""

    predictions = _prepare_predictions(predictions).to(dtype=torch.int64).reshape(-1)
    targets = targets.to(dtype=torch.int64).reshape(-1)

    valid = (targets >= 0) & (targets < num_classes)
    if ignore_index is not None:
        valid &= targets != ignore_index
    valid &= (predictions >= 0) & (predictions < num_classes)

    predictions = predictions[valid]
    targets = targets[valid]
    if targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)

    encoded = targets * num_classes + predictions
    histogram = torch.bincount(encoded, minlength=num_classes * num_classes)
    return histogram.reshape(num_classes, num_classes)


def mean_iou(confusion: torch.Tensor) -> float:
    """Compute mean IoU from a confusion matrix."""

    confusion = confusion.to(dtype=torch.float32)
    true_positive = torch.diag(confusion)
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    union = true_positive + false_positive + false_negative
    valid = union > 0
    if not torch.any(valid):
        return 0.0
    return float((true_positive[valid] / union[valid]).mean().item())


def pixel_accuracy(confusion: torch.Tensor) -> float:
    """Compute pixel accuracy from a confusion matrix."""

    confusion = confusion.to(dtype=torch.float32)
    total = confusion.sum()
    if total <= 0:
        return 0.0
    return float(torch.diag(confusion).sum().item() / total.item())


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Convenience wrapper that returns standard metrics."""

    confusion = confusion_matrix(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    return {
        "mean_iou": mean_iou(confusion),
        "pixel_accuracy": pixel_accuracy(confusion),
    }
