"""Training and evaluation loops for segmentation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from unet_ablation.data import build_dataloaders
from unet_ablation.metrics import confusion_matrix, mean_iou, pixel_accuracy
from unet_ablation.models import build_unet
from unet_ablation.utils import (
    ExperimentConfig,
    append_jsonl,
    ensure_dir,
    resolve_device,
    save_json,
    set_seed,
)
from unet_ablation.utils.visualization import save_colorized_mask, save_image_tensor


@dataclass(slots=True)
class TrainResult:
    """Summary of one training run."""

    experiment_name: str
    seed: int
    run_dir: Path
    checkpoint_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    history_path: Path
    best_metrics: dict[str, float]


def _run_directory(root: str | Path, experiment_name: str, seed: int) -> Path:
    return ensure_dir(Path(root) / experiment_name / f"seed_{seed}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "metrics": metrics,
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def _load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()

        batch_size = images.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size

    average_loss = total_loss / max(total_examples, 1)
    return {"loss": average_loss}


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    running_confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            predictions = logits.argmax(dim=1)

            batch_size = images.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            running_confusion += confusion_matrix(
                predictions.cpu(),
                masks.cpu(),
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "mean_iou": mean_iou(running_confusion),
        "pixel_accuracy": pixel_accuracy(running_confusion),
    }
    return metrics


def train_experiment(config: ExperimentConfig) -> TrainResult:
    """Train one experiment from config."""

    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    run_dir = _run_directory(config.train.output_dir, config.experiment_name, config.train.seed)
    checkpoint_dir = _run_directory(
        config.train.checkpoint_dir, config.experiment_name, config.train.seed
    )
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()
    save_json(run_dir / "resolved_config.json", config.to_dict())

    train_loader, val_loader = build_dataloaders(config.data)
    model = build_unet(config.model).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_metric = float("-inf")
    best_metrics: dict[str, float] = {}
    best_checkpoint = checkpoint_dir / "best.pt"
    last_checkpoint = checkpoint_dir / "last.pt"
    epochs_without_improvement = 0

    for epoch in range(1, config.train.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=config.train.gradient_clip_norm,
        )
        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=config.data.num_classes,
            ignore_index=config.data.ignore_index,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "mean_iou": val_metrics["mean_iou"],
            "pixel_accuracy": val_metrics["pixel_accuracy"],
        }
        append_jsonl(history_path, record)
        _save_checkpoint(last_checkpoint, model, optimizer, config, epoch, val_metrics)

        score = val_metrics["mean_iou"]
        if score > best_metric + config.train.early_stopping_min_delta:
            best_metric = score
            best_metrics = val_metrics
            epochs_without_improvement = 0
            _save_checkpoint(best_checkpoint, model, optimizer, config, epoch, val_metrics)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.early_stopping_patience:
                break

    if not best_checkpoint.exists():
        raise RuntimeError("Training completed without producing a best checkpoint")

    return TrainResult(
        experiment_name=config.experiment_name,
        seed=config.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        history_path=history_path,
        best_metrics=best_metrics,
    )


def evaluate_checkpoint(config: ExperimentConfig, checkpoint_path: str | Path) -> dict[str, float]:
    """Load a saved checkpoint and score it on the validation split."""

    device = resolve_device(config.train.device)
    _, val_loader = build_dataloaders(config.data)
    model = build_unet(config.model).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config.data.ignore_index)
    _load_checkpoint(Path(checkpoint_path), model, device)
    return evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=config.data.num_classes,
        ignore_index=config.data.ignore_index,
    )


def save_prediction_samples(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """Save a few validation predictions as qualitative examples."""

    device = resolve_device(config.train.device)
    _, val_loader = build_dataloaders(config.data)
    dataset = val_loader.dataset
    model = build_unet(config.model).to(device)
    _load_checkpoint(Path(checkpoint_path), model, device)
    model.eval()

    sample_root = (
        ensure_dir(Path(output_dir))
        if output_dir is not None
        else _run_directory(config.evaluation.output_dir, config.experiment_name, config.train.seed)
        / "samples"
    )
    sample_root = ensure_dir(sample_root)

    saved = 0
    dataset_index = 0
    with torch.no_grad():
        for images, masks in val_loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            images = images.cpu()
            masks = masks.cpu()

            for batch_index in range(predictions.size(0)):
                sample = dataset.samples[dataset_index]
                stem = Path(sample.image).stem
                prefix = f"{saved:03d}_{stem}"

                save_image_tensor(
                    images[batch_index],
                    sample_root / f"{prefix}_image.png",
                    mean=config.data.normalize_mean,
                    std=config.data.normalize_std,
                )
                save_colorized_mask(
                    predictions[batch_index],
                    sample_root / f"{prefix}_prediction.png",
                    num_classes=config.data.num_classes,
                    ignore_index=config.data.ignore_index,
                )
                save_colorized_mask(
                    masks[batch_index],
                    sample_root / f"{prefix}_target.png",
                    num_classes=config.data.num_classes,
                    ignore_index=config.data.ignore_index,
                )

                saved += 1
                dataset_index += 1
                if saved >= config.evaluation.num_visualizations:
                    return sample_root

    return sample_root
