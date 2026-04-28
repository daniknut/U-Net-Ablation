from __future__ import annotations

import json
from pathlib import Path

from unet_ablation.engine.trainer import (
    evaluate_checkpoint,
    save_prediction_samples,
    train_experiment,
)
from unet_ablation.utils.config import load_experiment_config


def test_train_experiment_writes_checkpoints_and_summary(smoke_config_path: Path) -> None:
    config = load_experiment_config(smoke_config_path)

    result = train_experiment(config)

    assert result.best_checkpoint.exists()
    assert result.last_checkpoint.exists()
    assert result.history_path.exists()
    assert result.summary_path.exists()
    assert result.epochs_completed >= 1

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["experiment_name"] == "smoke"
    assert summary["seed"] == 0
    assert Path(summary["best_checkpoint"]).exists()
    assert set(summary["best_metrics"]) == {"loss", "mean_iou", "pixel_accuracy"}


def test_evaluate_checkpoint_and_sample_export(smoke_config_path: Path) -> None:
    config = load_experiment_config(smoke_config_path)
    result = train_experiment(config)

    metrics = evaluate_checkpoint(config, result.best_checkpoint)
    sample_dir = save_prediction_samples(config, result.best_checkpoint)

    assert set(metrics) == {"loss", "mean_iou", "pixel_accuracy"}
    assert sample_dir.exists()

    exported = {path.name for path in sample_dir.iterdir()}
    assert any(name.endswith("_image.png") for name in exported)
    assert any(name.endswith("_prediction.png") for name in exported)
    assert any(name.endswith("_target.png") for name in exported)
