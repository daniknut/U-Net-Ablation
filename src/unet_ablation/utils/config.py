"""Experiment configuration loading."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(slots=True)
class DataConfig:
    root: str = "data/ade20k"
    metadata_dir: str = "data/ade20k/metadata"
    train_metadata: str = "train.jsonl"
    val_metadata: str = "val.jsonl"
    train_images_subdir: str = "images/training"
    train_masks_subdir: str = "annotations/training"
    val_images_subdir: str = "images/validation"
    val_masks_subdir: str = "annotations/validation"
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 4
    num_workers: int = 0
    num_classes: int = 151
    ignore_index: int = 255
    label_offset: int = 0
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        data = dict(payload)
        data["image_size"] = tuple(data.get("image_size", (256, 256)))
        data["normalize_mean"] = tuple(data.get("normalize_mean", (0.485, 0.456, 0.406)))
        data["normalize_std"] = tuple(data.get("normalize_std", (0.229, 0.224, 0.225)))
        return cls(**data)


@dataclass(slots=True)
class ModelConfig:
    in_channels: int = 3
    num_classes: int = 151
    encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    bottleneck_channels: int = 1024
    skip_mask: tuple[bool, bool, bool, bool] = (True, True, True, True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        data = dict(payload)
        data["encoder_channels"] = tuple(data.get("encoder_channels", (64, 128, 256, 512)))
        data["skip_mask"] = tuple(bool(flag) for flag in data.get("skip_mask", (True, True, True, True)))
        return cls(**data)


@dataclass(slots=True)
class TrainConfig:
    device: str = "auto"
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    output_dir: str = "artifacts/runs"
    checkpoint_dir: str = "artifacts/checkpoints"
    log_every: int = 10
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: float | None = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainConfig":
        return cls(**payload)


@dataclass(slots=True)
class EvaluationConfig:
    output_dir: str = "artifacts/eval"
    num_visualizations: int = 4

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationConfig":
        return cls(**payload)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data_config = DataConfig.from_dict(payload.get("data", {}))
        model_payload = dict(payload.get("model", {}))
        model_payload.setdefault("num_classes", data_config.num_classes)
        model_config = ModelConfig.from_dict(model_payload)
        if model_config.num_classes != data_config.num_classes:
            raise ValueError("model.num_classes must match data.num_classes")
        return cls(
            experiment_name=payload["experiment_name"],
            data=data_config,
            model=model_config,
            train=TrainConfig.from_dict(payload.get("train", {})),
            evaluation=EvaluationConfig.from_dict(payload.get("evaluation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config, resolving an optional base_config recursively."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if "base_config" in payload:
        base_path = (config_path.parent / payload.pop("base_config")).resolve()
        base_config = load_experiment_config(base_path).to_dict()
        payload = _deep_merge(base_config, payload)

    payload.setdefault("experiment_name", config_path.stem)
    return ExperimentConfig.from_dict(payload)
