"""Utility exports.

Configuration and file I/O helpers are importable without PyTorch. Runtime
helpers are loaded lazily because they depend on the training stack.
"""

from __future__ import annotations

from unet_ablation.utils.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    load_experiment_config,
)
from unet_ablation.utils.io import append_jsonl, save_json

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainConfig",
    "append_jsonl",
    "ensure_dir",
    "load_experiment_config",
    "resolve_device",
    "save_json",
    "set_seed",
]


def __getattr__(name: str):
    if name in {"ensure_dir", "resolve_device", "set_seed"}:
        from unet_ablation.utils.runtime import ensure_dir, resolve_device, set_seed

        globals().update(
            {
                "ensure_dir": ensure_dir,
                "resolve_device": resolve_device,
                "set_seed": set_seed,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
