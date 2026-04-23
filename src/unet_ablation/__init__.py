"""U-Net ablation starter package.

The package root stays lightweight so metadata preparation can run without
PyTorch installed. Training and model symbols are loaded lazily.
"""

from __future__ import annotations

__all__ = [
    "ExperimentConfig",
    "UNet",
    "build_unet",
    "load_experiment_config",
]


def __getattr__(name: str):
    if name in {"ExperimentConfig", "load_experiment_config"}:
        from unet_ablation.utils.config import ExperimentConfig, load_experiment_config

        globals().update(
            {
                "ExperimentConfig": ExperimentConfig,
                "load_experiment_config": load_experiment_config,
            }
        )
        return globals()[name]

    if name in {"UNet", "build_unet"}:
        from unet_ablation.models import UNet, build_unet

        globals().update({"UNet": UNet, "build_unet": build_unet})
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
