"""Dataset and metadata helpers.

Metadata utilities are available without PyTorch. Dataset and dataloader
helpers are loaded lazily because they depend on the training stack.
"""

from __future__ import annotations

from unet_ablation.data.metadata import (
    SegmentationSample,
    discover_split_samples,
    load_samples,
    write_metadata_file,
)

__all__ = [
    "ADE20KDataset",
    "SegmentationSample",
    "build_dataloaders",
    "discover_split_samples",
    "load_samples",
    "write_metadata_file",
]


def __getattr__(name: str):
    if name in {"ADE20KDataset", "build_dataloaders"}:
        from unet_ablation.data.ade20k import ADE20KDataset, build_dataloaders

        globals().update(
            {
                "ADE20KDataset": ADE20KDataset,
                "build_dataloaders": build_dataloaders,
            }
        )
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
