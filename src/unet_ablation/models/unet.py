"""Configurable U-Net implementation and ablation presets."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Sequence

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from unet_ablation.utils.config import ModelConfig


class UNetVariant(str, Enum):
    """Named skip-connection ablation presets."""

    BASELINE = "baseline"
    FULL_SKIP = "full_skip"
    HIGHRES_SKIP_ONLY = "highres_skip_only"
    NO_SKIP = "no_skip"


_VARIANT_SKIP_MASKS: dict[UNetVariant, tuple[bool, bool, bool, bool]] = {
    UNetVariant.BASELINE: (True, True, True, True),
    UNetVariant.FULL_SKIP: (True, True, True, True),
    UNetVariant.HIGHRES_SKIP_ONLY: (True, True, False, False),
    UNetVariant.NO_SKIP: (False, False, False, False),
}


def resolve_skip_mask(
    variant: str | UNetVariant | None,
    skip_mask: Sequence[bool] | None,
    num_stages: int,
) -> tuple[bool, ...]:
    """Resolve the effective skip mask from a named variant or explicit override."""

    if skip_mask is not None:
        resolved = tuple(bool(flag) for flag in skip_mask)
        if len(resolved) != num_stages:
            raise ValueError("skip_mask length must match the number of encoder stages")
        return resolved

    resolved_variant = UNetVariant(variant or UNetVariant.BASELINE.value)
    preset_mask = _VARIANT_SKIP_MASKS[resolved_variant]
    if len(preset_mask) != num_stages:
        raise ValueError("variant preset length must match the number of encoder stages")
    return preset_mask


class DoubleConv(nn.Module):
    """Two convolution blocks with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Four-stage U-Net with configurable skip connections."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        encoder_channels: Sequence[int],
        bottleneck_channels: int,
        skip_mask: Sequence[bool] | None = None,
        variant: str | UNetVariant | None = None,
    ) -> None:
        super().__init__()
        self.encoder_channels = tuple(int(channel) for channel in encoder_channels)
        self.variant = UNetVariant(variant or UNetVariant.BASELINE.value)
        self.skip_mask = resolve_skip_mask(self.variant, skip_mask, len(self.encoder_channels))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_blocks = nn.ModuleList()
        previous_channels = in_channels
        for channels in self.encoder_channels:
            self.down_blocks.append(DoubleConv(previous_channels, channels))
            previous_channels = channels

        self.bottleneck = DoubleConv(self.encoder_channels[-1], bottleneck_channels)

        self.upconvs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        current_channels = bottleneck_channels
        for skip_channels in reversed(self.encoder_channels):
            self.upconvs.append(
                nn.ConvTranspose2d(current_channels, skip_channels, kernel_size=2, stride=2)
            )
            # Disabled skip connections are replaced with zeros so decoder widths stay constant.
            self.up_blocks.append(DoubleConv(skip_channels * 2, skip_channels))
            current_channels = skip_channels

        self.head = nn.Conv2d(self.encoder_channels[0], num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[float, ...] | tuple[bool, ...]]]:
        encoder_features: list[torch.Tensor] = []
        for block in self.down_blocks:
            x = block(x)
            encoder_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_norms: list[float] = [0.0] * len(self.skip_mask)
        effective_mask: list[bool] = [False] * len(self.skip_mask)
        for decode_index, (upconv, block, skip_feature) in enumerate(
            zip(self.upconvs, self.up_blocks, reversed(encoder_features))
        ):
            x = upconv(x)
            if x.shape[-2:] != skip_feature.shape[-2:]:
                x = F.interpolate(x, size=skip_feature.shape[-2:], mode="bilinear", align_corners=False)

            stage_index = len(self.skip_mask) - 1 - decode_index
            use_skip = self.skip_mask[stage_index]
            effective_mask[stage_index] = use_skip
            skip_tensor = skip_feature if use_skip else torch.zeros_like(skip_feature)
            skip_norms[stage_index] = float(skip_tensor.abs().mean().detach().cpu())

            x = torch.cat((x, skip_tensor), dim=1)
            x = block(x)

        logits = self.head(x)
        if not return_debug:
            return logits

        debug = {
            "skip_enabled": tuple(effective_mask),
            "skip_norms": tuple(skip_norms),
        }
        return logits, debug


def build_unet(config: "ModelConfig") -> UNet:
    """Instantiate the U-Net from config."""

    return UNet(
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        encoder_channels=config.encoder_channels,
        bottleneck_channels=config.bottleneck_channels,
        variant=config.variant,
        skip_mask=config.skip_mask,
    )


def build_model(config: "ModelConfig") -> nn.Module:
    """Instantiate the configured segmentation model behind a shared interface."""

    if config.architecture != "unet":
        raise ValueError(f"Unsupported architecture: {config.architecture}")
    return build_unet(config)
