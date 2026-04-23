from __future__ import annotations

import torch

from unet_ablation.models import UNet


def test_all_variants_preserve_output_shape() -> None:
    variants = {
        "full_skip": (True, True, True, True),
        "highres_skip_only": (True, True, False, False),
        "no_skip": (False, False, False, False),
    }
    inputs = torch.randn(2, 3, 256, 256)

    for skip_mask in variants.values():
        model = UNet(
            in_channels=3,
            num_classes=5,
            encoder_channels=(16, 32, 64, 128),
            bottleneck_channels=256,
            skip_mask=skip_mask,
        )
        logits = model(inputs)
        assert logits.shape == (2, 5, 256, 256)


def test_debug_output_reflects_skip_mask_and_zero_fills_disabled_skips() -> None:
    inputs = torch.randn(1, 3, 128, 128)
    model = UNet(
        in_channels=3,
        num_classes=4,
        encoder_channels=(16, 32, 64, 128),
        bottleneck_channels=256,
        skip_mask=(True, True, False, False),
    )

    _, debug = model(inputs, return_debug=True)

    assert debug["skip_enabled"] == (True, True, False, False)
    assert debug["skip_norms"][0] > 0.0
    assert debug["skip_norms"][1] > 0.0
    assert debug["skip_norms"][2] == 0.0
    assert debug["skip_norms"][3] == 0.0
