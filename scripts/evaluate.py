#!/usr/bin/env python3
"""Evaluate a saved checkpoint and export qualitative examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet_ablation.engine.trainer import evaluate_checkpoint, save_prediction_samples
from unet_ablation.utils.config import load_experiment_config
from unet_ablation.utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    if args.output_dir is not None:
        config.evaluation.output_dir = args.output_dir

    metrics = evaluate_checkpoint(config, args.checkpoint)
    sample_dir = save_prediction_samples(config, args.checkpoint)
    output_root = Path(sample_dir).parent
    save_json(
        output_root / "metrics.json",
        {
            "checkpoint": str(args.checkpoint),
            "metrics": metrics,
            "samples_dir": str(sample_dir),
        },
    )
    print(
        json.dumps(
            {
                "metrics": metrics,
                "samples_dir": str(sample_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
