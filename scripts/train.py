#!/usr/bin/env python3
"""Train one segmentation experiment from a YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet_ablation.engine import train_experiment
from unet_ablation.utils import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)

    if args.seed is not None:
        config.train.seed = args.seed
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.checkpoint_dir is not None:
        config.train.checkpoint_dir = args.checkpoint_dir

    result = train_experiment(config)
    payload = {
        "experiment_name": result.experiment_name,
        "seed": result.seed,
        "run_dir": str(result.run_dir),
        "best_checkpoint": str(result.best_checkpoint),
        "best_metrics": result.best_metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
