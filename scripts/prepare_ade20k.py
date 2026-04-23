#!/usr/bin/env python3
"""Validate an ADE20K-style dataset layout and create metadata JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet_ablation.data import discover_split_samples, write_metadata_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ade20k"))
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--train-images-subdir", default="images/training")
    parser.add_argument("--train-masks-subdir", default="annotations/training")
    parser.add_argument("--val-images-subdir", default="images/validation")
    parser.add_argument("--val-masks-subdir", default="annotations/validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_dir or args.root / "metadata"

    train_samples = discover_split_samples(
        args.root / args.train_images_subdir,
        args.root / args.train_masks_subdir,
    )
    val_samples = discover_split_samples(
        args.root / args.val_images_subdir,
        args.root / args.val_masks_subdir,
    )

    write_metadata_file(metadata_dir / "train.jsonl", args.root, train_samples)
    write_metadata_file(metadata_dir / "val.jsonl", args.root, val_samples)

    summary = {
        "root": str(args.root.resolve()),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }
    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
