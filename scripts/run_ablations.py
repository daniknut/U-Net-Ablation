#!/usr/bin/env python3
"""Train and evaluate all configured skip-connection ablations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet_ablation.engine.trainer import (
    evaluate_checkpoint,
    save_prediction_samples,
    train_experiment,
)
from unet_ablation.utils.config import load_experiment_config
from unet_ablation.utils.io import save_json
from unet_ablation.utils.runtime import ensure_dir

EXPERIMENTS = ("full_skip", "highres_skip_only", "no_skip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=str, default="artifacts/runs")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints")
    parser.add_argument("--eval-dir", type=str, default="artifacts/eval")
    return parser.parse_args()


def _aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["experiment"])].append(record)

    summary: list[dict[str, object]] = []
    for experiment_name, rows in grouped.items():
        mean_ious = [float(row["mean_iou"]) for row in rows]
        pixel_accuracies = [float(row["pixel_accuracy"]) for row in rows]
        val_losses = [float(row["val_loss"]) for row in rows]
        best_run = max(rows, key=lambda row: float(row["mean_iou"]))
        summary.append(
            {
                "experiment": experiment_name,
                "runs": len(rows),
                "mean_iou_mean": mean(mean_ious),
                "mean_iou_std": pstdev(mean_ious) if len(mean_ious) > 1 else 0.0,
                "pixel_accuracy_mean": mean(pixel_accuracies),
                "pixel_accuracy_std": pstdev(pixel_accuracies) if len(pixel_accuracies) > 1 else 0.0,
                "val_loss_mean": mean(val_losses),
                "val_loss_std": pstdev(val_losses) if len(val_losses) > 1 else 0.0,
                "best_seed": int(best_run["seed"]),
                "best_checkpoint": str(best_run["checkpoint"]),
            }
        )
    return sorted(summary, key=lambda row: float(row["mean_iou_mean"]), reverse=True)


def _write_markdown_report(path: Path, summary: list[dict[str, object]], records: list[dict[str, object]]) -> None:
    lines = [
        "# Ablation Summary",
        "",
        "| Rank | Experiment | Runs | Mean IoU | Pixel Accuracy | Val Loss | Best Seed |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(summary, start=1):
        lines.append(
            f"| {rank} | {row['experiment']} | {row['runs']} | "
            f"{float(row['mean_iou_mean']):.4f} +/- {float(row['mean_iou_std']):.4f} | "
            f"{float(row['pixel_accuracy_mean']):.4f} +/- {float(row['pixel_accuracy_std']):.4f} | "
            f"{float(row['val_loss_mean']):.4f} +/- {float(row['val_loss_std']):.4f} | "
            f"{int(row['best_seed'])} |"
        )

    lines.extend(
        [
            "",
            "Best checkpoint per variant is recorded in `ablation_results.json` for exact reuse.",
            "",
            "## Per-Run Results",
            "",
            "| Experiment | Seed | Mean IoU | Pixel Accuracy | Val Loss | Checkpoint | Samples |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in sorted(records, key=lambda row: (str(row["experiment"]), int(row["seed"]))):
        lines.append(
            f"| {record['experiment']} | {record['seed']} | {float(record['mean_iou']):.4f} | "
            f"{float(record['pixel_accuracy']):.4f} | {float(record['val_loss']):.4f} | "
            f"{record['checkpoint']} | {record['samples_dir']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.report_dir)

    records: list[dict[str, object]] = []
    for experiment_name in EXPERIMENTS:
        config_path = args.config_dir / f"{experiment_name}.yaml"
        for seed in args.seeds:
            config = load_experiment_config(config_path)
            config.train.seed = seed
            config.train.output_dir = args.output_dir
            config.train.checkpoint_dir = args.checkpoint_dir
            config.evaluation.output_dir = args.eval_dir

            result = train_experiment(config)
            metrics = evaluate_checkpoint(config, result.best_checkpoint)
            sample_dir = save_prediction_samples(config, result.best_checkpoint)

            records.append(
                {
                    "experiment": experiment_name,
                    "seed": seed,
                    "val_loss": metrics["loss"],
                    "mean_iou": metrics["mean_iou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "checkpoint": str(result.best_checkpoint),
                    "run_dir": str(result.run_dir),
                    "summary_path": str(result.summary_path),
                    "epochs_completed": result.epochs_completed,
                    "samples_dir": str(sample_dir),
                }
            )

    summary = _aggregate(records)
    save_json(args.report_dir / "ablation_results.json", {"summary": summary, "runs": records})
    _write_markdown_report(args.report_dir / "ablation_summary.md", summary, records)
    print(json.dumps({"summary": summary, "runs": records}, indent=2))


if __name__ == "__main__":
    main()
