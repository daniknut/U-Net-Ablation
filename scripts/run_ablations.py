#!/usr/bin/env python3
"""Train and evaluate all configured skip-connection ablations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unet_ablation.engine import evaluate_checkpoint, save_prediction_samples, train_experiment
from unet_ablation.utils import ensure_dir, load_experiment_config, save_json

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
        mean_iou_value = sum(float(row["mean_iou"]) for row in rows) / len(rows)
        pixel_accuracy_value = sum(float(row["pixel_accuracy"]) for row in rows) / len(rows)
        summary.append(
            {
                "experiment": experiment_name,
                "runs": len(rows),
                "mean_iou": mean_iou_value,
                "pixel_accuracy": pixel_accuracy_value,
            }
        )
    return sorted(summary, key=lambda row: str(row["experiment"]))


def _write_markdown_report(path: Path, summary: list[dict[str, object]], records: list[dict[str, object]]) -> None:
    lines = [
        "# Ablation Summary",
        "",
        "| Experiment | Runs | Mean IoU | Pixel Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['experiment']} | {row['runs']} | {float(row['mean_iou']):.4f} | {float(row['pixel_accuracy']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Experiment | Seed | Mean IoU | Pixel Accuracy | Checkpoint |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['experiment']} | {record['seed']} | {float(record['mean_iou']):.4f} | {float(record['pixel_accuracy']):.4f} | {record['checkpoint']} |"
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
            save_prediction_samples(config, result.best_checkpoint)

            records.append(
                {
                    "experiment": experiment_name,
                    "seed": seed,
                    "mean_iou": metrics["mean_iou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "checkpoint": str(result.best_checkpoint),
                    "run_dir": str(result.run_dir),
                }
            )

    summary = _aggregate(records)
    save_json(args.report_dir / "ablation_results.json", {"summary": summary, "runs": records})
    _write_markdown_report(args.report_dir / "ablation_summary.md", summary, records)
    print(json.dumps({"summary": summary, "runs": records}, indent=2))


if __name__ == "__main__":
    main()
