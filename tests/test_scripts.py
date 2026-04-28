from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def _write_dummy_split(root: Path, split_name: str, samples: int = 2) -> None:
    image_dir = root / "images" / split_name
    mask_dir = root / "annotations" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(samples):
        image_array = np.full((48, 48, 3), fill_value=index * 32 + 64, dtype=np.uint8)
        mask_array = np.zeros((48, 48), dtype=np.uint8)
        mask_array[:, :16] = 1
        mask_array[:, 16:32] = 2

        Image.fromarray(image_array).save(image_dir / f"sample_{index}.jpg")
        Image.fromarray(mask_array).save(mask_dir / f"sample_{index}.png")


def test_run_ablations_writes_ranked_reports(tmp_path: Path, smoke_config_path: Path) -> None:
    dataset_root = tmp_path / "ade20k"
    base_output = tmp_path / "outputs"
    config_dir = tmp_path / "configs" / "experiments"
    config_dir.mkdir(parents=True, exist_ok=True)

    smoke_payload = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
    smoke_payload["data"]["root"] = str(dataset_root)
    smoke_payload["data"]["metadata_dir"] = str(dataset_root / "metadata")
    smoke_payload["train"]["output_dir"] = str(base_output / "runs")
    smoke_payload["train"]["checkpoint_dir"] = str(base_output / "checkpoints")
    smoke_payload["evaluation"]["output_dir"] = str(base_output / "eval")
    smoke_payload["train"]["epochs"] = 1

    base_config_path = tmp_path / "configs" / "base.yaml"
    base_config_path.parent.mkdir(parents=True, exist_ok=True)
    base_config_path.write_text(yaml.safe_dump(smoke_payload), encoding="utf-8")

    for experiment_name, variant in (
        ("full_skip", "full_skip"),
        ("highres_skip_only", "highres_skip_only"),
        ("no_skip", "no_skip"),
    ):
        experiment_path = config_dir / f"{experiment_name}.yaml"
        experiment_path.write_text(
            yaml.safe_dump(
                {
                    "base_config": "../base.yaml",
                    "experiment_name": experiment_name,
                    "model": {"variant": variant},
                }
            ),
            encoding="utf-8",
        )

    for split_name in ("training", "validation"):
        _write_dummy_split(dataset_root, split_name)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablations.py",
            "--config-dir",
            str(config_dir),
            "--seeds",
            "0",
            "1",
            "--report-dir",
            str(tmp_path / "reports"),
            "--output-dir",
            str(base_output / "runs"),
            "--checkpoint-dir",
            str(base_output / "checkpoints"),
            "--eval-dir",
            str(base_output / "eval"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert len(payload["summary"]) == 3
    assert len(payload["runs"]) == 6
    assert all("mean_iou_mean" in row for row in payload["summary"])
    assert all("mean_iou_std" in row for row in payload["summary"])
    assert all("samples_dir" in row for row in payload["runs"])

    markdown_report = (tmp_path / "reports" / "ablation_summary.md").read_text(encoding="utf-8")
    assert "Rank | Experiment" in markdown_report
    assert "Per-Run Results" in markdown_report
