#!/usr/bin/env python3
"""Train a behavioral cloning model from demonstration data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shepherding.research import load_yaml_config
from shepherding.imitation import train_behavioral_cloning_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavioral cloning model")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--training-config", type=str, default="configs/imitation/training.yaml")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    train_cfg = load_yaml_config(Path(args.training_config))["training"]
    train_cfg["n_sheep"] = int(config["environment"]["n_sheep"])
    train_cfg["max_obstacles"] = int(config["environment"]["max_obstacles"])
    if args.dataset_path is not None:
        train_cfg["dataset_path"] = args.dataset_path
    if args.output_dir is not None:
        train_cfg["output_dir"] = args.output_dir
        train_cfg["model_path"] = str(Path(args.output_dir) / "behavioral_cloning.pkl")
    metrics = train_behavioral_cloning_model(
        dataset_path=Path(train_cfg["dataset_path"]),
        output_dir=Path(train_cfg["output_dir"]),
        train_config=train_cfg,
    )
    print(f"Validation RMSE: {metrics['rmse']:.4f}")
    print(f"Validation angle error (deg): {metrics['mean_angle_error_deg']:.2f}")


if __name__ == "__main__":
    main()
