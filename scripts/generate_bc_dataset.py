#!/usr/bin/env python3
"""Generate demonstration data for behavioral cloning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shepherding.imitation import collect_demonstrations
from shepherding.research import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate behavioral cloning dataset")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--dataset-config", type=str, default="configs/imitation/dataset.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    dataset_cfg = load_yaml_config(Path(args.dataset_config))["dataset"]
    if args.output_dir is not None:
        dataset_cfg["output_dir"] = args.output_dir
    output_dir = Path(dataset_cfg["output_dir"])
    metadata = collect_demonstrations(
        env_config=config["environment"],
        collection_config=dataset_cfg,
        output_dir=output_dir,
    )
    print(f"Saved {metadata['collected_steps']} demonstrations to {output_dir}")


if __name__ == "__main__":
    main()
