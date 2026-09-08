#!/usr/bin/env python3
"""Train the research v3 environment with feedforward PPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401

from shepherding.research import load_yaml_config, train_v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feedforward PPO on HerdingEnv-v3")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_v3(
        config=load_yaml_config(Path(args.config)),
        model_type="feedforward",
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        run_name=args.run_name,
        scenario=args.scenario,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
