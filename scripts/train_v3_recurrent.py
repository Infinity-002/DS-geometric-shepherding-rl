#!/usr/bin/env python3
"""Train the research v3 environment with recurrent PPO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401

from shepherding.research import (
    ResearchMetricsCallback,
    build_curriculum_callback,
    build_recurrent_model,
    load_yaml_config,
    make_research_env,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recurrent PPO on HerdingEnv-v3")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--scenario", type=str, default="train")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    env_cfg = config["environment"]
    train_cfg = config["training"]
    ppo_cfg = config["ppo_recurrent"]

    total_timesteps = args.total_timesteps or int(train_cfg["total_timesteps"])
    run_name = args.run_name or f"recurrent_seed{args.seed}"

    env = make_research_env(env_cfg, seed=args.seed, scenario=args.scenario)
    tensorboard_log = f"{train_cfg['tensorboard_log']}/recurrent"
    model = build_recurrent_model(
        env=env,
        ppo_config=ppo_cfg,
        seed=args.seed,
        tensorboard_log=tensorboard_log,
    )
    callback = [
        ResearchMetricsCallback(log_freq=2048, verbose=1),
        build_curriculum_callback(total_timesteps, train_cfg.get("curriculum"), verbose=1),
    ]

    print("=" * 72)
    print("  Shepherding RL v3 – Recurrent PPO")
    print(f"  Run name         : {run_name}")
    print(f"  Seed             : {args.seed}")
    print(f"  Scenario         : {args.scenario}")
    print(f"  Total timesteps  : {total_timesteps:,}")
    print("=" * 72)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_dir = Path(train_cfg["save_dir"]) / "recurrent"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / run_name
    model.save(str(model_path))
    write_json(
        save_dir / f"{run_name}_metadata.json",
        {
            "run_name": run_name,
            "model_type": "recurrent",
            "seed": args.seed,
            "scenario": args.scenario,
            "total_timesteps": total_timesteps,
            "config_path": args.config,
        },
    )
    print(f"\nSaved model to {model_path}.zip")
    env.close()


if __name__ == "__main__":
    main()
