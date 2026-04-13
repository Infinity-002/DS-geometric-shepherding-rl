#!/usr/bin/env python3
"""Evaluate v3 models, export trajectories, and summarize results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401

from shepherding.baselines import HeuristicShepherdAgent
from shepherding.research import (
    aggregate_results,
    collect_episode,
    load_yaml_config,
    load_model,
    make_research_env,
    save_rows,
    save_summaries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HerdingEnv-v3 models")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["feedforward", "recurrent", "heuristic", "behavioral_cloning"],
        required=True,
    )
    parser.add_argument("--run-name", type=str, default="evaluation")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/research_v3")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    env_cfg = config["environment"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    episodes = args.episodes or int(train_cfg["eval_episodes"])
    if args.model_type == "heuristic":
        model = HeuristicShepherdAgent(
            n_sheep=int(env_cfg["n_sheep"]),
            max_obstacles=int(env_cfg["max_obstacles"]),
            grid_size=float(env_cfg["grid_size"]),
            visibility_radius=float(env_cfg["visibility_radius"]),
            flee_radius=float(env_cfg["flee_radius"]),
            success_radius=float(env_cfg["success_radius"]),
            use_cluster_targets=bool(
                config.get("imitation", {}).get("expert", {}).get("use_cluster_targets", False)
            ),
        )
    else:
        if args.model_path is None:
            raise ValueError(
                "--model-path is required for feedforward, recurrent, and behavioral_cloning evaluation."
            )
        model = load_model(args.model_type, args.model_path)
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_env_cfg = dict(env_cfg)
    eval_env_cfg["compute_expensive_metrics"] = True

    all_rows: List[dict[str, Any]] = []
    all_summaries = []

    scenarios = [("train", name) for name in eval_cfg["train_scenarios"]]
    scenarios += [("unseen", name) for name in eval_cfg["unseen_scenarios"]]

    for split, scenario in scenarios:
        env = make_research_env(eval_env_cfg, seed=args.seed, scenario=scenario)
        for episode_idx in range(episodes):
            episode_seed = args.seed + episode_idx
            rows, summary = collect_episode(
                env=env,
                model=model,
                deterministic=bool(eval_cfg["deterministic"]),
                max_steps=int(eval_env_cfg["max_steps"]),
                run_name=args.run_name,
                model_type=args.model_type,
                seed=episode_seed,
                split=split,
                scenario=scenario,
                episode_idx=episode_idx,
            )
            all_rows.extend(rows)
            all_summaries.append(summary)
        env.close()

    save_rows(output_dir / "trajectories.csv", all_rows)
    save_summaries(output_dir / "episode_summaries.csv", all_summaries)
    aggregate_results(
        output_dir / "episode_summaries.csv",
        output_dir / "aggregate_metrics.csv",
    )
    write_json(
        output_dir / "metadata.json",
        {
            "run_name": args.run_name,
            "model_type": args.model_type,
            "model_path": args.model_path,
            "episodes_per_scenario": episodes,
            "seed_start": args.seed,
        },
    )
    print(f"Saved evaluation outputs to {output_dir}")


if __name__ == "__main__":
    main()
