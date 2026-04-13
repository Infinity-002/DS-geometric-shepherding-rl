#!/usr/bin/env python3
"""Run multi-seed baselines and ablations for HerdingEnv-v3."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401

from shepherding.research import (
    ResearchMetricsCallback,
    aggregate_results,
    build_curriculum_callback,
    build_feedforward_model,
    build_recurrent_model,
    collect_episode,
    create_significance_table,
    load_yaml_config,
    make_research_env,
    save_rows,
    save_summaries,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run research ablations for v3")
    parser.add_argument("--config", type=str, default="configs/research/v3.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/research_v3/ablation_runs")
    parser.add_argument("--save-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    env_cfg = config["environment"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    total_timesteps = args.total_timesteps or int(train_cfg["total_timesteps"])
    episodes = args.episodes or int(train_cfg["eval_episodes"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "run_name": "feedforward_domain_randomized",
            "model_type": "feedforward",
            "env_overrides": {
                "domain_randomization": True,
                "randomize_visibility": True,
                "randomize_goal": True,
                "randomize_obstacles": True,
                "randomize_dynamics": True,
                "visibility_radius": env_cfg["visibility_radius"],
            },
        },
        {
            "run_name": "recurrent_domain_randomized",
            "model_type": "recurrent",
            "env_overrides": {
                "domain_randomization": True,
                "randomize_visibility": True,
                "randomize_goal": True,
                "randomize_obstacles": True,
                "randomize_dynamics": True,
                "visibility_radius": env_cfg["visibility_radius"],
            },
        },
        {
            "run_name": "recurrent_no_randomization",
            "model_type": "recurrent",
            "env_overrides": {
                "domain_randomization": False,
                "randomize_visibility": False,
                "randomize_goal": False,
                "randomize_obstacles": False,
                "randomize_dynamics": False,
                "visibility_radius": env_cfg["visibility_radius"],
            },
        },
        {
            "run_name": "recurrent_full_visibility",
            "model_type": "recurrent",
            "env_overrides": {
                "domain_randomization": True,
                "randomize_visibility": False,
                "randomize_goal": True,
                "randomize_obstacles": True,
                "randomize_dynamics": True,
                "visibility_radius": env_cfg["grid_size"] * 2.0,
            },
        },
    ]

    all_rows: List[dict[str, Any]] = []
    all_summaries = []

    for variant in variants:
        for seed in args.seeds:
            run_label = f"{variant['run_name']}_seed{seed}"
            env_config = deepcopy(env_cfg)
            env_config.update(variant["env_overrides"])

            train_env = make_research_env(env_config, seed=seed, scenario="train")
            callback = [
                ResearchMetricsCallback(log_freq=2048, verbose=1),
                build_curriculum_callback(
                    total_timesteps,
                    train_cfg.get("curriculum"),
                    verbose=1,
                ),
            ]

            if variant["model_type"] == "recurrent":
                model = build_recurrent_model(
                    env=train_env,
                    ppo_config=config["ppo_recurrent"],
                    seed=seed,
                    tensorboard_log=f"{train_cfg['tensorboard_log']}/{variant['run_name']}",
                )
            else:
                model = build_feedforward_model(
                    env=train_env,
                    ppo_config=config["ppo_feedforward"],
                    seed=seed,
                    tensorboard_log=f"{train_cfg['tensorboard_log']}/{variant['run_name']}",
                )

            print("=" * 80)
            print(f"Running {run_label}")
            print(f"Total timesteps: {total_timesteps:,}")
            print("=" * 80)
            model.learn(total_timesteps=total_timesteps, callback=callback)

            if args.save_models:
                model_dir = output_dir / "models" / variant["run_name"]
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(model_dir / run_label))

            train_env.close()

            scenarios = [("train", name) for name in eval_cfg["train_scenarios"]]
            scenarios += [("unseen", name) for name in eval_cfg["unseen_scenarios"]]
            for split, scenario in scenarios:
                eval_env_config = deepcopy(env_config)
                eval_env_config["compute_expensive_metrics"] = True
                eval_env = make_research_env(eval_env_config, seed=seed, scenario=scenario)
                for episode_idx in range(episodes):
                    rows, summary = collect_episode(
                        env=eval_env,
                        model=model,
                        deterministic=bool(eval_cfg["deterministic"]),
                        max_steps=int(eval_env_config["max_steps"]),
                        run_name=variant["run_name"],
                        model_type=variant["model_type"],
                        seed=seed,
                        split=split,
                        scenario=scenario,
                        episode_idx=episode_idx,
                    )
                    all_rows.extend(rows)
                    all_summaries.append(summary)
                eval_env.close()

    save_rows(output_dir / "trajectories.csv", all_rows)
    save_summaries(output_dir / "episode_summaries.csv", all_summaries)
    aggregate_results(output_dir / "episode_summaries.csv", output_dir / "aggregate_metrics.csv")
    create_significance_table(
        output_dir / "episode_summaries.csv",
        output_dir / "significance_tests.csv",
    )
    write_json(
        output_dir / "metadata.json",
        {
            "config": args.config,
            "seeds": args.seeds,
            "total_timesteps": total_timesteps,
            "episodes_per_scenario": episodes,
            "variants": [variant["run_name"] for variant in variants],
        },
    )
    print(f"Saved ablation outputs to {output_dir}")


if __name__ == "__main__":
    main()
