#!/usr/bin/env python3
"""Evaluate a trained v2 PPO model and export benchmark-style summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import shepherding.envs  # noqa: F401
import gymnasium as gym

from shepherding.research.io import save_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a v2 PPO model for comparison plots")
    parser.add_argument("--model-path", type=str, default="scripts/models/ppo_herding_v2.zip")
    parser.add_argument("--run-name", type=str, default="rl_v2_ppo")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results/v2/benchmark")
    parser.add_argument("--visibility-radius", type=float, default=8.0)
    parser.add_argument("--n-obstacles", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = PPO.load(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for episode_idx in range(args.episodes):
        episode_seed = args.seed + episode_idx
        env = _make_env(
            seed=episode_seed,
            visibility_radius=args.visibility_radius,
            n_obstacles=args.n_obstacles,
        )
        obs, info = env.reset(seed=episode_seed)
        inner = env.unwrapped

        total_reward = 0.0
        dog_path_length = 0.0
        visible_sum = 0.0
        last_info = info

        for step in range(args.max_steps):
            prev_dog = inner.dog_pos.copy()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            dog_path_length += float(np.linalg.norm(inner.dog_pos - prev_dog))
            visible_sum += float(last_info.get("n_visible_sheep", 0)) / float(inner.n_sheep)
            if terminated or truncated:
                break

        summaries.append(
            {
                "run_name": args.run_name,
                "model_type": "v2_reinforcement",
                "seed": episode_seed,
                "split": "train",
                "scenario": "v2_default",
                "episode_idx": episode_idx,
                "success": int(last_info.get("all_at_goal", False)),
                "episode_length": int(last_info.get("step", step + 1)),
                "episode_return": float(total_reward),
                "mean_dist_to_goal": float(last_info.get("mean_dist_to_goal", 0.0)),
                "visible_ratio": float(last_info.get("n_visible_sheep", 0)) / float(inner.n_sheep),
                "avg_visibility_ratio": float(visible_sum) / float(step + 1),
                "flock_hull_area": np.nan,
                "stray_count": np.nan,
                "collision_count": np.nan,
                "collision_event_count": np.nan,
                "dog_path_length": float(dog_path_length),
                "avg_reward_base": np.nan,
                "avg_reward_progress": np.nan,
                "avg_reward_worst_sheep": np.nan,
                "avg_reward_visibility_loss": np.nan,
                "avg_reward_visibility_gain": np.nan,
                "avg_reward_zero_visibility": np.nan,
                "avg_reward_stray": np.nan,
                "avg_reward_drive": np.nan,
                "avg_reward_collision": np.nan,
                "avg_reward_success_bonus": np.nan,
            }
        )
        env.close()

    save_rows(output_dir / "episode_summaries.csv", summaries)
    aggregate = _aggregate(summaries)
    save_rows(output_dir / "aggregate_metrics.csv", aggregate)
    write_json(
        output_dir / "metadata.json",
        {
            "model_path": args.model_path,
            "run_name": args.run_name,
            "episodes": args.episodes,
            "seed_start": args.seed,
            "visibility_radius": args.visibility_radius,
            "n_obstacles": args.n_obstacles,
        },
    )
    print(f"Saved v2 benchmark outputs to {output_dir}")


def _make_env(seed: int, visibility_radius: float, n_obstacles: int) -> gym.Env:
    if n_obstacles == 0:
        obstacles = []
    elif n_obstacles == 2:
        obstacles = None
    else:
        obstacles = [(3.0 + i * 4.0, 5.0 + i * 2.5, 3.0, 0.8) for i in range(n_obstacles)]
    env = gym.make(
        "HerdingEnv-v2",
        visibility_radius=visibility_radius,
        obstacles=obstacles,
    )
    env.reset(seed=seed)
    return env


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    success = np.array([float(r["success"]) for r in rows], dtype=np.float32)
    returns = np.array([float(r["episode_return"]) for r in rows], dtype=np.float32)
    lengths = np.array([float(r["episode_length"]) for r in rows], dtype=np.float32)
    mean_dist = np.array([float(r["mean_dist_to_goal"]) for r in rows], dtype=np.float32)
    vis = np.array([float(r["avg_visibility_ratio"]) for r in rows], dtype=np.float32)
    path = np.array([float(r["dog_path_length"]) for r in rows], dtype=np.float32)
    first = rows[0]
    return [
        {
            "run_name": first["run_name"],
            "model_type": first["model_type"],
            "split": "train",
            "scenario": "v2_default",
            "success_rate": float(np.mean(success)),
            "mean_episode_return": float(np.mean(returns)),
            "std_episode_return": float(np.std(returns)),
            "mean_episode_length": float(np.mean(lengths)),
            "mean_dist_to_goal": float(np.mean(mean_dist)),
            "mean_visible_ratio": float(np.mean(vis)),
            "mean_avg_visibility": float(np.mean(vis)),
            "mean_hull_area": np.nan,
            "mean_stray_count": np.nan,
            "mean_collision_count": np.nan,
            "mean_dog_path_length": float(np.mean(path)),
            "seeds": int(len(rows)),
        }
    ]


if __name__ == "__main__":
    main()
