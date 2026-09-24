"""Evaluate the v1 (HerdingEnv-v0) and v2 (HerdingEnv-v2) PPO models.

    uv run python scripts/evaluate_legacy_versions.py --version v1
    uv run python scripts/evaluate_legacy_versions.py --version v2

Each version is scored in the setting it was trained on (goal fixed at (18, 18))
and with the goal drawn uniformly from the same region v3 uses. Neither
observation contains the goal, so the second setting measures how much of the
policy is tied to the fixed corner. A uniform random policy on the training
setting gives a floor for comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

import shepherding.envs  # noqa: E402,F401

VERSIONS = {
    "v1": ("HerdingEnv-v0", "models/legacy_v1/ppo_herding.zip"),
    "v2": ("HerdingEnv-v2", "scripts/models/ppo_herding_v2.zip"),
}
GOAL_LOW, GOAL_HIGH = 2.4, 17.6


def bootstrap_ci(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(env_id: str, policy, goal_mode: str, episodes: int, seed_start: int) -> pd.DataFrame:
    rows = []
    goal_rng = np.random.default_rng(seed_start)
    for ep in range(episodes):
        goal = (18.0, 18.0)
        if goal_mode == "random":
            goal = tuple(goal_rng.uniform(GOAL_LOW, GOAL_HIGH, size=2).round(3))
        env = gym.make(env_id, goal=goal)
        inner = env.unwrapped
        obs, _ = env.reset(seed=seed_start + ep)
        env.action_space.seed(seed_start + ep)
        steps = 0
        while True:
            action = policy(obs, env)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        dists = np.linalg.norm(inner.sheep_pos - inner.goal, axis=1)
        rows.append(
            {
                "success": float(terminated),
                "fraction_at_goal": float(np.mean(dists < inner.success_radius)),
                "mean_dist_to_goal": float(dists.mean()),
                "episode_length": steps,
            }
        )
        env.close()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=sorted(VERSIONS), required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--output-dir", default="results/legacy")
    args = parser.parse_args()
    torch.set_num_threads(1)

    env_id, model_path = VERSIONS[args.version]
    model = PPO.load(str(ROOT / model_path), device="cpu")
    ppo = lambda obs, env: model.predict(obs, deterministic=True)[0]  # noqa: E731
    rand = lambda obs, env: env.action_space.sample()  # noqa: E731

    settings = [
        ("ppo", "fixed", ppo),
        ("ppo", "random", ppo),
        ("random_policy", "fixed", rand),
    ]
    summary = []
    for agent, goal_mode, policy in settings:
        df = run(env_id, policy, goal_mode, args.episodes, args.seed_start)
        lo, hi = bootstrap_ci(df.fraction_at_goal.to_numpy())
        summary.append(
            {
                "version": args.version,
                "env": env_id,
                "agent": agent,
                "goal": goal_mode,
                "episodes": len(df),
                "success_rate": df.success.mean(),
                "fraction_at_goal": df.fraction_at_goal.mean(),
                "fraction_at_goal_ci_low": lo,
                "fraction_at_goal_ci_high": hi,
                "mean_dist_to_goal": df.mean_dist_to_goal.mean(),
                "mean_episode_length": df.episode_length.mean(),
            }
        )
        print(summary[-1], flush=True)

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / f"{args.version}_report.csv", index=False)
    print(f"Wrote {out / f'{args.version}_report.csv'}")


if __name__ == "__main__":
    main()
