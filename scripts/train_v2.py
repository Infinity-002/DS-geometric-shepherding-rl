#!/usr/bin/env python3
"""
train_v2.py – Train a PPO agent on HerdingEnv-v2.

New features over train.py
--------------------------
* Trains the enhanced environment: limited visibility, obstacles, autonomous
  sheep dynamics.
* Extra CLI flags: --visibility-radius, --n-obstacles (use 0 to disable).
* Saves model to ``models/ppo_herding_v2.zip``.

Usage
-----
    python train_v2.py                                # defaults
    python train_v2.py --total-timesteps 2e6         # longer run
    python train_v2.py --visibility-radius 10 --n-obstacles 0  # no obstacles
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.spatial import ConvexHull
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym
import shepherding.envs  # noqa: F401 – registers HerdingEnv-v0 and v2


# ======================================================================
# Custom Callback
# ======================================================================

class ShepherdingV2Callback(BaseCallback):
    """Log shepherding metrics plus v2-specific visibility metric.

    Tracked metrics
    ---------------
    * **mean_success_rate**      – rolling fraction of episodes that ended
      with all sheep at the goal.
    * **mean_flock_perimeter**   – rolling mean convex-hull perimeter of the
      flock at episode end.
    * **mean_visible_sheep_pct** – average fraction of sheep visible to the
      dog at episode end (v2-specific).
    """

    def __init__(self, log_freq: int = 2048, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._successes: List[bool] = []
        self._perimeters: List[float] = []
        self._vis_pcts: List[float] = []

    def _on_step(self) -> bool:
        infos: List[Dict] = self.locals.get("infos", [])
        dones: np.ndarray = self.locals.get("dones", np.array([]))

        for idx, done in enumerate(dones):
            if done and idx < len(infos):
                info = infos[idx]
                self._successes.append(info.get("all_at_goal", False))

                # Fetch raw env for sheep_pos and n_visible_sheep
                env = self.training_env.envs[idx]
                # Unwrap Monitor (and any other wrappers) to reach the raw env
                inner = env
                while hasattr(inner, "env"):
                    inner = inner.env
                sheep_pos = getattr(inner, "sheep_pos", None)
                n_sheep = getattr(inner, "n_sheep", 10)  # safe default = 10

                if sheep_pos is not None and sheep_pos.shape[0] >= 3:
                    try:
                        hull = ConvexHull(sheep_pos)
                        self._perimeters.append(float(hull.area))
                    except Exception:
                        pass

                n_vis = info.get("n_visible_sheep", n_sheep)
                self._vis_pcts.append(float(n_vis) / max(n_sheep, 1))

        if self.num_timesteps % self.log_freq == 0 and self._successes:
            mean_sr = float(np.mean(self._successes[-100:]))
            mean_pm = float(np.mean(self._perimeters[-100:])) if self._perimeters else 0.0
            mean_vis = float(np.mean(self._vis_pcts[-100:])) if self._vis_pcts else 0.0
            self.logger.record("shepherding/mean_success_rate", mean_sr)
            self.logger.record("shepherding/mean_flock_perimeter", mean_pm)
            self.logger.record("shepherding/mean_visible_sheep_pct", mean_vis)
            if self.verbose:
                print(
                    f"[Step {self.num_timesteps:>8d}]  "
                    f"SR: {mean_sr:.3f}  |  "
                    f"Perim: {mean_pm:.2f}  |  "
                    f"Visible: {mean_vis:.1%}"
                )
        return True


# ======================================================================
# Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on HerdingEnv-v2")
    parser.add_argument("--total-timesteps", type=float, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument(
        "--visibility-radius",
        type=float,
        default=8.0,
        help="Dog visibility radius in world units (default: 8.0)",
    )
    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=2,
        help="Number of default obstacles (0 = none, default: 2)",
    )
    return parser.parse_args()


def make_env(
    seed: int = 0,
    visibility_radius: float = 8.0,
    n_obstacles: int = 2,
) -> gym.Env:
    """Create and wrap HerdingEnv-v2 with Monitor."""
    if n_obstacles == 0:
        obstacles = []
    elif n_obstacles == 2:
        obstacles = None  # use the built-in default layout
    else:
        # Generate n_obstacles evenly-spaced horizontal walls
        obstacles = [
            (3.0 + i * 4.0, 5.0 + i * 2.5, 3.0, 0.8)
            for i in range(n_obstacles)
        ]

    env = gym.make(
        "HerdingEnv-v2",
        visibility_radius=visibility_radius,
        obstacles=obstacles,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main() -> None:
    args = parse_args()
    total_timesteps = int(args.total_timesteps)

    print("=" * 60)
    print("  Shepherding RL v2 – PPO Training")
    print(f"  Total timesteps    : {total_timesteps:,}")
    print(f"  Seed               : {args.seed}")
    print(f"  Visibility radius  : {args.visibility_radius}")
    print(f"  Obstacles          : {args.n_obstacles}")
    print("=" * 60)

    env = make_env(
        seed=args.seed,
        visibility_radius=args.visibility_radius,
        n_obstacles=args.n_obstacles,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    callback = ShepherdingV2Callback(log_freq=2048, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "ppo_herding_v2"
    model.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
