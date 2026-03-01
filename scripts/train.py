#!/usr/bin/env python3
"""
train.py – Train a PPO agent on the HerdingEnv.

Usage
-----
    python train.py                        # default 500 000 steps
    python train.py --total-timesteps 1e6  # custom
    python train.py --seed 42              # reproducible

Logs *Mean Success Rate* and *Mean Flock Perimeter* via a custom callback.
Saves the trained model to ``models/ppo_herding.zip``.
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
import shepherding.envs  # noqa: F401 – triggers HerdingEnv registration


# ======================================================================
# Custom Callback
# ======================================================================

class ShepherdingMetricsCallback(BaseCallback):
    """Log shepherding-specific metrics every *log_freq* steps.

    Tracked metrics
    ---------------
    * **mean_success_rate** – rolling fraction of episodes that ended with
      all sheep at the goal.
    * **mean_flock_perimeter** – rolling mean of the convex-hull perimeter
      of the flock at each episode end.
    """

    def __init__(
        self,
        log_freq: int = 2048,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.log_freq: int = log_freq
        self._successes: List[bool] = []
        self._perimeters: List[float] = []

    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        infos: List[Dict] = self.locals.get("infos", [])
        dones: np.ndarray = self.locals.get("dones", np.array([]))

        for idx, done in enumerate(dones):
            if done and idx < len(infos):
                info = infos[idx]
                self._successes.append(info.get("all_at_goal", False))

                # Perimeter from the raw env (not the Monitor wrapper)
                env = self.training_env.envs[idx]
                sheep_pos = getattr(env, "sheep_pos", None)
                if sheep_pos is None:
                    # Unwrap Monitor if necessary
                    inner = getattr(env, "env", None)
                    sheep_pos = getattr(inner, "sheep_pos", None)
                if sheep_pos is not None and sheep_pos.shape[0] >= 3:
                    try:
                        hull = ConvexHull(sheep_pos)
                        self._perimeters.append(float(hull.area))  # 2-D perimeter
                    except Exception:
                        pass

        if self.num_timesteps % self.log_freq == 0 and self._successes:
            mean_sr = float(np.mean(self._successes[-100:]))
            mean_pm = (
                float(np.mean(self._perimeters[-100:]))
                if self._perimeters
                else 0.0
            )
            self.logger.record("shepherding/mean_success_rate", mean_sr)
            self.logger.record("shepherding/mean_flock_perimeter", mean_pm)
            if self.verbose:
                print(
                    f"[Step {self.num_timesteps:>8d}]  "
                    f"Success rate: {mean_sr:.3f}  |  "
                    f"Flock perimeter: {mean_pm:.2f}"
                )
        return True


# ======================================================================
# Main
# ======================================================================

def make_env(seed: int = 0) -> gym.Env:
    """Create and wrap HerdingEnv with Monitor."""
    env = gym.make("HerdingEnv-v0")
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on HerdingEnv")
    parser.add_argument(
        "--total-timesteps",
        type=float,
        default=2_000_000,
        help="Total training timesteps (default: 2 000 000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save the trained model (default: models/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_timesteps = int(args.total_timesteps)

    print("=" * 60)
    print("  Shepherding RL – PPO Training")
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Seed            : {args.seed}")
    print("=" * 60)

    env = make_env(seed=args.seed)

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

    callback = ShepherdingMetricsCallback(log_freq=2048, verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "ppo_herding"
    model.save(str(model_path))
    print(f"\n✓ Model saved to {model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
