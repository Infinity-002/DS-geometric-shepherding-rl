#!/usr/bin/env python3
"""
evaluate.py – Load a trained PPO model and visualise a herding episode.

Usage
-----
    python evaluate.py                              # default model path
    python evaluate.py --model models/ppo_herding   # custom model
    python evaluate.py --save animation.gif         # save to file

Renders the dog, sheep, goal, and the **Convex Hull** of the flock at
every step using matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.spatial import ConvexHull
from stable_baselines3 import PPO

import gymnasium as gym
import envs  # noqa: F401 – triggers HerdingEnv registration


# ======================================================================
# Simulation
# ======================================================================

def run_episode(
    model: PPO,
    env: gym.Env,
    max_steps: int = 500,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Run one episode and collect per-step positions.

    Returns
    -------
    dog_history : list[np.ndarray]
        Dog position at each step, shape ``(2,)``.
    sheep_history : list[np.ndarray]
        Sheep positions at each step, shape ``(N, 2)``.
    goal : np.ndarray
        The environment goal position.
    """
    obs, _ = env.reset()
    dog_history: List[np.ndarray] = []
    sheep_history: List[np.ndarray] = []

    inner_env = env.unwrapped
    goal: np.ndarray = inner_env.goal

    for _ in range(max_steps):
        dog_history.append(inner_env.dog_pos.copy())
        sheep_history.append(inner_env.sheep_pos.copy())

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            # Capture final frame
            dog_history.append(inner_env.dog_pos.copy())
            sheep_history.append(inner_env.sheep_pos.copy())
            break

    return dog_history, sheep_history, goal


# ======================================================================
# Visualisation
# ======================================================================

def animate_episode(
    dog_history: List[np.ndarray],
    sheep_history: List[np.ndarray],
    goal: np.ndarray,
    grid_size: float = 20.0,
    save_path: Optional[str] = None,
    fps: int = 15,
) -> None:
    """Create an animated matplotlib plot of the herding episode.

    The Convex Hull of the flock is highlighted as a translucent polygon
    at every frame.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f1a")

    # Static: goal marker
    ax.plot(
        goal[0], goal[1],
        marker="*", color="#ffd700", markersize=20,
        zorder=5, label="Goal",
    )

    # Dynamic artists
    (sheep_scatter,) = ax.plot(
        [], [], "o",
        color="#7ec8e3", markersize=7, markeredgecolor="white",
        markeredgewidth=0.5, label="Sheep", zorder=4,
    )
    (dog_scatter,) = ax.plot(
        [], [], "s",
        color="#ff6b6b", markersize=12, markeredgecolor="white",
        markeredgewidth=1.0, label="Dog", zorder=5,
    )
    hull_patch = plt.Polygon(
        [[0, 0]], closed=True,
        fill=True, facecolor="#7ec8e3", alpha=0.15,
        edgecolor="#7ec8e3", linewidth=1.5, linestyle="--",
        zorder=3, label="Convex Hull",
    )
    ax.add_patch(hull_patch)

    title = ax.set_title("", color="white", fontsize=14, fontweight="bold")
    ax.legend(
        loc="upper left", fontsize=9,
        facecolor="#16213e", edgecolor="white", labelcolor="white",
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    # Grid lines
    ax.grid(True, color="white", alpha=0.08)

    def _init():
        sheep_scatter.set_data([], [])
        dog_scatter.set_data([], [])
        hull_patch.set_xy([[0, 0]])
        return sheep_scatter, dog_scatter, hull_patch, title

    def _update(frame: int):
        sp = sheep_history[frame]
        dp = dog_history[frame]

        sheep_scatter.set_data(sp[:, 0], sp[:, 1])
        dog_scatter.set_data([dp[0]], [dp[1]])

        # Convex hull
        if sp.shape[0] >= 3:
            try:
                hull = ConvexHull(sp)
                verts = sp[hull.vertices]
                hull_patch.set_xy(verts)
            except Exception:
                hull_patch.set_xy([[0, 0]])
        else:
            hull_patch.set_xy([[0, 0]])

        title.set_text(
            f"Shepherding Simulation  •  Step {frame}/{len(dog_history) - 1}"
        )
        return sheep_scatter, dog_scatter, hull_patch, title

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(dog_history),
        interval=1000 // fps,
        blit=True,
    )

    if save_path:
        suffix = Path(save_path).suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps, dpi=100)
        print(f"✓ Animation saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained herding agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_herding",
        help="Path to trained model (without .zip extension)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save animation to file (e.g. animation.gif or animation.mp4)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the evaluation episode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Shepherding RL – Evaluation & Visualisation")
    print(f"  Model     : {args.model}")
    print(f"  Max steps : {args.max_steps}")
    print("=" * 60)

    env = gym.make("HerdingEnv-v0")
    model = PPO.load(args.model, env=env)

    dog_hist, sheep_hist, goal = run_episode(model, env, max_steps=args.max_steps)
    print(f"Episode finished in {len(dog_hist) - 1} steps.")

    animate_episode(
        dog_hist, sheep_hist, goal,
        grid_size=env.unwrapped.grid_size,
        save_path=args.save,
    )

    env.close()


if __name__ == "__main__":
    main()
