#!/usr/bin/env python3
"""
evaluate_v2.py – Load a trained v2 agent and visualise a herding episode.

Renders all v2-specific elements:
  * Dog **visibility circle** (dashed white ring)
  * **Obstacles** as filled grey rectangles
  * Visible sheep in bright blue; invisible sheep dimmed/greyed
  * Convex Hull of the *visible* flock only

Usage
-----
    python evaluate_v2.py                              # default model
    python evaluate_v2.py --model models/ppo_herding_v2
    python evaluate_v2.py --save animation_v2.gif
    python evaluate_v2.py --random-agent              # random action baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow, Rectangle
from scipy.spatial import ConvexHull
from stable_baselines3 import PPO

import gymnasium as gym
import shepherding.envs  # noqa: F401

from shepherding.utils.geometry_v2 import visible_sheep_mask


# ======================================================================
# Simulation
# ======================================================================

def run_episode(
    model: Optional[PPO],
    env: gym.Env,
    max_steps: int = 600,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, List[List]]:
    """Run one episode and collect per-step state.

    Parameters
    ----------
    model : PPO or None
        Trained model.  If ``None``, random actions are used.
    env : gym.Env
        The herding environment (HerdingEnv-v2).

    Returns
    -------
    dog_history     : list of (2,) arrays
    sheep_history   : list of (N, 2) arrays
    vis_mask_hist   : list of (N,) bool arrays – which sheep are visible
    goal            : (2,) array
    obstacles       : list of (x, y, w, h) tuples
    """
    obs, _ = env.reset()
    inner = env.unwrapped
    goal: np.ndarray = inner.goal
    obstacles = inner.obstacles
    visibility_radius: float = inner.visibility_radius

    dog_history: List[np.ndarray] = []
    sheep_history: List[np.ndarray] = []
    vis_history: List[np.ndarray] = []

    for _ in range(max_steps):
        dog_history.append(inner.dog_pos.copy())
        sheep_history.append(inner.sheep_pos.copy())
        vis_history.append(
            visible_sheep_mask(inner.dog_pos, inner.sheep_pos, visibility_radius)
        )

        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            dog_history.append(inner.dog_pos.copy())
            sheep_history.append(inner.sheep_pos.copy())
            vis_history.append(
                visible_sheep_mask(inner.dog_pos, inner.sheep_pos, visibility_radius)
            )
            break

    return dog_history, sheep_history, vis_history, goal, obstacles


# ======================================================================
# Visualisation
# ======================================================================

def animate_episode(
    dog_history: List[np.ndarray],
    sheep_history: List[np.ndarray],
    vis_history: List[np.ndarray],
    goal: np.ndarray,
    obstacles: List,
    grid_size: float = 20.0,
    visibility_radius: float = 8.0,
    save_path: Optional[str] = None,
    fps: int = 15,
) -> None:
    """Create an animated matplotlib visualisation of the v2 episode."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#12121f")
    fig.patch.set_facecolor("#0a0a14")

    # ── Static elements ────────────────────────────────────────────────

    # Goal star
    ax.plot(
        goal[0], goal[1],
        marker="*", color="#ffd700", markersize=22,
        zorder=6, label="Goal",
    )
    goal_circle = plt.Circle(
        (goal[0], goal[1]), 2.0,
        fill=False, edgecolor="#ffd700", linewidth=1.0,
        linestyle=":", alpha=0.5, zorder=5,
    )
    ax.add_patch(goal_circle)

    # Obstacles (rendered once, static)
    for (rx, ry, rw, rh) in obstacles:
        rect_patch = Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor="#ff8c42", facecolor="#3a2a1a",
            zorder=4, label="Obstacle",
        )
        ax.add_patch(rect_patch)
    # De-duplicate legend entries for obstacles
    if obstacles:
        legend_obstacle = mpatches.Patch(
            facecolor="#3a2a1a", edgecolor="#ff8c42", label="Obstacle"
        )

    # Grid
    ax.grid(True, color="white", alpha=0.06)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444466")

    # ── Dynamic artists ────────────────────────────────────────────────

    # Visible sheep (bright blue)
    (sheep_vis_plot,) = ax.plot(
        [], [], "o",
        color="#7ec8e3", markersize=7, markeredgecolor="white",
        markeredgewidth=0.5, label="Visible sheep", zorder=5,
    )
    # Invisible sheep (dimmed grey)
    (sheep_invis_plot,) = ax.plot(
        [], [], "o",
        color="#444466", markersize=6, markeredgecolor="#666699",
        markeredgewidth=0.3, label="Hidden sheep", zorder=4, alpha=0.6,
    )
    # Dog
    (dog_plot,) = ax.plot(
        [], [], "s",
        color="#ff6b6b", markersize=13, markeredgecolor="white",
        markeredgewidth=1.2, label="Dog", zorder=7,
    )
    # Visibility circle (dashed ring around dog)
    vis_circle = Circle(
        (0, 0), visibility_radius,
        fill=False, edgecolor="white", linewidth=1.0,
        linestyle="--", alpha=0.3, zorder=6,
    )
    ax.add_patch(vis_circle)

    # Convex hull of visible flock
    hull_patch = plt.Polygon(
        [[0, 0]], closed=True,
        fill=True, facecolor="#7ec8e3", alpha=0.10,
        edgecolor="#7ec8e3", linewidth=1.2, linestyle="--",
        zorder=3,
    )
    ax.add_patch(hull_patch)

    title = ax.set_title("", color="white", fontsize=13, fontweight="bold")

    # Legend
    legend_items = [
        mpatches.Patch(color="#ffd700", label="Goal"),
        mpatches.Patch(color="#ff6b6b", label="Dog"),
        mpatches.Patch(color="#7ec8e3", label="Visible sheep"),
        mpatches.Patch(color="#444466", label="Hidden sheep"),
    ]
    if obstacles:
        legend_items.append(legend_obstacle)
    ax.legend(
        handles=legend_items,
        loc="upper left", fontsize=9,
        facecolor="#1a1a2e", edgecolor="#666699", labelcolor="white",
    )

    # ── Animation functions ────────────────────────────────────────────

    def _init():
        sheep_vis_plot.set_data([], [])
        sheep_invis_plot.set_data([], [])
        dog_plot.set_data([], [])
        vis_circle.center = (0, 0)
        hull_patch.set_xy([[0, 0]])
        return sheep_vis_plot, sheep_invis_plot, dog_plot, vis_circle, hull_patch, title

    def _update(frame: int):
        sp = sheep_history[frame]
        dp = dog_history[frame]
        vm = vis_history[frame]

        vis_sheep = sp[vm]
        invis_sheep = sp[~vm]

        if vis_sheep.shape[0] > 0:
            sheep_vis_plot.set_data(vis_sheep[:, 0], vis_sheep[:, 1])
        else:
            sheep_vis_plot.set_data([], [])

        if invis_sheep.shape[0] > 0:
            sheep_invis_plot.set_data(invis_sheep[:, 0], invis_sheep[:, 1])
        else:
            sheep_invis_plot.set_data([], [])

        dog_plot.set_data([dp[0]], [dp[1]])
        vis_circle.center = (dp[0], dp[1])

        # Convex hull of *visible* sheep only
        if vis_sheep.shape[0] >= 3:
            try:
                hull = ConvexHull(vis_sheep)
                hull_patch.set_xy(vis_sheep[hull.vertices])
            except Exception:
                hull_patch.set_xy([[0, 0]])
        else:
            hull_patch.set_xy([[0, 0]])

        n_vis = int(np.sum(vm))
        n_total = sp.shape[0]
        title.set_text(
            f"Shepherding v2  •  Step {frame}/{len(dog_history) - 1}"
            f"  •  Visible: {n_vis}/{n_total} sheep"
        )
        return sheep_vis_plot, sheep_invis_plot, dog_plot, vis_circle, hull_patch, title

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
        anim.save(save_path, writer=writer, fps=fps, dpi=110)
        print(f"✓ Animation saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HerdingEnv-v2 agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_herding_v2",
        help="Path to trained model (without .zip). Ignored if --random-agent.",
    )
    parser.add_argument("--save", type=str, default=None, help="Save path (e.g. animation_v2.gif)")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-agent",
        action="store_true",
        help="Use random actions (no model required – great for sanity checking)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Shepherding RL v2 – Evaluation & Visualisation")
    print(f"  Model      : {'[random agent]' if args.random_agent else args.model}")
    print(f"  Max steps  : {args.max_steps}")
    print("=" * 60)

    env = gym.make("HerdingEnv-v2")

    model: Optional[PPO] = None
    if not args.random_agent:
        model = PPO.load(args.model, env=env)

    dog_hist, sheep_hist, vis_hist, goal, obstacles = run_episode(
        model, env, max_steps=args.max_steps
    )
    print(f"Episode finished in {len(dog_hist) - 1} steps.")

    animate_episode(
        dog_hist,
        sheep_hist,
        vis_hist,
        goal,
        obstacles,
        grid_size=env.unwrapped.grid_size,
        visibility_radius=env.unwrapped.visibility_radius,
        save_path=args.save,
    )

    env.close()


if __name__ == "__main__":
    main()
