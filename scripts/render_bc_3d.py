#!/usr/bin/env python3
"""Render a Behavioral Cloning agent episode as a 3D animation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import shepherding.envs  # noqa: F401
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from shepherding.utils.geometry_v2 import visible_sheep_mask
from shepherding.research import load_yaml_config
from shepherding.imitation.model import load_behavioral_cloning_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a BC episode in 3D")
    parser.add_argument("--config", type=str, default="configs/research/v3_structured.yaml")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--scenario", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def run_episode(
    model: Any,
    env: gym.Env,
    max_steps: int,
    seed: int,
    scenario: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, List[Tuple[float, float, float, float]], float]:
    obs, _ = env.reset(seed=seed, options={"scenario": scenario})
    inner = env.unwrapped

    dog_history: List[np.ndarray] = []
    sheep_history: List[np.ndarray] = []
    vis_history: List[np.ndarray] = []

    for _ in range(max_steps):
        dog_history.append(inner.dog_pos.copy())
        sheep_history.append(inner.sheep_pos.copy())
        vis_history.append(
            visible_sheep_mask(inner.dog_pos, inner.sheep_pos, inner.visibility_radius)
        )

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            dog_history.append(inner.dog_pos.copy())
            sheep_history.append(inner.sheep_pos.copy())
            vis_history.append(
                visible_sheep_mask(inner.dog_pos, inner.sheep_pos, inner.visibility_radius)
            )
            break

    return (
        dog_history,
        sheep_history,
        vis_history,
        inner.goal.copy(),
        list(inner.obstacles),
        float(inner.grid_size),
    )


def _cuboid_faces(x: float, y: float, z: float, dx: float, dy: float, dz: float) -> List[List[Tuple[float, float, float]]]:
    x2, y2, z2 = x + dx, y + dy, z + dz
    return [
        [(x, y, z), (x2, y, z), (x2, y2, z), (x, y2, z)],
        [(x, y, z2), (x2, y, z2), (x2, y2, z2), (x, y2, z2)],
        [(x, y, z), (x2, y, z), (x2, y, z2), (x, y, z2)],
        [(x2, y, z), (x2, y2, z), (x2, y2, z2), (x2, y, z2)],
        [(x, y2, z), (x2, y2, z), (x2, y2, z2), (x, y2, z2)],
        [(x, y, z), (x, y2, z), (x, y2, z2), (x, y, z2)],
    ]


def animate_episode(
    dog_history: List[np.ndarray],
    sheep_history: List[np.ndarray],
    vis_history: List[np.ndarray],
    goal: np.ndarray,
    obstacles: List[Tuple[float, float, float, float]],
    grid_size: float,
    save_path: Optional[str] = None,
    fps: int = 15,
) -> None:
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#101827")
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, 3.0)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Height", color="white")
    ax.tick_params(colors="white")
    ax.xaxis.pane.set_facecolor((0.08, 0.10, 0.16, 1.0))
    ax.yaxis.pane.set_facecolor((0.08, 0.10, 0.16, 1.0))
    ax.zaxis.pane.set_facecolor((0.08, 0.10, 0.16, 1.0))
    ax.view_init(elev=32, azim=-58)

    gx = np.linspace(0, grid_size, 2)
    gy = np.linspace(0, grid_size, 2)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = np.zeros_like(XX)
    ax.plot_surface(XX, YY, ZZ, color="#1f6f5f", alpha=0.25, shade=False)

    for rx, ry, rw, rh in obstacles:
        faces = _cuboid_faces(rx, ry, 0.0, rw, rh, 1.2)
        poly = Poly3DCollection(
            faces,
            facecolors="#8b5a2b",
            edgecolors="#ffb066",
            linewidths=0.8,
            alpha=0.75,
        )
        ax.add_collection3d(poly)

    ax.scatter(
        [goal[0]],
        [goal[1]],
        [0.2],
        color="#ffd700",
        marker="*",
        s=260,
        label="Goal",
        depthshade=False,
    )

    sheep_vis = ax.scatter([], [], [], color="#8bd3ff", s=50, depthshade=True, label="Visible sheep")
    sheep_hidden = ax.scatter([], [], [], color="#6b7280", s=35, depthshade=True, label="Hidden sheep")
    dog_plot = ax.scatter([], [], [], color="#ff6b6b", s=120, marker="s", depthshade=True, label="Dog")
    dog_trail, = ax.plot([], [], [], color="#ff9b9b", linewidth=2.0, alpha=0.8)
    title = ax.set_title("", color="white", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")

    def _set_scatter(scatter: Any, points_xy: np.ndarray, z_value: float) -> None:
        if points_xy.size == 0:
            scatter._offsets3d = ([], [], [])
            return
        scatter._offsets3d = (
            points_xy[:, 0],
            points_xy[:, 1],
            np.full(points_xy.shape[0], z_value),
        )

    def _update(frame: int):
        sheep = sheep_history[frame]
        dog = dog_history[frame]
        visible = vis_history[frame]

        _set_scatter(sheep_vis, sheep[visible], 0.18)
        _set_scatter(sheep_hidden, sheep[~visible], 0.12)
        _set_scatter(dog_plot, dog.reshape(1, 2), 0.35)

        trail = np.asarray(dog_history[: frame + 1])
        dog_trail.set_data(trail[:, 0], trail[:, 1])
        dog_trail.set_3d_properties(np.full(trail.shape[0], 0.35))

        title.set_text(
            f"Shepherding BC 3D Render  |  Step {frame}/{len(dog_history) - 1}"
        )
        return sheep_vis, sheep_hidden, dog_plot, dog_trail, title

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(dog_history),
        interval=1000 // fps,
        blit=False,
    )

    if save_path:
        suffix = Path(save_path).suffix.lower()
        writer = "pillow" if suffix == ".gif" else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps, dpi=120)
        print(f"Saved 3D animation to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config))
    env_cfg = dict(config["environment"])
    env_cfg["compute_expensive_metrics"] = True
    max_steps = args.max_steps

    env = gym.make("HerdingEnv-v3", **env_cfg)
    model = load_behavioral_cloning_agent(args.model_path)
    
    dog_history, sheep_history, vis_history, goal, obstacles, grid_size = run_episode(
        model=model,
        env=env,
        max_steps=max_steps,
        seed=args.seed,
        scenario=args.scenario,
    )
    animate_episode(
        dog_history=dog_history,
        sheep_history=sheep_history,
        vis_history=vis_history,
        goal=goal,
        obstacles=obstacles,
        grid_size=grid_size,
        save_path=args.save,
        fps=args.fps,
    )
    env.close()


if __name__ == "__main__":
    main()
