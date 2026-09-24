"""Regenerate the figures used in docs/paper/main.typ.

    uv run python scripts/paper_figures.py

Writes three PNGs to docs/paper/figures/:
  v3_layouts.png   sampled procedural layouts, one per obstacle topology
  v3_training.png  validation delivery and curriculum stage over training
  v3_results.png   held-out FracGoal with 95% bootstrap CIs per agent
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gymnasium as gym  # noqa: E402

import shepherding.envs  # noqa: E402,F401
from shepherding.research import load_yaml_config  # noqa: E402

OUT = ROOT / "docs/paper/figures"
RESULTS = ROOT / "results/generalization_v3"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.5,
    }
)

SCENARIOS = [
    ("test_procedural", "Procedural\ntest"),
    ("unseen_split_field", "Split\nfield"),
    ("unseen_dense", "Dense"),
    ("unseen_open_field", "Open\nfield"),
    ("unseen_corridor", "Corridor"),
    ("unseen_narrow_gate", "Narrow\ngate"),
]


def plot_layouts(config_path: Path) -> None:
    env_cfg = dict(load_yaml_config(config_path)["environment"])
    env_cfg["scenario"] = "test_procedural"
    env = gym.make("HerdingEnv-v3", **env_cfg).unwrapped

    wanted = ["blobs", "corridor", "gate", "bars"]
    found: dict[str, tuple] = {}
    seed = 100_000
    while len(found) < len(wanted) and seed < 100_400:
        env.reset(seed=seed)
        if env.topology in wanted and env.topology not in found:
            found[env.topology] = (
                list(env.obstacles),
                env.sheep_pos.copy(),
                env.dog_pos.copy(),
                env.goal.copy(),
                float(env.visibility_radius),
                float(env.success_radius),
            )
        seed += 1
    env.close()

    fig, axes = plt.subplots(1, len(found), figsize=(7.0, 2.05))
    for ax, name in zip(axes, wanted):
        obstacles, sheep, dog, goal, r_vis, r_goal = found[name]
        g = float(env_cfg["grid_size"])
        for x, y, w, h in obstacles:
            ax.add_patch(Rectangle((x, y), w, h, color="#8c6d46"))
        ax.add_patch(Circle(goal, r_goal, color="#f2c14e", alpha=0.6))
        ax.add_patch(Circle(dog, r_vis, fill=False, ls="--", lw=0.6, color="#c0392b"))
        ax.scatter(sheep[:, 0], sheep[:, 1], s=5, color="#2c6fbb", zorder=3)
        ax.scatter([dog[0]], [dog[1]], s=14, marker="s", color="#c0392b", zorder=4)
        ax.set_xlim(0, g)
        ax.set_ylim(0, g)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}, $N$={len(sheep)}", pad=2)
    fig.tight_layout(pad=0.6)
    fig.savefig(OUT / "v3_layouts.png", dpi=300)
    plt.close(fig)


def parse_log(path: Path) -> tuple[pd.DataFrame, list[tuple[int, float]]]:
    text = path.read_text(errors="ignore")
    val = [
        (int(s.replace(",", "")), float(sr), float(fg))
        for s, sr, fg in re.findall(
            r"\[validation @ ([\d,]+)\] SR=([\d.]+) \| FracGoal=([\d.]+)", text
        )
    ]
    stages = [(0, 0.0)] + [
        (int(s.replace(",", "")), float(st))
        for st, s in re.findall(
            r"curriculum (?:advanced|fell back) to stage ([\d.]+) at timestep ([\d,]+)",
            text,
        )
    ]
    return pd.DataFrame(val, columns=["step", "sr", "frac"]), stages


def plot_training(logs: list[Path]) -> None:
    fig, (ax, ax_st) = plt.subplots(
        2, 1, figsize=(3.4, 2.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    colors = ["#2c6fbb", "#e67e22", "#27ae60"]
    for i, log in enumerate(logs):
        val, stages = parse_log(log)
        if val.empty:
            continue
        label = f"seed {i}"
        ax.plot(val.step / 1e3, val.frac, color=colors[i], lw=1.0, marker="o", ms=1.8, label=label)
        ax.plot(val.step / 1e3, val.sr, color=colors[i], lw=0.7, ls=":")
        end = val.step.max()
        xs = [s for s, _ in stages] + [end]
        ys = [v for _, v in stages] + [stages[-1][1]]
        ax_st.step(np.array(xs) / 1e3, ys, where="post", color=colors[i], lw=0.9)
    ax.set_ylabel("Validation")
    ax.set_ylim(0, 0.7)
    ax.plot([], [], color="grey", lw=1.0, label="FracGoal")
    ax.plot([], [], color="grey", lw=0.7, ls=":", label="Success")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.grid(alpha=0.3, lw=0.4)
    ax_st.set_ylabel("Stage")
    ax_st.set_yticks([0, 0.33, 0.66, 1.0])
    ax_st.set_ylim(-0.05, 1.05)
    ax_st.set_xlabel("Environment steps (thousands)")
    ax_st.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "v3_training.png", dpi=300)
    plt.close(fig)


def plot_results(rl_dirs: list[str]) -> None:
    agents = [("Heuristic", ["heuristic"], "#7f8c8d"), ("BC", ["behavioral_cloning"], "#e67e22")]
    agents.append(("Recurrent PPO", rl_dirs, "#2c6fbb"))
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    width = 0.26
    x = np.arange(len(SCENARIOS))
    for k, (label, dirs, color) in enumerate(agents):
        frames = [
            pd.read_csv(RESULTS / d / "generalization_report.csv").set_index("scenario")
            for d in dirs
        ]
        mean = np.array([np.mean([f.loc[s, "fraction_at_goal"] for f in frames]) for s, _ in SCENARIOS])
        if len(frames) == 1:
            lo = np.array([frames[0].loc[s, "fraction_at_goal_ci_low"] for s, _ in SCENARIOS])
            hi = np.array([frames[0].loc[s, "fraction_at_goal_ci_high"] for s, _ in SCENARIOS])
        else:
            vals = np.array([[f.loc[s, "fraction_at_goal"] for f in frames] for s, _ in SCENARIOS])
            lo, hi = vals.min(axis=1), vals.max(axis=1)
        pos = x + (k - 1) * width
        ax.bar(pos, mean, width, color=color, label=label)
        ax.errorbar(pos, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="black", elinewidth=0.5, capsize=1)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in SCENARIOS])
    ax.set_ylabel("Fraction of flock at goal")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "v3_results.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/v3.yaml")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["logs/runs/run5_v3_seed0.log", "logs/runs/run6_v3_seed1.log", "logs/runs/run7_v3_seed2.log"],
    )
    parser.add_argument("--rl-dirs", nargs="+", default=["rppo_run5_fixed"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_layouts(ROOT / args.config)
    plot_training([ROOT / p for p in args.logs if (ROOT / p).exists()])
    plot_results(args.rl_dirs)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
