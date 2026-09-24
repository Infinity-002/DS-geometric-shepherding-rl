"""Regenerate the figures used in docs/paper/main.typ.

    uv run python scripts/paper_figures.py

Writes three PNGs to docs/paper/figures/:
  v3_layouts.png   sampled procedural layouts, one per obstacle topology
  v3_training.png  validation delivery and curriculum stage over training
  v3_results.png   held-out FracGoal with 95% bootstrap CIs per agent
  v3_rollouts.png  heuristic vs recurrent PPO paths on the same preset episodes
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


def _stage_shares(stages: list[tuple[int, float]], end: int) -> dict[float, float]:
    shares: dict[float, float] = {0.0: 0.0, 0.33: 0.0, 0.66: 0.0, 1.0: 0.0}
    bounds = [s for s, _ in stages] + [end]
    for (start, stage), stop in zip(stages, bounds[1:]):
        shares[stage] = shares.get(stage, 0.0) + (stop - start)
    return {k: v / end for k, v in shares.items()}


def plot_training(groups: list[tuple[str, list[Path], str]]) -> None:
    fig, (ax, ax_st) = plt.subplots(
        2, 1, figsize=(3.4, 2.35), gridspec_kw={"height_ratios": [1.5, 1]}
    )
    bars = []
    for label, logs, color in groups:
        curves = []
        for i, log in enumerate(logs):
            val, stages = parse_log(log)
            curves.append(val.set_index("step").frac)
            end = int(val.step.max())
            bars.append((f"{label} s{i}", _stage_shares(stages, end)))
        df = pd.concat(curves, axis=1).dropna()
        steps = df.index.to_numpy() / 1e3
        ax.plot(steps, df.mean(axis=1), color=color, lw=1.1, label=f"{label} (mean of {len(logs)})")
        ax.fill_between(steps, df.min(axis=1), df.max(axis=1), color=color, alpha=0.18, lw=0)
    ax.set_ylabel("Validation FracGoal")
    ax.set_xlabel("Environment steps (thousands)", labelpad=1)
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3, lw=0.4)

    colors = {0.0: "#d5d8dc", 0.33: "#85c1e9", 0.66: "#1f618d", 1.0: "#000000"}
    y = np.arange(len(bars))[::-1]
    for yi, (name, shares) in zip(y, bars):
        left = 0.0
        for stage in (0.0, 0.33, 0.66):
            ax_st.barh(yi, shares[stage], left=left, color=colors[stage], height=0.7,
                       label=f"stage {stage:g}" if yi == y[0] else None)
            left += shares[stage]
    ax_st.set_yticks(y)
    ax_st.set_yticklabels([n for n, _ in bars])
    ax_st.set_xlim(0, 1)
    ax_st.set_xlabel("Share of training steps", labelpad=1)
    ax_st.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(OUT / "v3_training.png", dpi=300)
    plt.close(fig)


def plot_results(rl_groups: list[tuple[str, list[str], str]]) -> None:
    agents = [("Heuristic", ["heuristic"], "#7f8c8d"), ("BC", ["behavioral_cloning"], "#e67e22")]
    agents += rl_groups
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    width = 0.8 / len(agents)
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
        pos = x + (k - (len(agents) - 1) / 2) * width
        ax.bar(pos, mean, width, color=color, label=label)
        ax.errorbar(pos, mean, yerr=[mean - lo, hi - mean], fmt="none", ecolor="black", elinewidth=0.5, capsize=1)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in SCENARIOS])
    ax.set_ylabel("Fraction of flock at goal")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(ncol=4, frameon=False, loc="upper center", columnspacing=0.8, handlelength=1.2)
    ax.grid(axis="y", alpha=0.3, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "v3_results.png", dpi=300)
    plt.close(fig)


def _pick_episode(scenario: str, rl_dir: str) -> int:
    """First episode where RL and the heuristic end differently, else episode 0."""
    rl = pd.read_csv(RESULTS / rl_dir / scenario / "episode_summaries.csv")
    heu = pd.read_csv(RESULTS / "heuristic" / "episode_summaries.csv")
    heu = heu[heu.scenario == scenario].set_index("episode_idx")
    rl = rl.set_index("episode_idx")
    diff = (rl.success != heu.success.reindex(rl.index)).to_numpy().nonzero()[0]
    return int(rl.index[diff[0]]) if len(diff) else 0


def _load_path(csv: Path, scenario: str, episode: int) -> pd.DataFrame:
    cols = ["scenario", "episode_idx", "step", "dog_x", "dog_y", "centroid_x", "centroid_y", "fraction_at_goal"]
    chunks = pd.read_csv(csv, usecols=cols, chunksize=200_000)
    parts = [c[(c.scenario == scenario) & (c.episode_idx == episode)] for c in chunks]
    return pd.concat(parts).sort_values("step")


def plot_rollouts(config_path: Path, rl_dir: str) -> None:
    env_cfg = dict(load_yaml_config(config_path)["environment"])
    panels = []
    for scenario, title in [("unseen_split_field", "Split field"), ("unseen_open_field", "Open field")]:
        env_cfg["scenario"] = scenario
        env = gym.make("HerdingEnv-v3", **env_cfg).unwrapped
        env.reset(seed=100_000)
        obstacles, goal, r_goal = list(env.obstacles), env.goal.copy(), float(env.success_radius)
        env.close()
        episode = _pick_episode(scenario, rl_dir)
        for agent, csv in [
            ("Heuristic", RESULTS / "heuristic" / "trajectories.csv"),
            ("Recurrent PPO", RESULTS / rl_dir / scenario / "trajectories.csv"),
        ]:
            panels.append((f"{title}: {agent}", obstacles, goal, r_goal, _load_path(csv, scenario, episode)))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 2.2))
    for ax, (title, obstacles, goal, r_goal, path) in zip(axes, panels):
        for x, y, w, h in obstacles:
            ax.add_patch(Rectangle((x, y), w, h, color="#8c6d46"))
        ax.add_patch(Circle(goal, r_goal, color="#f2c14e", alpha=0.6))
        ax.plot(path.dog_x, path.dog_y, color="#c0392b", lw=0.5, alpha=0.8, label="dog")
        ax.plot(path.centroid_x, path.centroid_y, color="#2c6fbb", lw=1.3, label="flock centroid")
        ax.scatter([path.centroid_x.iloc[0]], [path.centroid_y.iloc[0]], s=10, color="#2c6fbb", zorder=4)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        end = path.iloc[-1]
        ax.set_title(f"{title}\n{int(end.step) + 1} steps, {end.fraction_at_goal:.0%} delivered", pad=2)
    axes[0].legend(loc="lower right", frameon=False, fontsize=5)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(top=0.82)
    fig.savefig(OUT / "v3_rollouts.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research/v3.yaml")
    parser.add_argument("--rollout-dir", default="rppo_run5_fixed")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base_logs = [ROOT / f"logs/runs/{n}" for n in ("run5_v3_seed0.log", "run6_v3_seed1.log", "run7_v3_seed2.log")]
    gate_logs = [ROOT / f"logs/runs/run{8 + s}_gate045_seed{s}.log" for s in range(3)]
    base_dirs = ["rppo_run5_fixed", "rppo_seed1_fixed", "rppo_seed2_fixed"]
    gate_dirs = [f"gate045_seed{s}_fixed" for s in range(3)]
    plot_layouts(ROOT / args.config)
    plot_training([("Base", base_logs, "#2c6fbb"), ("Gate 0.45", gate_logs, "#27ae60")])
    plot_results([("RL base", base_dirs, "#2c6fbb"), ("RL gate", gate_dirs, "#27ae60")])
    plot_rollouts(ROOT / args.config, args.rollout_dir)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
