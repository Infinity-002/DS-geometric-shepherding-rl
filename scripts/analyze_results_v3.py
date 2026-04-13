#!/usr/bin/env python3
"""Create publication-style plots and summary tables from v3 results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze v3 experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/research_v3/ablation_runs",
        help="Directory containing episode_summaries.csv and trajectories.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")

    summaries = pd.read_csv(results_dir / "episode_summaries.csv")
    trajectories = pd.read_csv(results_dir / "trajectories.csv")

    agg = (
        summaries.groupby(["run_name", "model_type", "split", "scenario"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_return=("episode_return", "mean"),
            std_return=("episode_return", "std"),
            mean_visible_ratio=("avg_visibility_ratio", "mean"),
            mean_path_length=("dog_path_length", "mean"),
            mean_collisions=("collision_count", "mean"),
            mean_strays=("stray_count", "mean"),
        )
        .sort_values(["split", "scenario", "run_name"])
    )
    agg.to_csv(tables_dir / "summary_table.csv", index=False)

    pivot = agg.pivot_table(
        index=["split", "scenario"],
        columns="run_name",
        values="success_rate",
    )
    pivot.to_csv(tables_dir / "success_rate_pivot.csv")

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=agg,
        x="scenario",
        y="success_rate",
        hue="run_name",
        errorbar=None,
    )
    plt.ylim(0.0, 1.0)
    plt.title("Success Rate Across Seen and Unseen Scenarios")
    plt.tight_layout()
    plt.savefig(figures_dir / "success_rate_by_scenario.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=summaries,
        x="scenario",
        y="episode_return",
        hue="run_name",
    )
    plt.title("Episode Return Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "episode_return_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=summaries,
        x="avg_visibility_ratio",
        y="episode_return",
        hue="run_name",
        style="split",
        s=100,
    )
    plt.title("Visibility vs Episode Return")
    plt.tight_layout()
    plt.savefig(figures_dir / "visibility_vs_return.png", dpi=200)
    plt.close()

    traj_subset = (
        trajectories.sort_values(["run_name", "episode_idx", "step"])
        .groupby(["run_name", "scenario", "episode_idx"], as_index=False)
        .head(1)
    )
    top_runs = agg["run_name"].drop_duplicates().tolist()[:4]
    plt.figure(figsize=(12, 10))
    for run_name in top_runs:
        run_traj = trajectories[trajectories["run_name"] == run_name]
        if run_traj.empty:
            continue
        best_episode = (
            summaries[summaries["run_name"] == run_name]
            .sort_values("episode_return", ascending=False)
            .iloc[0]
        )
        episode = run_traj[
            (run_traj["episode_idx"] == best_episode["episode_idx"])
            & (run_traj["scenario"] == best_episode["scenario"])
        ]
        plt.plot(episode["dog_x"], episode["dog_y"], label=run_name)
    plt.title("Dog Trajectory on Best Episode per Variant")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "trajectory_overlay.png", dpi=200)
    plt.close()

    comparison = (
        summaries.groupby(["run_name", "split"], as_index=False)
        .agg(
            success_rate=("success", "mean"),
            mean_return=("episode_return", "mean"),
            mean_visible_ratio=("avg_visibility_ratio", "mean"),
            mean_collisions=("collision_count", "mean"),
        )
    )
    comparison.to_csv(tables_dir / "train_vs_unseen_table.csv", index=False)

    reward_cols = [
        "avg_reward_base",
        "avg_reward_progress",
        "avg_reward_worst_sheep",
        "avg_reward_visibility_loss",
        "avg_reward_visibility_gain",
        "avg_reward_zero_visibility",
        "avg_reward_stray",
        "avg_reward_drive",
        "avg_reward_collision",
        "avg_reward_success_bonus",
    ]
    available_reward_cols = [col for col in reward_cols if col in summaries.columns]
    if available_reward_cols:
        reward_table = (
            summaries.groupby(["run_name", "split", "scenario"], as_index=False)[
                available_reward_cols
            ]
            .mean()
            .sort_values(["split", "scenario", "run_name"])
        )
        reward_table.to_csv(tables_dir / "reward_decomposition_table.csv", index=False)
    print(f"Saved figures to {figures_dir}")
    print(f"Saved tables to {tables_dir}")


if __name__ == "__main__":
    main()
