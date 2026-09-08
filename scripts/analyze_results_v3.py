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

from shepherding.research.reporting import (  # noqa: E402
    available_aggs,
    generalization_gap_table,
    holdout_splits,
    warn_if_success_is_degenerate,
)


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

    warn_if_success_is_degenerate(summaries)

    agg_spec = {
        "success_rate": ("success", "mean"),
        "fraction_at_goal": ("fraction_at_goal", "mean"),
        "best_fraction_at_goal": ("best_fraction_at_goal", "mean"),
        "mean_return": ("episode_return", "mean"),
        "std_return": ("episode_return", "std"),
        "mean_visible_ratio": ("avg_visibility_ratio", "mean"),
        "mean_dist_to_goal": ("mean_dist_to_goal", "mean"),
        "mean_max_dist_to_goal": ("max_dist_to_goal", "mean"),
        "mean_path_length": ("dog_path_length", "mean"),
        "mean_collisions": ("collision_count", "mean"),
        "mean_strays": ("stray_count", "mean"),
    }
    agg = (
        summaries.groupby(["run_name", "model_type", "split", "scenario"], as_index=False)
        .agg(**available_aggs(summaries, agg_spec))
        .sort_values(["split", "scenario", "run_name"])
    )
    agg.to_csv(tables_dir / "summary_table.csv", index=False)

    # One pivot and one bar chart per headline metric. Success rate alone is
    # all-or-nothing over the whole flock and is frequently zero everywhere,
    # which makes a success-only figure unreadable.
    for metric, label, limit in (
        ("success_rate", "Success Rate", (0.0, 1.0)),
        ("fraction_at_goal", "Fraction of Flock Delivered", (0.0, 1.0)),
    ):
        if metric not in agg.columns:
            continue
        agg.pivot_table(
            index=["split", "scenario"], columns="run_name", values=metric
        ).to_csv(tables_dir / f"{metric}_pivot.csv")

        plt.figure(figsize=(12, 7))
        sns.barplot(data=agg, x="scenario", y=metric, hue="run_name", errorbar=None)
        plt.ylim(*limit)
        plt.title(f"{label} Across Seen, Unseen and Held-out Scenarios")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_by_scenario.png", dpi=200)
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

    comparison_spec = {
        "success_rate": ("success", "mean"),
        "fraction_at_goal": ("fraction_at_goal", "mean"),
        "mean_episode_return": ("episode_return", "mean"),
        "mean_dist_to_goal": ("mean_dist_to_goal", "mean"),
        "mean_visible_ratio": ("avg_visibility_ratio", "mean"),
        "mean_collisions": ("collision_count", "mean"),
    }
    comparison = summaries.groupby(["run_name", "split"], as_index=False).agg(
        **available_aggs(summaries, comparison_spec)
    )
    comparison.to_csv(tables_dir / "split_comparison_table.csv", index=False)

    # One gap table per held-out split, so the procedural test suite is reported
    # rather than silently folded into "unseen".
    for split_name in holdout_splits(comparison):
        gaps = generalization_gap_table(
            comparison, holdout_split=split_name, group_key="run_name"
        )
        if gaps.empty:
            continue
        gaps.to_csv(tables_dir / f"generalization_gap_{split_name}.csv", index=False)

        metric = (
            "fraction_at_goal" if "fraction_at_goal_gap" in gaps.columns else "success_rate"
        )
        plot_df = gaps.melt(
            id_vars="run_name",
            value_vars=[f"train_{metric}", f"holdout_{metric}"],
            var_name="split",
            value_name=metric,
        )
        plot_df["split"] = plot_df["split"].str.replace(
            "holdout", split_name, regex=False
        )
        plt.figure(figsize=(12, 7))
        sns.barplot(data=plot_df, x="run_name", y=metric, hue="split", errorbar=None)
        plt.ylim(0.0, 1.0)
        plt.title(f"Train vs {split_name.title()} Generalization Gap")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / f"generalization_gap_{split_name}.png", dpi=200)
        plt.close()

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
