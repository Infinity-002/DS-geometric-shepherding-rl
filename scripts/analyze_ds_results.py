#!/usr/bin/env python3
"""Create clean, report-grade figures for heuristic vs BC vs RL comparisons."""

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


DISPLAY_NAMES = {
    "heuristic_cluster_aware": "Heuristic",
    "heuristic_cluster_aware_fast": "Heuristic",
    "heuristic_cluster_aware_improved": "Heuristic",
    "behavioral_cloning_rf": "Behavioral Cloning",
    "behavioral_cloning_rf_fast": "Behavioral Cloning",
    "behavioral_cloning_rf_improved": "Behavioral Cloning",
    "recurrent_domain_randomized": "RL (Recurrent PPO)",
    "recurrent_domain_randomized_fast": "RL (Recurrent PPO)",
    "recurrent_domain_randomized_improved": "RL (Recurrent PPO)",
    "recurrent_structured_seed0": "RL (Structured PPO)",
    "rl_v2_ppo": "RL (v2 PPO)",
}

DISPLAY_ORDER = [
    "RL (v2 PPO)",
    "Heuristic",
    "Behavioral Cloning",
    "RL (Recurrent PPO)",
    "RL (Structured PPO)",
]

PALETTE = {
    "RL (v2 PPO)": "#6d597a",
    "Heuristic": "#3d5a80",
    "Behavioral Cloning": "#2a9d8f",
    "RL (Recurrent PPO)": "#e76f51",
    "RL (Structured PPO)": "#8d5fd3",
}

SCENARIO_LABELS = {
    "train": "Train",
    "unseen_split_field": "Split Field",
    "unseen_open_field": "Open Field",
    "unseen_corridor": "Corridor",
    "unseen_dense": "Dense",
    "unseen_narrow_gate": "Narrow Gate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DS benchmark results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/research_v3_fast/ds_benchmark",
        help="Directory containing episode_summaries.csv and aggregate_metrics.csv",
    )
    parser.add_argument(
        "--bc-metrics",
        type=str,
        default="models/imitation_fast/random_forest/metrics.json",
        help="Optional behavioral cloning metrics JSON file",
    )
    parser.add_argument(
        "--extra-results-dir",
        type=str,
        default=None,
        help="Optional second results directory to merge into the plots (for example v2 benchmark outputs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures_ds"
    tables_dir = results_dir / "tables_ds"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    summaries = pd.read_csv(results_dir / "episode_summaries.csv")
    aggregates = pd.read_csv(results_dir / "aggregate_metrics.csv")
    if args.extra_results_dir:
        extra_dir = Path(args.extra_results_dir)
        extra_summaries = pd.read_csv(extra_dir / "episode_summaries.csv")
        extra_aggregates = pd.read_csv(extra_dir / "aggregate_metrics.csv")
        summaries = pd.concat([summaries, extra_summaries], ignore_index=True, sort=False)
        aggregates = pd.concat([aggregates, extra_aggregates], ignore_index=True, sort=False)

    summaries["method"] = summaries["run_name"].map(_display_name)
    aggregates["method"] = aggregates["run_name"].map(_display_name)
    summaries = summaries[summaries["method"].isin(DISPLAY_ORDER)].copy()
    aggregates = aggregates[aggregates["method"].isin(DISPLAY_ORDER)].copy()
    summaries["scenario_short"] = summaries["scenario"].map(_scenario_label)
    aggregates["scenario_short"] = aggregates["scenario"].map(_scenario_label)

    overall = _build_overall_table(aggregates)
    gaps = _build_generalization_table(overall)
    scenario_table = _build_scenario_table(aggregates)
    ranking = _build_method_ranking(overall)

    overall.to_csv(tables_dir / "overall_method_summary.csv", index=False)
    gaps.to_csv(tables_dir / "generalization_gap.csv", index=False)
    scenario_table.to_csv(tables_dir / "scenario_metric_matrix.csv", index=False)
    ranking.to_csv(tables_dir / "method_ranking.csv", index=False)

    _plot_main_dashboard(overall, gaps, figures_dir / "main_dashboard.png")
    _plot_scenario_heatmaps(aggregates, figures_dir / "scenario_heatmaps.png")
    _plot_return_profiles(summaries, figures_dir / "return_profiles.png")
    _plot_progress_tradeoff(overall, figures_dir / "progress_tradeoff.png")

    bc_metrics_path = Path(args.bc_metrics)
    if bc_metrics_path.exists():
        bc_metrics = pd.read_json(bc_metrics_path, typ="series")
        bc_metrics.to_frame(name="value").to_csv(tables_dir / "bc_offline_metrics.csv")
        _plot_bc_metrics(bc_metrics, figures_dir / "bc_offline_metrics.png")

    print(f"Saved refined DS figures to {figures_dir}")
    print(f"Saved refined DS tables to {tables_dir}")


def _display_name(run_name: str) -> str:
    return DISPLAY_NAMES.get(run_name, run_name.replace("_", " ").title())


def _scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario.replace("_", " ").title())


def _ordered_methods(values: pd.Series) -> pd.Categorical:
    active = [name for name in DISPLAY_ORDER if name in set(values)]
    return pd.Categorical(values, categories=active, ordered=True)


def _build_overall_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        aggregates.groupby(["method", "split"], as_index=False)
        .agg(
            success_rate=("success_rate", "mean"),
            mean_episode_return=("mean_episode_return", "mean"),
            mean_episode_length=("mean_episode_length", "mean"),
            mean_dist_to_goal=("mean_dist_to_goal", "mean"),
            mean_stray_count=("mean_stray_count", "mean"),
            mean_collision_count=("mean_collision_count", "mean"),
            mean_dog_path_length=("mean_dog_path_length", "mean"),
        )
    )
    grouped["method"] = _ordered_methods(grouped["method"])
    return grouped.sort_values(["split", "method"]).reset_index(drop=True)


def _build_generalization_table(overall: pd.DataFrame) -> pd.DataFrame:
    train = overall[overall["split"] == "train"].rename(
        columns={
            "success_rate": "train_success_rate",
            "mean_episode_return": "train_return",
            "mean_dist_to_goal": "train_mean_dist_to_goal",
        }
    )
    unseen = overall[overall["split"] == "unseen"].rename(
        columns={
            "success_rate": "unseen_success_rate",
            "mean_episode_return": "unseen_return",
            "mean_dist_to_goal": "unseen_mean_dist_to_goal",
        }
    )
    merged = train.merge(unseen, on="method", how="inner")
    merged["success_gap"] = merged["train_success_rate"] - merged["unseen_success_rate"]
    merged["return_gap"] = merged["train_return"] - merged["unseen_return"]
    merged["method"] = _ordered_methods(merged["method"])
    return merged.sort_values("method").reset_index(drop=True)


def _build_scenario_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    table = aggregates[
        [
            "method",
            "split",
            "scenario_short",
            "success_rate",
            "mean_episode_return",
            "mean_dist_to_goal",
            "mean_stray_count",
            "mean_collision_count",
            "mean_dog_path_length",
        ]
    ].copy()
    table["method"] = _ordered_methods(table["method"])
    return table.sort_values(["split", "scenario_short", "method"]).reset_index(drop=True)


def _build_method_ranking(overall: pd.DataFrame) -> pd.DataFrame:
    unseen = overall[overall["split"] == "unseen"].copy()
    if unseen.empty:
        return unseen
    unseen["rank_success"] = unseen["success_rate"].rank(ascending=False, method="min")
    unseen["rank_return"] = unseen["mean_episode_return"].rank(ascending=False, method="min")
    unseen["rank_distance"] = unseen["mean_dist_to_goal"].rank(ascending=True, method="min")
    unseen["rank_efficiency"] = unseen["mean_dog_path_length"].rank(ascending=True, method="min")
    unseen["composite_rank"] = (
        unseen["rank_success"]
        + unseen["rank_return"]
        + unseen["rank_distance"]
        + unseen["rank_efficiency"]
    )
    unseen["method"] = _ordered_methods(unseen["method"])
    return unseen.sort_values(["composite_rank", "method"]).reset_index(drop=True)


def _plot_main_dashboard(overall: pd.DataFrame, gaps: pd.DataFrame, output_path: Path) -> None:
    unseen = overall[overall["split"] == "unseen"].copy()
    unseen["method"] = _ordered_methods(unseen["method"])
    gaps["method"] = _ordered_methods(gaps["method"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Shepherding Comparison Dashboard", fontsize=20, fontweight="bold")

    _barh_with_labels(
        axes[0, 0],
        unseen.sort_values("success_rate", ascending=True),
        x="success_rate",
        y="method",
        title="Unseen Success Rate",
        xlabel="Success Rate",
        formatter="{:.2f}",
        xlim=(0.0, 1.0),
    )

    _dumbbell_success_plot(axes[0, 1], gaps)

    _barh_with_labels(
        axes[1, 0],
        unseen.sort_values("mean_dist_to_goal", ascending=False),
        x="mean_dist_to_goal",
        y="method",
        title="Unseen Mean Distance to Goal",
        xlabel="Mean Distance to Goal",
        formatter="{:.2f}",
    )

    _barh_with_labels(
        axes[1, 1],
        unseen.sort_values("mean_episode_return", ascending=True),
        x="mean_episode_return",
        y="method",
        title="Unseen Mean Episode Return",
        xlabel="Episode Return",
        formatter="{:.1f}",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_scenario_heatmaps(aggregates: pd.DataFrame, output_path: Path) -> None:
    methods = [m for m in DISPLAY_ORDER if m in set(aggregates["method"])]
    scenario_order = _scenario_order(aggregates["scenario_short"])
    success = (
        aggregates.pivot_table(index="method", columns="scenario_short", values="success_rate")
        .reindex(index=methods, columns=scenario_order)
    )
    distance = (
        aggregates.pivot_table(index="method", columns="scenario_short", values="mean_dist_to_goal")
        .reindex(index=methods, columns=scenario_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Scenario-by-Scenario Performance", fontsize=19, fontweight="bold")

    sns.heatmap(
        success,
        annot=success.round(2),
        fmt="",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "Success Rate"},
        ax=axes[0],
    )
    axes[0].set_title("Success Rate")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        distance,
        annot=distance.round(2),
        fmt="",
        cmap="YlOrRd_r",
        linewidths=0.5,
        cbar_kws={"label": "Mean Distance to Goal"},
        ax=axes[1],
    )
    axes[1].set_title("Goal Proximity")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_return_profiles(summaries: pd.DataFrame, output_path: Path) -> None:
    plot_df = summaries.copy()
    plot_df["method"] = _ordered_methods(plot_df["method"])
    order = [m for m in DISPLAY_ORDER if m in set(plot_df["method"])]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    fig.suptitle("Episode Return Profiles", fontsize=19, fontweight="bold")

    for ax, split in zip(axes, ["train", "unseen"]):
        split_df = plot_df[plot_df["split"] == split].copy()
        if split_df.empty:
            ax.axis("off")
            continue
        sns.boxplot(
            data=split_df,
            x="method",
            y="episode_return",
            order=order,
            palette=[PALETTE[name] for name in order],
            width=0.55,
            fliersize=0,
            ax=ax,
        )
        sns.stripplot(
            data=split_df,
            x="method",
            y="episode_return",
            order=order,
            hue="scenario_short",
            dodge=False,
            jitter=0.10,
            alpha=0.75,
            size=6,
            ax=ax,
        )
        legend = ax.legend(title="Scenario", loc="best", frameon=True)
        ax.set_title(split.title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=8)
        if split == "train" and legend is not None:
            legend.remove()
    axes[0].set_ylabel("Episode Return")
    axes[1].set_ylabel("")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_progress_tradeoff(overall: pd.DataFrame, output_path: Path) -> None:
    unseen = overall[overall["split"] == "unseen"].copy()
    unseen["method"] = _ordered_methods(unseen["method"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=unseen,
        x="mean_dog_path_length",
        y="mean_dist_to_goal",
        hue="method",
        hue_order=[m for m in DISPLAY_ORDER if m in set(unseen["method"])],
        palette=PALETTE,
        s=220,
        ax=ax,
    )
    for row in unseen.itertuples(index=False):
        ax.annotate(
            str(row.method),
            (row.mean_dog_path_length, row.mean_dist_to_goal),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_title("Unseen Efficiency vs Final Goal Proximity")
    ax.set_xlabel("Average Dog Path Length")
    ax.set_ylabel("Mean Distance to Goal")
    legend = ax.legend(title="Method", frameon=True)
    if legend is not None:
        legend.remove()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_bc_metrics(metrics: pd.Series, output_path: Path) -> None:
    error_keys = ["mse", "rmse", "mae", "mean_angle_error_deg"]
    fit_keys = ["r2_dx", "r2_dy"]
    errors = pd.DataFrame(
        [{"metric": key, "value": float(metrics[key])} for key in error_keys if key in metrics.index]
    )
    fit = pd.DataFrame(
        [{"metric": key, "value": float(metrics[key])} for key in fit_keys if key in metrics.index]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("Behavioral Cloning Offline Validation", fontsize=18, fontweight="bold")

    if not errors.empty:
        sns.barplot(
            data=errors,
            x="value",
            y="metric",
            color=PALETTE["Behavioral Cloning"],
            ax=axes[0],
        )
        axes[0].set_title("Error Metrics")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("")
    else:
        axes[0].axis("off")

    if not fit.empty:
        sns.barplot(
            data=fit,
            x="value",
            y="metric",
            color=PALETTE["Behavioral Cloning"],
            ax=axes[1],
        )
        axes[1].set_xlim(min(-0.1, float(fit["value"].min()) - 0.05), 1.0)
        axes[1].set_title("Fit Metrics")
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("")
    else:
        axes[1].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _barh_with_labels(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    formatter: str,
    xlim: tuple[float, float] | None = None,
) -> None:
    methods = list(data[y])
    colors = [PALETTE.get(str(method), "#6c757d") for method in methods]
    sns.barplot(data=data, x=x, y=y, palette=colors, orient="h", ax=ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for patch, value in zip(ax.patches, data[x].to_list()):
        xpos = patch.get_width()
        ypos = patch.get_y() + patch.get_height() / 2.0
        offset = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] > ax.get_xlim()[0] else 1.0)
        ax.text(xpos + offset, ypos, formatter.format(value), va="center", fontsize=10)


def _dumbbell_success_plot(ax: plt.Axes, gap_df: pd.DataFrame) -> None:
    methods = [m for m in DISPLAY_ORDER if m in set(gap_df["method"])]
    gap_df = gap_df.set_index("method").reindex(methods).reset_index()
    y_positions = range(len(gap_df))

    for idx, row in enumerate(gap_df.itertuples(index=False)):
        color = PALETTE.get(str(row.method), "#6c757d")
        ax.plot(
            [row.train_success_rate, row.unseen_success_rate],
            [idx, idx],
            color=color,
            linewidth=3,
            alpha=0.9,
        )
        ax.scatter(row.train_success_rate, idx, color=color, s=110, marker="o", zorder=3)
        ax.scatter(row.unseen_success_rate, idx, color=color, s=110, marker="s", zorder=3)
        ax.text(row.train_success_rate + 0.02, idx + 0.12, f"{row.train_success_rate:.2f}", fontsize=9)
        ax.text(row.unseen_success_rate + 0.02, idx - 0.22, f"{row.unseen_success_rate:.2f}", fontsize=9)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(m) for m in gap_df["method"]])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Success Rate")
    ax.set_title("Train vs Unseen Reliability")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylabel("")


def _scenario_order(values: pd.Series) -> list[str]:
    preferred = ["Train", "Split Field", "Open Field", "Corridor", "Dense", "Narrow Gate"]
    present = list(dict.fromkeys(values.tolist()))
    ordered = [name for name in preferred if name in present]
    ordered.extend([name for name in present if name not in ordered])
    return ordered


if __name__ == "__main__":
    main()
