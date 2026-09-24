#!/usr/bin/env python3
"""Create clean, report-grade figures for heuristic vs BC vs RL comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
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
    preferred_holdout_split,
)


DISPLAY_NAMES = {
    "heuristic_cluster_aware_fast": "Heuristic",
    "behavioral_cloning_rf_fast": "Behavioral Cloning",
    "recurrent_domain_randomized_fast": "RL (Domain Randomized)",
    "rl_structured_eval_v2": "RL (Structured v3)",
}

DISPLAY_ORDER = [
    "Heuristic",
    "Behavioral Cloning",
    "RL (Domain Randomized)",
    "RL (Structured v3)",
]

PALETTE = {
    "Heuristic": "#3d5a80",
    "Behavioral Cloning": "#2a9d8f",
    "RL (Domain Randomized)": "#e01e37",
    "RL (Structured v3)": "#e01e37",
}

SCENARIO_LABELS = {
    "train": "Train",
    "test_procedural": "Procedural Test",
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
    parser.add_argument(
        "--exclude-run-names",
        nargs="*",
        default=[],
        help="Optional list of run_name values to exclude after merging results.",
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

    if args.exclude_run_names:
        excluded = set(args.exclude_run_names)
        summaries = summaries[~summaries["run_name"].isin(excluded)].copy()
        aggregates = aggregates[~aggregates["run_name"].isin(excluded)].copy()

    summaries["method"] = summaries["run_name"].map(_display_name)
    aggregates["method"] = aggregates["run_name"].map(_display_name)
    summaries = summaries[summaries["method"].isin(DISPLAY_ORDER)].copy()
    aggregates = aggregates[aggregates["method"].isin(DISPLAY_ORDER)].copy()
    summaries["scenario_short"] = summaries["scenario"].map(_scenario_label)
    aggregates["scenario_short"] = aggregates["scenario"].map(_scenario_label)

    overall = _build_overall_table(aggregates)
    scenario_table = _build_scenario_table(aggregates)

    # "test" (the procedural held-out suite) wins over "unseen" when present;
    # previously a results directory containing it would have had those rows
    # dropped from every table and figure.
    primary_holdout = preferred_holdout_split(overall)
    if primary_holdout is None:
        raise SystemExit(
            "No held-out split found in the results. Expected at least one of "
            "'test' or 'unseen' in episode_summaries.csv."
        )
    print(f"Reporting generalization against the '{primary_holdout}' split.")

    overall.to_csv(tables_dir / "overall_method_summary.csv", index=False)
    scenario_table.to_csv(tables_dir / "scenario_metric_matrix.csv", index=False)

    # A gap table per held-out split, so 'unseen' stays available for continuity
    # with older reports even when 'test' is the headline.
    gaps_by_split = {}
    for split_name in holdout_splits(overall):
        split_gaps = _build_generalization_table(overall, split_name)
        if split_gaps.empty:
            continue
        gaps_by_split[split_name] = split_gaps
        split_gaps.to_csv(tables_dir / f"generalization_gap_{split_name}.csv", index=False)
    gaps = gaps_by_split.get(primary_holdout, pd.DataFrame())

    ranking = _build_method_ranking(overall, primary_holdout)
    ranking.to_csv(tables_dir / "method_ranking.csv", index=False)

    _plot_main_dashboard(overall, gaps, primary_holdout, figures_dir / "main_dashboard.png")
    _plot_scenario_heatmaps(aggregates, figures_dir / "scenario_heatmaps.png")
    _plot_return_profiles(summaries, figures_dir / "return_profiles.png")
    _plot_progress_tradeoff(overall, primary_holdout, figures_dir / "progress_tradeoff.png")

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
    spec = {
        "success_rate": ("success_rate", "mean"),
        "fraction_at_goal": ("mean_fraction_at_goal", "mean"),
        "best_fraction_at_goal": ("mean_best_fraction_at_goal", "mean"),
        "mean_episode_return": ("mean_episode_return", "mean"),
        "mean_episode_length": ("mean_episode_length", "mean"),
        "mean_dist_to_goal": ("mean_dist_to_goal", "mean"),
        "mean_max_dist_to_goal": ("mean_max_dist_to_goal", "mean"),
        "mean_stray_count": ("mean_stray_count", "mean"),
        "mean_collision_count": ("mean_collision_count", "mean"),
        "mean_dog_path_length": ("mean_dog_path_length", "mean"),
    }
    grouped = aggregates.groupby(["method", "split"], as_index=False).agg(
        **available_aggs(aggregates, spec)
    )
    grouped["method"] = _ordered_methods(grouped["method"])
    return grouped.sort_values(["split", "method"]).reset_index(drop=True)


def _headline_metric(frame: pd.DataFrame) -> tuple[str, str]:
    """Pick the metric to lead the figures with.

    Prefers fraction-of-flock-delivered: the strict success flag needs every
    sheep inside the goal radius and is routinely zero for every method, which
    makes a success-only dashboard say nothing.
    """
    if "fraction_at_goal" in frame.columns and float(
        frame["fraction_at_goal"].fillna(0.0).max()
    ) > 0.0:
        return "fraction_at_goal", "Fraction of Flock Delivered"
    return "success_rate", "Success Rate"


def _build_generalization_table(overall: pd.DataFrame, holdout_split: str) -> pd.DataFrame:
    merged = generalization_gap_table(
        overall,
        holdout_split=holdout_split,
        group_key="method",
        metrics=(
            "success_rate",
            "fraction_at_goal",
            "mean_episode_return",
            "mean_dist_to_goal",
        ),
    )
    if merged.empty:
        return merged
    merged["method"] = _ordered_methods(merged["method"])
    return merged.sort_values("method").reset_index(drop=True)


def _build_scenario_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "method",
        "split",
        "scenario_short",
        "success_rate",
        "mean_fraction_at_goal",
        "mean_best_fraction_at_goal",
        "mean_episode_return",
        "mean_dist_to_goal",
        "mean_max_dist_to_goal",
        "mean_stray_count",
        "mean_collision_count",
        "mean_dog_path_length",
    ]
    table = aggregates[[c for c in columns if c in aggregates.columns]].copy()
    table["method"] = _ordered_methods(table["method"])
    return table.sort_values(["split", "scenario_short", "method"]).reset_index(drop=True)


def _build_method_ranking(overall: pd.DataFrame, holdout_split: str) -> pd.DataFrame:
    holdout = overall[overall["split"] == holdout_split].copy()
    if holdout.empty:
        return holdout
    metric, _ = _headline_metric(holdout)
    holdout["rank_delivery"] = holdout[metric].rank(ascending=False, method="min")
    holdout["rank_return"] = holdout["mean_episode_return"].rank(ascending=False, method="min")
    holdout["rank_distance"] = holdout["mean_dist_to_goal"].rank(ascending=True, method="min")
    holdout["rank_efficiency"] = holdout["mean_dog_path_length"].rank(
        ascending=True, method="min"
    )
    holdout["composite_rank"] = (
        holdout["rank_delivery"]
        + holdout["rank_return"]
        + holdout["rank_distance"]
        + holdout["rank_efficiency"]
    )
    holdout["ranking_metric"] = metric
    holdout["ranking_split"] = holdout_split
    holdout["method"] = _ordered_methods(holdout["method"])
    return holdout.sort_values(["composite_rank", "method"]).reset_index(drop=True)


def _plot_main_dashboard(
    overall: pd.DataFrame,
    gaps: pd.DataFrame,
    holdout_split: str,
    output_path: Path,
) -> None:
    if not gaps.empty:
        gaps = gaps.copy()
        gaps["method"] = _ordered_methods(gaps["method"])
    metric, metric_label = _headline_metric(overall)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Shepherding Comparison Dashboard", fontsize=20, fontweight="bold")

    _barh_with_labels(
        axes[0, 0],
        overall[overall["split"] == "train"].sort_values(metric, ascending=True),
        x=metric,
        y="method",
        title=f"Training {metric_label}",
        xlabel=metric_label,
        formatter="{:.2f}",
        xlim=(0.0, 1.0),
    )

    _dumbbell_gap_plot(axes[0, 1], gaps, metric, metric_label, holdout_split)

    _barh_with_labels(
        axes[1, 0],
        overall[overall["split"] == "train"].sort_values("mean_dist_to_goal", ascending=False),
        x="mean_dist_to_goal",
        y="method",
        title="Training Goal Proximity",
        xlabel="Mean Distance to Goal",
        formatter="{:.2f}",
    )

    _barh_with_labels(
        axes[1, 1],
        overall[overall["split"] == "train"].sort_values("mean_episode_return", ascending=True),
        x="mean_episode_return",
        y="method",
        title="Training Episode Return",
        xlabel="Episode Return",
        formatter="{:.1f}",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_scenario_heatmaps(aggregates: pd.DataFrame, output_path: Path) -> None:
    methods = [m for m in DISPLAY_ORDER if m in set(aggregates["method"])]
    scenario_order = _scenario_order(aggregates["scenario_short"])

    def pivot(column: str) -> pd.DataFrame | None:
        if column not in aggregates.columns:
            return None
        return aggregates.pivot_table(
            index="method", columns="scenario_short", values=column
        ).reindex(index=methods, columns=scenario_order)

    panels = [
        (pivot("success_rate"), "Success Rate by Scenario", "YlGnBu", (0.0, 1.0), "Success Rate"),
        (
            pivot("mean_fraction_at_goal"),
            "Fraction of Flock Delivered",
            "YlGnBu",
            (0.0, 1.0),
            "Fraction Delivered",
        ),
        (
            pivot("mean_dist_to_goal"),
            "Distance to Goal (Lower is Better)",
            "YlOrRd_r",
            None,
            "Goal Proximity",
        ),
    ]
    panels = [panel for panel in panels if panel[0] is not None]

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.5))
    axes = np.atleast_1d(axes)
    fig.suptitle("Scenario-by-Scenario Performance", fontsize=19, fontweight="bold")

    for ax, (data, title, cmap, limits, cbar_label) in zip(axes, panels):
        sns.heatmap(
            data,
            annot=data.round(2),
            fmt="",
            cmap=cmap,
            vmin=None if limits is None else limits[0],
            vmax=None if limits is None else limits[1],
            linewidths=1.5,
            cbar_kws={"label": cbar_label},
            ax=ax,
            square=True,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def _plot_return_profiles(summaries: pd.DataFrame, output_path: Path) -> None:
    plot_df = summaries.copy()
    plot_df["method"] = _ordered_methods(plot_df["method"])
    order = [m for m in DISPLAY_ORDER if m in set(plot_df["method"])]

    splits = [
        name for name in ("train", "unseen", "test") if name in set(plot_df["split"])
    ]
    if not splits:
        return
    fig, axes = plt.subplots(
        1, len(splits), figsize=(7.5 * len(splits), 5.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    fig.suptitle("Episode Return Profiles", fontsize=19, fontweight="bold")

    for ax, split in zip(axes, splits):
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
        if split != splits[-1] and legend is not None:
            legend.remove()
    axes[0].set_ylabel("Episode Return")
    for ax in axes[1:]:
        ax.set_ylabel("")
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_progress_tradeoff(
    overall: pd.DataFrame, holdout_split: str, output_path: Path
) -> None:
    holdout = overall[overall["split"] == holdout_split].copy()
    holdout["method"] = _ordered_methods(holdout["method"])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.scatterplot(
        data=holdout,
        x="mean_dog_path_length",
        y="mean_dist_to_goal",
        hue="method",
        hue_order=[m for m in DISPLAY_ORDER if m in set(holdout["method"])],
        palette=PALETTE,
        s=220,
        ax=ax,
    )
    for row in holdout.itertuples(index=False):
        ax.annotate(
            str(row.method),
            (row.mean_dog_path_length, row.mean_dist_to_goal),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=11,
        )
    ax.set_title(f"{holdout_split.title()} Efficiency vs Final Goal Proximity")
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
        _annotate_value_bars(axes[0], formatter="{:.3f}", inside_threshold=1.0)
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
        _annotate_value_bars(axes[1], formatter="{:.3f}", inside_threshold=0.25)
    else:
        axes[1].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _annotate_value_bars(
    ax: plt.Axes,
    *,
    formatter: str = "{:.2f}",
    inside_threshold: float = 0.2,
) -> None:
    x_min, x_max = ax.get_xlim()
    span = max(x_max - x_min, 1e-8)
    outside_offset = span * 0.012
    inside_offset = span * 0.02

    for patch in ax.patches:
        width = float(patch.get_width())
        ypos = patch.get_y() + patch.get_height() / 2.0
        label = formatter.format(width)

        if width >= inside_threshold:
            ax.text(
                width - inside_offset,
                ypos,
                label,
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                width + outside_offset,
                ypos,
                label,
                va="center",
                ha="left",
                fontsize=10,
                fontweight="bold",
                color="#1f2933",
            )


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
    plot_data = data.copy()
    methods = [str(method) for method in plot_data[y]]
    colors = [PALETTE.get(str(method), "#6c757d") for method in methods]
    sns.barplot(data=plot_data, x=x, y=y, palette=colors, orient="h", ax=ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")

    # Seaborn may render categorical bars in category order rather than row order,
    # so look up the displayed y-labels and annotate using the matching values.
    value_by_method = {
        str(method): float(value)
        for method, value in zip(plot_data[y].astype(str), plot_data[x])
    }
    rendered_methods = [tick.get_text() for tick in ax.get_yticklabels()]
    for patch, method in zip(ax.patches, rendered_methods):
        value = value_by_method.get(method)
        if value is None:
            continue
        xpos = patch.get_width()
        ypos = patch.get_y() + patch.get_height() / 2.0
        offset = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim()[1] > ax.get_xlim()[0] else 1.0)
        ax.text(xpos + offset, ypos, formatter.format(value), va="center", fontsize=10)


def _dumbbell_gap_plot(
    ax: plt.Axes,
    gap_df: pd.DataFrame,
    metric: str,
    metric_label: str,
    holdout_split: str,
) -> None:
    train_col = f"train_{metric}"
    holdout_col = f"holdout_{metric}"
    if gap_df.empty or train_col not in gap_df.columns:
        ax.axis("off")
        return

    methods = [m for m in DISPLAY_ORDER if m in set(gap_df["method"])]
    gap_df = gap_df.set_index("method").reindex(methods).reset_index()
    y_positions = range(len(gap_df))

    for idx, row in enumerate(gap_df.itertuples(index=False)):
        color = PALETTE.get(str(row.method), "#6c757d")
        train_value = float(getattr(row, train_col))
        holdout_value = float(getattr(row, holdout_col))
        ax.plot(
            [train_value, holdout_value], [idx, idx], color=color, linewidth=3, alpha=0.9
        )
        ax.scatter(train_value, idx, color=color, s=110, marker="o", zorder=3)
        ax.scatter(holdout_value, idx, color=color, s=110, marker="s", zorder=3)
        ax.text(train_value + 0.02, idx + 0.12, f"{train_value:.2f}", fontsize=9)
        ax.text(holdout_value + 0.02, idx - 0.22, f"{holdout_value:.2f}", fontsize=9)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([str(m) for m in gap_df["method"]])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(metric_label)
    ax.set_title(f"Train (o) vs {holdout_split.title()} (s) {metric_label}")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylabel("")


def _scenario_order(values: pd.Series) -> list[str]:
    preferred = [
        "Train",
        "Split Field",
        "Open Field",
        "Corridor",
        "Dense",
        "Narrow Gate",
        "Procedural Test",
    ]
    present = list(dict.fromkeys(values.tolist()))
    ordered = [name for name in preferred if name in present]
    ordered.extend([name for name in present if name not in ordered])
    return ordered


if __name__ == "__main__":
    main()
