"""Helpers for turning evaluation CSVs into report tables.

Two concerns are shared by every analysis script:

* **Graded delivery metrics.** ``success`` is all-or-nothing over the whole
  flock, so under wide domain randomization it sits at zero for long stretches
  and every figure keyed on it is flat and uninformative. ``fraction_at_goal``
  moves continuously and is what the generalization story should be told with.
* **The held-out split.** Results now carry three splits (``train``,
  ``unseen``, ``test``). Scripts that hardcode ``train``/``unseen`` silently drop
  the procedural test suite — the one number that is actually held out.

Everything here degrades gracefully on result directories produced before the
graded metrics existed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd


#: Per-episode columns written by ``EpisodeSummary``.
GRADED_EPISODE_COLUMNS: Tuple[str, ...] = (
    "fraction_at_goal",
    "best_fraction_at_goal",
    "max_dist_to_goal",
    "steps_to_80pct_collected",
)

#: Corresponding columns in an aggregated ``aggregate_metrics.csv``.
GRADED_AGGREGATE_COLUMNS: Tuple[str, ...] = (
    "mean_fraction_at_goal",
    "mean_best_fraction_at_goal",
    "mean_max_dist_to_goal",
)

#: Preference order when picking which split represents "held out". The
#: procedural suite is disjoint from the training distribution by construction,
#: so it outranks the five hand-authored presets.
HOLDOUT_SPLIT_PRIORITY: Tuple[str, ...] = ("test", "unseen")


def available_aggs(
    df: pd.DataFrame, spec: Dict[str, Tuple[str, str]]
) -> Dict[str, Tuple[str, str]]:
    """Drop aggregation entries whose source column is missing from *df*."""
    return {out: (col, op) for out, (col, op) in spec.items() if col in df.columns}


def preferred_holdout_split(df: pd.DataFrame, column: str = "split") -> str | None:
    """Return the most meaningful held-out split present in *df*."""
    if column not in df.columns:
        return None
    present = set(df[column].dropna().unique())
    for candidate in HOLDOUT_SPLIT_PRIORITY:
        if candidate in present:
            return candidate
    return None


def holdout_splits(df: pd.DataFrame, column: str = "split") -> List[str]:
    """Return every held-out split present, in priority order."""
    if column not in df.columns:
        return []
    present = set(df[column].dropna().unique())
    return [name for name in HOLDOUT_SPLIT_PRIORITY if name in present]


def generalization_gap_table(
    overall: pd.DataFrame,
    holdout_split: str,
    group_key: str = "method",
    metrics: Sequence[str] = ("success_rate", "fraction_at_goal", "mean_episode_return"),
) -> pd.DataFrame:
    """Pair per-group train scores against one held-out split.

    Returns a frame with ``train_<metric>``, ``holdout_<metric>`` and
    ``<metric>_gap`` columns plus a ``holdout_split`` label, so downstream plots
    do not need to know which split they are showing.
    """
    metrics = [metric for metric in metrics if metric in overall.columns]
    if not metrics or "split" not in overall.columns:
        return pd.DataFrame()

    columns = [group_key, *metrics]
    train = overall.loc[overall["split"] == "train", columns].rename(
        columns={metric: f"train_{metric}" for metric in metrics}
    )
    holdout = overall.loc[overall["split"] == holdout_split, columns].rename(
        columns={metric: f"holdout_{metric}" for metric in metrics}
    )
    if train.empty or holdout.empty:
        return pd.DataFrame()

    merged = train.merge(holdout, on=group_key, how="inner")
    for metric in metrics:
        # Distance-style metrics improve as they shrink; the sign convention here
        # is always "positive means the held-out split is worse".
        if "dist" in metric:
            merged[f"{metric}_gap"] = merged[f"holdout_{metric}"] - merged[f"train_{metric}"]
        else:
            merged[f"{metric}_gap"] = merged[f"train_{metric}"] - merged[f"holdout_{metric}"]
    merged.insert(1, "holdout_split", holdout_split)
    return merged


def warn_if_success_is_degenerate(df: pd.DataFrame, column: str = "success") -> None:
    """Print a note when the strict success rate carries no signal."""
    if column not in df.columns or df.empty:
        return
    if float(df[column].max()) <= 0.0:
        print(
            f"[note] Every episode has {column}=0, so success-rate figures will be "
            "flat. Read the fraction-at-goal tables and figures instead: the "
            "all-or-nothing flag requires every sheep inside the goal radius."
        )
