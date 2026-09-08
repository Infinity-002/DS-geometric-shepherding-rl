from __future__ import annotations

import unittest

import pandas as pd

from shepherding.research.reporting import (
    available_aggs,
    generalization_gap_table,
    holdout_splits,
    preferred_holdout_split,
)


def _overall() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "A", "split": "train", "success_rate": 0.8,
             "fraction_at_goal": 0.9, "mean_dist_to_goal": 2.0},
            {"method": "A", "split": "unseen", "success_rate": 0.4,
             "fraction_at_goal": 0.6, "mean_dist_to_goal": 5.0},
            {"method": "A", "split": "test", "success_rate": 0.2,
             "fraction_at_goal": 0.5, "mean_dist_to_goal": 7.0},
        ]
    )


class ReportingTests(unittest.TestCase):
    def test_procedural_test_split_outranks_named_presets(self) -> None:
        """The bug this guards: `test` rows silently dropped from every report."""
        self.assertEqual(preferred_holdout_split(_overall()), "test")
        self.assertEqual(holdout_splits(_overall()), ["test", "unseen"])

    def test_falls_back_to_unseen_for_older_results(self) -> None:
        old = _overall()[lambda df: df["split"] != "test"]
        self.assertEqual(preferred_holdout_split(old), "unseen")

    def test_returns_none_when_no_holdout_split_exists(self) -> None:
        train_only = _overall()[lambda df: df["split"] == "train"]
        self.assertIsNone(preferred_holdout_split(train_only))
        self.assertEqual(holdout_splits(train_only), [])

    def test_gap_is_positive_when_the_holdout_split_is_worse(self) -> None:
        gaps = generalization_gap_table(
            _overall(), holdout_split="test",
            metrics=("success_rate", "fraction_at_goal", "mean_dist_to_goal"),
        )
        row = gaps.iloc[0]
        self.assertEqual(row["holdout_split"], "test")
        self.assertAlmostEqual(row["success_rate_gap"], 0.6)
        self.assertAlmostEqual(row["fraction_at_goal_gap"], 0.4)
        # Distance improves as it shrinks, so the sign convention is inverted.
        self.assertAlmostEqual(row["mean_dist_to_goal_gap"], 5.0)

    def test_gap_table_is_empty_when_a_side_is_missing(self) -> None:
        train_only = _overall()[lambda df: df["split"] == "train"]
        self.assertTrue(generalization_gap_table(train_only, "test").empty)

    def test_available_aggs_drops_missing_columns(self) -> None:
        df = pd.DataFrame({"success": [1, 0]})
        spec = {"a": ("success", "mean"), "b": ("fraction_at_goal", "mean")}
        self.assertEqual(available_aggs(df, spec), {"a": ("success", "mean")})


if __name__ == "__main__":
    unittest.main()
