from __future__ import annotations

import unittest

import numpy as np

from shepherding.research.callbacks import AdaptiveCurriculumCallback


class AdaptiveCurriculumTests(unittest.TestCase):
    def test_stage_advances_when_thresholds_are_met(self) -> None:
        callback = AdaptiveCurriculumCallback(
            stages=[
                {"stage": 0.0},
                {"stage": 0.5, "min_success_rate": 0.5, "min_visibility_ratio": 0.4},
                {"stage": 1.0, "min_success_rate": 0.8, "min_visibility_ratio": 0.6},
            ],
            window=5,
            warmup_episodes=2,
        )
        callback.successes = [1.0, 1.0, 1.0, 1.0]
        callback.visibilities = [0.8, 0.8, 0.8, 0.8]
        callback.collision_events = [0.0, 0.0, 0.0, 0.0]
        callback.progress_rewards = [0.1, 0.1, 0.1, 0.1]

        self.assertAlmostEqual(callback._compute_stage(), 1.0)

    def test_warmup_holds_initial_stage(self) -> None:
        callback = AdaptiveCurriculumCallback(
            stages=[{"stage": 0.0}, {"stage": 1.0, "min_success_rate": 0.5}],
            window=5,
            warmup_episodes=10,
        )
        callback.successes = [1.0, 1.0]
        callback.visibilities = [0.9, 0.9]
        callback.collision_events = [0.0, 0.0]
        callback.progress_rewards = [0.1, 0.1]

        self.assertAlmostEqual(callback._compute_stage(), 0.0)


if __name__ == "__main__":
    unittest.main()


class CurriculumInstrumentationTests(unittest.TestCase):
    def test_stage_occupancy_is_tracked(self) -> None:
        from shepherding.research.callbacks import StageTracker

        tracker = StageTracker()
        tracker.update(0.0, 1000)
        tracker.update(0.0, 2000)
        tracker.record_transition(0.33, 2000)
        tracker.update(0.33, 5000)

        summary = tracker.summary(final_stage=0.33)
        self.assertEqual(summary["final_stage"], 0.33)
        self.assertFalse(summary["reached_final_stage"])
        self.assertEqual(summary["timesteps_per_stage"][0.0], 2000)
        self.assertEqual(summary["timesteps_per_stage"][0.33], 3000)
        self.assertAlmostEqual(sum(summary["fraction_per_stage"].values()), 1.0)
        self.assertEqual(len(summary["stage_transitions"]), 1)

    def test_fraction_at_goal_can_gate_the_curriculum(self) -> None:
        """The strict success rate stays at zero far too long to drive the gate."""
        callback = AdaptiveCurriculumCallback(
            stages=[{"stage": 0.0}, {"stage": 1.0, "min_fraction_at_goal": 0.6}],
            window=5,
            warmup_episodes=2,
        )
        callback.successes = [0.0] * 4
        callback.fractions_at_goal = [0.7] * 4
        callback.visibilities = [0.8] * 4
        callback.collision_events = [0.0] * 4
        callback.progress_rewards = [0.1] * 4
        self.assertAlmostEqual(callback._compute_stage(), 1.0)

        callback.fractions_at_goal = [0.2] * 4
        self.assertAlmostEqual(callback._compute_stage(), 0.0)
