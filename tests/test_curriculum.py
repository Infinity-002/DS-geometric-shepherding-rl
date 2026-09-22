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


class CurriculumHysteresisTests(unittest.TestCase):
    """The trial run flipped 0.00 <-> 0.33 inside a single rollout.

    ``_compute_stage`` is memoryless, so a metric resting on a threshold
    re-derives a different stage on consecutive steps. ``_resolve_stage`` wraps
    it with a dwell lockout and a demotion margin; these tests pin both.
    """

    def _callback(self, **kwargs) -> AdaptiveCurriculumCallback:
        callback = AdaptiveCurriculumCallback(
            stages=[
                {"stage": 0.0},
                {"stage": 0.33, "min_fraction_at_goal": 0.30, "min_progress_reward": -0.04},
                {"stage": 0.66, "min_fraction_at_goal": 0.55},
            ],
            window=5,
            warmup_episodes=2,
            demote_margin=0.25,
            min_dwell_steps=25000,
            **kwargs,
        )
        callback.successes = [0.0] * 4
        callback.visibilities = [0.8] * 4
        callback.collision_events = [0.0] * 4
        callback.progress_rewards = [0.0] * 4
        return callback

    def test_a_grazing_dip_does_not_demote(self) -> None:
        """0.28 against a 0.30 gate is inside the margin, so the stage holds."""
        callback = self._callback()
        callback.current_stage = 0.33
        callback.num_timesteps = 100_000
        callback._last_change_step = 40_000
        callback.fractions_at_goal = [0.28] * 4

        self.assertAlmostEqual(callback._compute_stage(), 0.0)
        self.assertAlmostEqual(callback._resolve_stage(), 0.33)

    def test_a_real_collapse_still_demotes(self) -> None:
        callback = self._callback()
        callback.current_stage = 0.33
        callback.num_timesteps = 100_000
        callback._last_change_step = 40_000
        callback.fractions_at_goal = [0.10] * 4

        self.assertAlmostEqual(callback._resolve_stage(), 0.0)

    def test_dwell_lockout_blocks_a_demotion_right_after_a_change(self) -> None:
        callback = self._callback()
        callback.current_stage = 0.33
        callback.num_timesteps = 100_000
        callback._last_change_step = 90_000  # 10k < the 25k lockout
        callback.fractions_at_goal = [0.10] * 4

        self.assertAlmostEqual(callback._resolve_stage(), 0.33)

    def test_promotion_ignores_the_demotion_margin(self) -> None:
        """Widening randomization stays as demanding as the config asks."""
        callback = self._callback()
        callback.current_stage = 0.0
        callback.num_timesteps = 100_000
        callback.fractions_at_goal = [0.28] * 4

        self.assertAlmostEqual(callback._compute_stage(margin=0.25), 0.33)
        self.assertAlmostEqual(callback._resolve_stage(), 0.0)

    def test_margin_loosens_a_negative_lower_bound_downward(self) -> None:
        """``min_progress_reward`` is negative; the margin must not tighten it."""
        callback = self._callback()
        callback.current_stage = 0.33
        callback.num_timesteps = 100_000
        callback._last_change_step = 40_000
        callback.fractions_at_goal = [0.45] * 4
        callback.progress_rewards = [-0.045] * 4  # misses -0.04, inside -0.05

        self.assertAlmostEqual(callback._compute_stage(), 0.0)
        self.assertAlmostEqual(callback._resolve_stage(), 0.33)

    def test_collision_ceiling_is_loosened_upward_by_the_margin(self) -> None:
        callback = AdaptiveCurriculumCallback(
            stages=[{"stage": 0.0}, {"stage": 0.33, "max_collision_event_count": 10.0}],
            window=5,
            warmup_episodes=2,
            demote_margin=0.25,
            min_dwell_steps=0,
        )
        callback.successes = [0.0] * 4
        callback.fractions_at_goal = [0.5] * 4
        callback.visibilities = [0.8] * 4
        callback.progress_rewards = [0.0] * 4
        callback.current_stage = 0.33
        callback.num_timesteps = 100_000

        callback.collision_events = [11.5] * 4  # over 10.0, under 12.5
        self.assertAlmostEqual(callback._resolve_stage(), 0.33)

        callback.collision_events = [13.0] * 4
        self.assertAlmostEqual(callback._resolve_stage(), 0.0)
