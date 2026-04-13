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
