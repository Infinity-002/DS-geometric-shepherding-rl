from __future__ import annotations

import unittest

import numpy as np

from shepherding.baselines import HeuristicShepherdAgent


class HeuristicBaselineTests(unittest.TestCase):
    def test_predict_returns_valid_action(self) -> None:
        agent = HeuristicShepherdAgent(
            n_sheep=3,
            max_obstacles=2,
            grid_size=20.0,
            visibility_radius=7.5,
            flee_radius=5.5,
            success_radius=2.0,
        )
        obs = np.array(
            [
                4.0,
                4.0,
                10.0,
                10.0,
                1.0,
                0.5,
                1.2,
                -0.1,
                999.0,
                999.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
            ],
            dtype=np.float32,
        )

        action, _ = agent.predict(obs, deterministic=True)

        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertLessEqual(float(np.linalg.norm(action)), 1.0001)

    def test_cluster_aware_predict_returns_valid_action(self) -> None:
        agent = HeuristicShepherdAgent(
            n_sheep=4,
            max_obstacles=0,
            grid_size=20.0,
            visibility_radius=7.5,
            flee_radius=5.5,
            success_radius=2.0,
            use_cluster_targets=True,
        )
        obs = np.array(
            [
                5.0,
                5.0,
                10.0,
                10.0,
                1.0,
                1.0,
                1.1,
                0.9,
                6.0,
                6.0,
                6.2,
                5.8,
            ],
            dtype=np.float32,
        )

        action, _ = agent.predict(obs, deterministic=True)

        self.assertEqual(action.shape, (2,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertLessEqual(float(np.linalg.norm(action)), 1.0001)


if __name__ == "__main__":
    unittest.main()
