from __future__ import annotations

import unittest
import importlib.util

import numpy as np

from shepherding.imitation.features import feature_names, observation_to_features
from shepherding.imitation.model import BehavioralCloningAgent


class _StubEstimator:
    def __init__(self) -> None:
        self.last_columns: list[str] | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        batch = x.shape[0]
        self.last_columns = list(getattr(x, "columns", []))
        return np.tile(np.array([[3.0, 4.0]], dtype=np.float32), (batch, 1))


class ImitationTests(unittest.TestCase):
    def test_feature_vector_has_expected_size(self) -> None:
        n_sheep = 3
        max_obstacles = 2
        observation = np.array(
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

        features = observation_to_features(observation, n_sheep=n_sheep)

        self.assertEqual(features.shape[0], len(feature_names(n_sheep, max_obstacles)))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_behavioral_cloning_agent_normalizes_action(self) -> None:
        estimator = _StubEstimator()
        expected_names = feature_names(3, 2)
        agent = BehavioralCloningAgent(
            estimator=estimator,
            n_sheep=3,
            max_obstacles=2,
            model_feature_names=expected_names,
        )
        observation = np.zeros(18, dtype=np.float32)

        action, _ = agent.predict(observation)

        self.assertTrue(np.allclose(action, np.array([0.6, 0.8], dtype=np.float32)))
        self.assertLessEqual(float(np.linalg.norm(action)), 1.0001)
        if importlib.util.find_spec("pandas") is not None:
            self.assertEqual(estimator.last_columns, expected_names)
        else:
            self.assertEqual(estimator.last_columns, [])


if __name__ == "__main__":
    unittest.main()
