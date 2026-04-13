from __future__ import annotations

import unittest

import numpy as np

from shepherding.envs.herding_env_v3 import HerdingEnvV3


class HerdingEnvV3Tests(unittest.TestCase):
    def test_reset_is_deterministic_for_named_scenario(self) -> None:
        env = HerdingEnvV3(domain_randomization=False, curriculum_mode=False)
        obs_a, info_a = env.reset(seed=123, options={"scenario": "unseen_dense"})
        obs_b, info_b = env.reset(seed=123, options={"scenario": "unseen_dense"})

        np.testing.assert_allclose(obs_a, obs_b)
        self.assertEqual(info_a["scenario"], "unseen_dense")
        self.assertEqual(info_b["scenario"], "unseen_dense")
        env.close()

    def test_step_returns_expected_shapes_and_reward_terms(self) -> None:
        env = HerdingEnvV3(domain_randomization=False, curriculum_mode=False)
        obs, _ = env.reset(seed=0, options={"scenario": "unseen_open_field"})
        next_obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.2]))

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(next_obs.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("reward_progress", info)
        self.assertIn("visible_ratio", info)
        env.close()


if __name__ == "__main__":
    unittest.main()
