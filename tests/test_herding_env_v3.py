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

    def test_egocentric_observation_stays_normalized(self) -> None:
        """No feature may leave the normalized range, in particular for unseen sheep.

        The legacy layout marked invisible sheep with 999.0, which dominated the
        first layer of an unnormalized MLP.
        """
        env = HerdingEnvV3(visibility_radius=1.0, curriculum_mode=False)
        obs, _ = env.reset(seed=5, options={"scenario": "unseen_dense"})
        for _ in range(60):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertLessEqual(float(np.max(np.abs(obs))), 10.0)
            if terminated or truncated:
                break
        env.close()

    def test_invisible_sheep_encoded_as_zero_with_flag(self) -> None:
        env = HerdingEnvV3(curriculum_mode=False)
        env.reset(seed=1, options={"scenario": "unseen_open_field"})
        env.visibility_radius = 0.01  # named presets override the constructor value
        obs = env._get_obs()
        sheep_block = obs[26 : 26 + 3 * env.max_sheep].reshape(env.max_sheep, 3)
        np.testing.assert_allclose(sheep_block, 0.0)
        env.close()

    def test_goal_frame_observation_is_rotation_invariant(self) -> None:
        """Rotating the whole scene by 90 degrees must not change the observation."""
        env = HerdingEnvV3(observation_frame="goal", curriculum_mode=False, n_lidar_rays=4)
        env.reset(seed=0, options={"scenario": "unseen_open_field"})
        grid = env.grid_size

        env.dog_pos = np.array([6.0, 5.0], dtype=np.float32)
        env.sheep_pos = np.array([[8.0, 7.0], [9.0, 6.0]], dtype=np.float32)
        env.n_sheep = 2
        env.goal = np.array([15.0, 14.0], dtype=np.float32)
        env.obstacles = []
        env.visibility_radius = 20.0
        env._last_seen_centroid = np.mean(env.sheep_pos, axis=0)
        baseline = env._get_obs()

        def rotate(point: np.ndarray) -> np.ndarray:
            # (x, y) -> (grid - y, x): a 90 degree rotation about the arena centre.
            return np.array([grid - point[1], point[0]], dtype=np.float32)

        env.dog_pos = rotate(env.dog_pos)
        env.sheep_pos = np.stack([rotate(p) for p in env.sheep_pos])
        env.goal = rotate(env.goal)
        env._last_seen_centroid = np.mean(env.sheep_pos, axis=0)
        rotated = env._get_obs()

        np.testing.assert_allclose(baseline, rotated, atol=1e-4)
        env.close()

    def test_procedural_scenarios_are_seed_deterministic(self) -> None:
        env = HerdingEnvV3(curriculum_mode=False, max_sheep=16, randomize_sheep_count=True)
        first, info_a = env.reset(seed=42, options={"scenario": "test_procedural"})
        second, info_b = env.reset(seed=42, options={"scenario": "test_procedural"})
        np.testing.assert_allclose(first, second)
        self.assertEqual(info_a["topology"], info_b["topology"])
        env.close()

    def test_procedural_topologies_are_sampled(self) -> None:
        env = HerdingEnvV3(curriculum_mode=False)
        topologies = set()
        for seed in range(40):
            _, info = env.reset(seed=seed, options={"scenario": "train"})
            topologies.add(info["topology"])
        self.assertGreaterEqual(len(topologies), 3)
        env.close()

    def test_graded_delivery_metrics_are_reported(self) -> None:
        env = HerdingEnvV3(curriculum_mode=False)
        env.reset(seed=0, options={"scenario": "unseen_open_field"})
        _, _, _, _, info = env.step(np.array([0.1, 0.1]))
        for key in (
            "fraction_at_goal",
            "best_fraction_at_goal",
            "steps_to_80pct_collected",
            "max_dist_to_goal",
        ):
            self.assertIn(key, info)
        env.close()

    def test_sheep_count_randomization_requires_egocentric_mode(self) -> None:
        with self.assertRaises(ValueError):
            HerdingEnvV3(observation_mode="legacy", randomize_sheep_count=True)

    def test_legacy_observation_layout_is_preserved(self) -> None:
        env = HerdingEnvV3(observation_mode="legacy", curriculum_mode=False)
        expected = 4 + 2 * env.n_sheep + 4 * env.max_obstacles
        obs, _ = env.reset(seed=0, options={"scenario": "unseen_open_field"})
        self.assertEqual(obs.shape, (expected,))
        env.close()


if __name__ == "__main__":
    unittest.main()
