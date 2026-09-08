from __future__ import annotations

import unittest

import numpy as np

from shepherding.scenarios.generators import (
    TEST_LAYOUT_SPEC,
    TRAIN_LAYOUT_SPEC,
    sample_goal_and_spawn,
    sample_layout,
)


class GeneratorTests(unittest.TestCase):
    def test_all_topologies_are_reachable(self) -> None:
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(300):
            topology, _ = sample_layout(
                rng, 20.0, TRAIN_LAYOUT_SPEC, keepouts=[(3.0, 3.0), (17.0, 17.0)],
                keepout_radius=2.0, max_obstacles=8,
            )
            seen.add(topology)
        self.assertTrue({"blobs", "corridor", "gate", "bars"}.issubset(seen))

    def test_layouts_keep_start_and_goal_connected(self) -> None:
        """A sampled gate or corridor must never seal the goal off."""
        rng = np.random.default_rng(1)
        keepouts = [(3.0, 3.0), (17.0, 17.0)]
        for _ in range(200):
            _, obstacles = sample_layout(
                rng, 20.0, TEST_LAYOUT_SPEC, keepouts=keepouts,
                keepout_radius=2.0, max_obstacles=8,
            )
            from shepherding.scenarios.generators import _is_connected

            self.assertTrue(_is_connected(obstacles, 20.0, keepouts))

    def test_obstacles_avoid_keepout_points(self) -> None:
        rng = np.random.default_rng(2)
        keepouts = [(4.0, 4.0), (16.0, 15.0)]
        for _ in range(100):
            _, obstacles = sample_layout(
                rng, 20.0, TRAIN_LAYOUT_SPEC, keepouts=keepouts,
                keepout_radius=2.0, max_obstacles=8,
            )
            for rx, ry, rw, rh in obstacles:
                for px, py in keepouts:
                    cx = float(np.clip(px, rx, rx + rw))
                    cy = float(np.clip(py, ry, ry + rh))
                    self.assertGreaterEqual(float(np.hypot(px - cx, py - cy)), 2.0)

    def test_goals_cover_all_quadrants(self) -> None:
        """Goals must not stay in the top-right corner the way they used to."""
        rng = np.random.default_rng(3)
        quadrants = set()
        for _ in range(400):
            goal, _ = sample_goal_and_spawn(rng, 20.0)
            quadrants.add((goal[0] > 10.0, goal[1] > 10.0))
        self.assertEqual(len(quadrants), 4)

    def test_goal_and_spawn_are_separated(self) -> None:
        rng = np.random.default_rng(4)
        for _ in range(200):
            goal, bounds = sample_goal_and_spawn(rng, 20.0, min_separation_frac=0.45)
            centre = np.array(
                [
                    (bounds[0][0] + bounds[1][0]) * 0.5 * 20.0,
                    (bounds[0][1] + bounds[1][1]) * 0.5 * 20.0,
                ]
            )
            self.assertGreaterEqual(float(np.linalg.norm(centre - np.array(goal))), 8.0)

    def test_max_obstacles_is_respected(self) -> None:
        rng = np.random.default_rng(5)
        for _ in range(100):
            _, obstacles = sample_layout(
                rng, 20.0, TEST_LAYOUT_SPEC, keepouts=[(3.0, 3.0), (17.0, 17.0)],
                keepout_radius=2.0, max_obstacles=6,
            )
            self.assertLessEqual(len(obstacles), 6)


if __name__ == "__main__":
    unittest.main()
