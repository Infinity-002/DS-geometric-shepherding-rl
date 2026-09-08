from __future__ import annotations

import unittest

import numpy as np

from shepherding.utils.geometry_v2 import ray_angles, raycast_distances


class RaycastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = np.array([10.0, 10.0], dtype=np.float32)
        self.angles = ray_angles(4)  # east, north, west, south

    def test_empty_arena_returns_boundary_distances(self) -> None:
        hits = raycast_distances(self.origin, self.angles, [], 20.0, 30.0)
        np.testing.assert_allclose(hits, [10.0, 10.0, 10.0, 10.0], atol=1e-4)

    def test_wall_occludes_the_ray_facing_it(self) -> None:
        hits = raycast_distances(self.origin, self.angles, [(15.0, 0.0, 1.0, 20.0)], 20.0, 30.0)
        self.assertAlmostEqual(float(hits[0]), 5.0, places=4)
        np.testing.assert_allclose(hits[1:], [10.0, 10.0, 10.0], atol=1e-4)

    def test_obstacles_behind_the_ray_are_ignored(self) -> None:
        hits = raycast_distances(self.origin, self.angles, [(2.0, 9.0, 1.0, 2.0)], 20.0, 30.0)
        self.assertAlmostEqual(float(hits[0]), 10.0, places=4)
        self.assertAlmostEqual(float(hits[2]), 7.0, places=4)

    def test_distances_are_clipped_to_max_range(self) -> None:
        hits = raycast_distances(self.origin, self.angles, [], 20.0, 3.0)
        np.testing.assert_allclose(hits, 3.0, atol=1e-4)

    def test_a_wall_and_a_blob_of_equal_offset_read_identically(self) -> None:
        """The whole point of the lidar encoding: shape is not directly observed."""
        wall = raycast_distances(self.origin, ray_angles(1), [(14.0, 0.0, 1.0, 20.0)], 20.0, 20.0)
        blob = raycast_distances(self.origin, ray_angles(1), [(14.0, 9.5, 1.0, 1.0)], 20.0, 20.0)
        np.testing.assert_allclose(wall, blob, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
