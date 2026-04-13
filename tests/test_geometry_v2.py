from __future__ import annotations

import unittest

import numpy as np

from shepherding.utils.geometry_v2 import clip_to_free_space, visible_sheep_mask


class GeometryV2Tests(unittest.TestCase):
    def test_clip_blocks_segment_tunneling(self) -> None:
        obstacle = [(2.0, 0.0, 1.0, 2.0)]
        current = np.array([0.0, 1.0], dtype=np.float32)
        proposed = np.array([4.0, 1.0], dtype=np.float32)

        clipped = clip_to_free_space(current, proposed, obstacle, grid_size=10.0)

        self.assertLess(float(clipped[0]), 2.0)
        self.assertAlmostEqual(float(clipped[1]), 1.0, places=4)

    def test_visible_sheep_mask_respects_radius(self) -> None:
        dog = np.array([0.0, 0.0], dtype=np.float32)
        sheep = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float32)
        mask = visible_sheep_mask(dog, sheep, visibility_radius=2.0)
        np.testing.assert_array_equal(mask, np.array([True, False]))


if __name__ == "__main__":
    unittest.main()
