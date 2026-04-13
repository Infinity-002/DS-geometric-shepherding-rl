from __future__ import annotations

import unittest

from shepherding.scenarios import available_scenarios, scenario_presets


class ScenarioLibraryTests(unittest.TestCase):
    def test_expected_scenarios_exist(self) -> None:
        names = available_scenarios(20.0)
        self.assertIn("unseen_corridor", names)
        self.assertIn("unseen_dense", names)
        self.assertIn("unseen_narrow_gate", names)

    def test_presets_have_spawn_bounds(self) -> None:
        presets = scenario_presets(20.0)
        self.assertIsNotNone(presets["unseen_dense"].spawn_bounds)
        self.assertIsInstance(presets["unseen_open_field"].obstacles, tuple)


if __name__ == "__main__":
    unittest.main()
