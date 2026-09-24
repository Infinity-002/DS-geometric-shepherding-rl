import unittest

from shepherding.research.models import _with_lr_schedule


class LearningRateScheduleTest(unittest.TestCase):
    def test_linear_schedule_decays_to_zero(self) -> None:
        config = _with_lr_schedule({"learning_rate": 3e-4, "lr_schedule": "linear"})
        self.assertNotIn("lr_schedule", config)
        self.assertAlmostEqual(config["learning_rate"](1.0), 3e-4)
        self.assertAlmostEqual(config["learning_rate"](0.5), 1.5e-4)
        self.assertAlmostEqual(config["learning_rate"](0.0), 0.0)

    def test_constant_is_the_default(self) -> None:
        config = _with_lr_schedule({"learning_rate": 3e-4})
        self.assertEqual(config["learning_rate"], 3e-4)


if __name__ == "__main__":
    unittest.main()
