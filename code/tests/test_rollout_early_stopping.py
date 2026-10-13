import unittest

from diffusion_policy.workspace.rollout_early_stopping import (
    update_rollout_early_stopping,
)


class RolloutEarlyStoppingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "enabled": True,
            "monitor_key": "test_mean_score",
            "threshold": 1.0,
            "patience": 3,
        }

    def test_stops_after_three_consecutive_full_successes(self):
        streak = 0
        for expected_streak in (1, 2, 3):
            result = update_rollout_early_stopping(
                {"test_mean_score": 1.0}, self.cfg, streak)
            streak = result.streak
            self.assertEqual(streak, expected_streak)
            self.assertEqual(result.should_stop, expected_streak == 3)

    def test_failure_resets_streak(self):
        result = update_rollout_early_stopping(
            {"test_mean_score": 0.95}, self.cfg, 2)
        self.assertEqual(result.streak, 0)
        self.assertFalse(result.should_stop)

    def test_accepts_robomimic_metric_alias(self):
        result = update_rollout_early_stopping(
            {"test/mean_score": 1.0}, self.cfg, 2)
        self.assertEqual(result.score_key, "test/mean_score")
        self.assertTrue(result.should_stop)

    def test_disabled_returns_none(self):
        self.assertIsNone(update_rollout_early_stopping(
            {}, {"enabled": False}, 0))

    def test_enabled_missing_metric_is_explicit_error(self):
        with self.assertRaises(KeyError):
            update_rollout_early_stopping({"train_loss": 0.1}, self.cfg, 0)


if __name__ == "__main__":
    unittest.main()
