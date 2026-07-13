import unittest

from harnesscad.data.dataengine.reward.intent2exec_overlong_filter import (
    Sequence_,
    filter_overlong,
    is_truncated,
    rl_objective,
    truncated_indices,
)


def _seq(length, reward=1.0, log_prob=-1.0):
    return Sequence_(length=length, reward=reward, log_prob=log_prob)


class TestTruncated(unittest.TestCase):
    def test_at_limit(self):
        self.assertTrue(is_truncated(100, 100))

    def test_under(self):
        self.assertFalse(is_truncated(99, 100))

    def test_bad_tmax(self):
        with self.assertRaises(ValueError):
            is_truncated(10, 0)


class TestFilter(unittest.TestCase):
    def setUp(self):
        self.seqs = [_seq(10), _seq(100), _seq(50), _seq(120)]

    def test_keeps_short(self):
        kept = filter_overlong(self.seqs, 100)
        self.assertEqual([s.length for s in kept], [10, 50])

    def test_truncated_indices(self):
        self.assertEqual(truncated_indices(self.seqs, 100), [1, 3])


class TestObjective(unittest.TestCase):
    def test_mean_over_kept(self):
        seqs = [_seq(10, reward=2.0, log_prob=-1.0),
                _seq(200, reward=9.0, log_prob=-1.0)]
        # only the first survives: 2.0 * -1.0
        self.assertAlmostEqual(rl_objective(seqs, 100), -2.0)

    def test_all_truncated(self):
        seqs = [_seq(200), _seq(300)]
        self.assertEqual(rl_objective(seqs, 100), 0.0)

    def test_verbose_correct_not_penalised_relative(self):
        # A long correct sample is dropped, not dragged in with a harsh penalty.
        kept_only_short = rl_objective([_seq(10, 5.0, -0.5)], 100)
        with_long = rl_objective([_seq(10, 5.0, -0.5), _seq(500, 5.0, -0.5)], 100)
        self.assertAlmostEqual(kept_only_short, with_long)


if __name__ == "__main__":
    unittest.main()
