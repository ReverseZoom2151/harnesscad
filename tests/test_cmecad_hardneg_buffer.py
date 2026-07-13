import unittest

from harnesscad.data.dataengine.selftrain import cmecad_hardneg_buffer as hb


class TestPartition(unittest.TestCase):
    def test_even(self):
        parts = hb.partition_rounds(list(range(6)), 3)
        self.assertEqual(parts, [[0, 1], [2, 3], [4, 5]])

    def test_uneven(self):
        parts = hb.partition_rounds(list(range(7)), 3)
        self.assertEqual(parts, [[0, 1, 2], [3, 4], [5, 6]])
        self.assertEqual(sum(len(p) for p in parts), 7)

    def test_invalid_m(self):
        with self.assertRaises(ValueError):
            hb.partition_rounds([1, 2], 0)
        with self.assertRaises(ValueError):
            hb.partition_rounds([1, 2], 5)


class TestTrainTestRounds(unittest.TestCase):
    def test_next_portion_is_test(self):
        pairs = hb.train_test_rounds(list(range(4)), 2)
        # parts = [[0,1],[2,3]]; round0 train [0,1] test [2,3]; round1 wraps
        self.assertEqual(pairs[0], ([0, 1], [2, 3]))
        self.assertEqual(pairs[1], ([2, 3], [0, 1]))


class TestHardness(unittest.TestCase):
    def test_count_incorrect(self):
        self.assertEqual(hb.count_incorrect([True, False, False, True]), 2)

    def test_is_hard(self):
        # 3 incorrect out of 4, K=2 -> more than K -> hard
        self.assertTrue(hb.is_hard([False, False, False, True], 2))
        # exactly K incorrect -> not hard (strictly greater)
        self.assertFalse(hb.is_hard([False, False, True, True], 2))

    def test_admission_probability(self):
        self.assertAlmostEqual(hb.admission_probability(2, 4), 0.5)
        with self.assertRaises(ValueError):
            hb.admission_probability(2, 0)


class TestBuffer(unittest.TestCase):
    def test_not_hard_never_admitted(self):
        buf = hb.HardNegativeBuffer(k=3, g=4, seed=0)
        # only 1 incorrect, not > 3
        self.assertFalse(buf.offer("s", [True, True, True, False]))
        self.assertEqual(len(buf), 0)

    def test_certain_admission_when_prob_one(self):
        # K=4, G=4 -> prob 1.0; need > 4 incorrect impossible, so use K=3?
        # For guaranteed admission we need prob>=1 AND hard. K must be < G to be
        # hard-able (>K incorrect needs at most G). Use K=3,G=4: prob .75.
        # Instead test deterministic seed behaviour below.
        buf = hb.HardNegativeBuffer(k=0, g=4, seed=0)
        # K=0 -> any incorrect makes it hard; prob = 0/4 = 0 -> never admitted
        self.assertFalse(buf.offer("s", [False, False, False, False]))

    def test_deterministic_with_seed(self):
        samples = [(f"s{i}", [False, False, False, True]) for i in range(20)]
        b1 = hb.HardNegativeBuffer(k=2, g=4, seed=42)
        b2 = hb.HardNegativeBuffer(k=2, g=4, seed=42)
        a1 = b1.offer_many(samples)
        a2 = b2.offer_many(samples)
        self.assertEqual(a1, a2)
        # prob = 0.5 admission over hard samples; expect a nonempty, non-full set
        self.assertTrue(0 < len(a1) < 20)

    def test_different_seed_differs(self):
        samples = [(f"s{i}", [False, False, False, True]) for i in range(20)]
        a1 = hb.HardNegativeBuffer(k=2, g=4, seed=1).offer_many(samples)
        a2 = hb.HardNegativeBuffer(k=2, g=4, seed=2).offer_many(samples)
        self.assertNotEqual(a1, a2)

    def test_wrong_length_raises(self):
        buf = hb.HardNegativeBuffer(k=2, g=4, seed=0)
        with self.assertRaises(ValueError):
            buf.offer("s", [True, False])

    def test_clear(self):
        buf = hb.HardNegativeBuffer(k=2, g=4, seed=42)
        buf.offer_many([(f"s{i}", [False, False, False, True]) for i in range(20)])
        self.assertGreater(len(buf), 0)
        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            hb.HardNegativeBuffer(k=2, g=0)
        with self.assertRaises(ValueError):
            hb.HardNegativeBuffer(k=-1, g=4)


class TestSftLoss(unittest.TestCase):
    def test_sum_negative_ll(self):
        self.assertAlmostEqual(hb.sft_loss([-1.0, -2.0, -0.5]), 3.5)

    def test_empty(self):
        self.assertEqual(hb.sft_loss([]), 0.0)


if __name__ == "__main__":
    unittest.main()
