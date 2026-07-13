import unittest

from harnesscad.domain.reconstruction.cadparser_schema import command, pad_sequence, SOS, EOS
from harnesscad.domain.reconstruction.cadparser_sequence_stats import (
    augment, length_distribution, operation_ratio, sequence_length, split_steps,
    truncate_last_step,
)


def sketch_extrude():
    return [command("L", x=0.0, y=0.0), command("L", x=1.0, y=0.0),
            command("L", x=1.0, y=1.0), command("E", e1=0.5, e2=0.0)]


def two_step():
    # sketch+extrude, then a chamfer step
    return sketch_extrude() + [command("Cf", px=0.0, py=0.0, pz=0.0)]


class TestSequenceStats(unittest.TestCase):
    def test_split_steps_folds_sketch_and_extrude(self):
        steps = split_steps(sketch_extrude())
        self.assertEqual(len(steps), 1)
        self.assertEqual(sequence_length(sketch_extrude()), 1)

    def test_two_steps(self):
        self.assertEqual(sequence_length(two_step()), 2)

    def test_ignores_terminators(self):
        padded = list(pad_sequence(sketch_extrude(), nc=16))
        self.assertEqual(sequence_length(padded), 1)

    def test_operation_ratio_sums_to_one(self):
        ratios = operation_ratio([sketch_extrude(), two_step()])
        self.assertAlmostEqual(sum(ratios.values()), 1.0, places=9)
        # 3 L + 1 E across first, 3 L + 1 E + 1 Cf across second -> 6 L / 9 total
        self.assertAlmostEqual(ratios["L"], 6 / 9, places=9)

    def test_length_distribution_buckets(self):
        dist = length_distribution([sketch_extrude(), two_step()])
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
        # both are length 1 and 2 -> both in 0-5 bucket
        self.assertAlmostEqual(dist["0-5"], 1.0, places=9)

    def test_truncate_last_step(self):
        cut = truncate_last_step(two_step())
        self.assertIsNotNone(cut)
        self.assertEqual(sequence_length(cut), 1)
        # single-step workflow cannot be truncated
        self.assertIsNone(truncate_last_step(sketch_extrude()))

    def test_augment_longest_first(self):
        # three steps -> two augmented prefixes (2 steps, then 1 step)
        three = two_step() + [command("F", px=1.0, py=1.0, pz=1.0)]
        variants = augment(three)
        self.assertEqual(len(variants), 2)
        self.assertEqual(sequence_length(variants[0]), 2)
        self.assertEqual(sequence_length(variants[1]), 1)

    def test_augment_max_variants(self):
        three = two_step() + [command("F", px=1.0, py=1.0, pz=1.0)]
        self.assertEqual(len(augment(three, max_variants=1)), 1)

    def test_empty_corpus(self):
        self.assertEqual(operation_ratio([]), {})
        self.assertTrue(all(v == 0.0 for v in length_distribution([]).values()))


if __name__ == "__main__":
    unittest.main()
