import unittest

from harnesscad.io.formats.step import parse
from harnesscad.eval.bench.geometry.step_file_metrics import (
    aec_gap, average_entity_count, center_align, centroid, chamfer_distance,
    completes, completion_rate, entity_count, geometric_reward,
    geometric_reward_for, median_scaled_chamfer_distance, rms_scale,
    scaled_chamfer_distance,
)


def _file(n_points):
    lines = ["ISO-10303-21;", "HEADER;", "ENDSEC;", "DATA;"]
    for i in range(1, n_points + 1):
        lines.append(f"#{i}=CARTESIAN_POINT('',(0.,0.,0.));")
    lines += ["ENDSEC;", "END-ISO-10303-21;", ""]
    return "\n".join(lines)


class TestCompletion(unittest.TestCase):
    def test_complete(self):
        self.assertTrue(completes(_file(1)))

    def test_incomplete(self):
        self.assertFalse(completes("ISO-10303-21;\nDATA;\n#1=PLANE("))

    def test_trailing_whitespace_ok(self):
        self.assertTrue(completes("END-ISO-10303-21;\n\n  "))

    def test_rate(self):
        self.assertAlmostEqual(
            completion_rate([_file(1), "broken", _file(2)]), 2 / 3)

    def test_rate_empty(self):
        self.assertEqual(completion_rate([]), 0.0)


class TestEntityCount(unittest.TestCase):
    def test_count(self):
        self.assertEqual(entity_count(parse(_file(5))), 5)

    def test_average(self):
        steps = [parse(_file(2)), parse(_file(4))]
        self.assertEqual(average_entity_count(steps), 3.0)

    def test_aec_gap(self):
        gen = [parse(_file(10))]
        gt = [parse(_file(6))]
        self.assertEqual(aec_gap(gen, gt), 4.0)


class TestPointCloud(unittest.TestCase):
    def test_centroid(self):
        self.assertEqual(centroid([(0, 0, 0), (2, 2, 2)]), (1, 1, 1))

    def test_center_align(self):
        out = center_align([(0, 0, 0), (2, 0, 0)])
        self.assertEqual(out, [(-1, 0, 0), (1, 0, 0)])

    def test_rms_scale(self):
        # points at +/-1 along x: rms = 1
        self.assertAlmostEqual(rms_scale([(-1, 0, 0), (1, 0, 0)]), 1.0)

    def test_chamfer_identical_is_zero(self):
        cloud = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertAlmostEqual(chamfer_distance(cloud, cloud), 0.0)

    def test_chamfer_symmetric(self):
        a = [(0, 0, 0), (1, 0, 0)]
        b = [(0, 0, 0), (2, 0, 0)]
        self.assertAlmostEqual(chamfer_distance(a, b), chamfer_distance(b, a))

    def test_chamfer_empty_raises(self):
        with self.assertRaises(ValueError):
            chamfer_distance([], [(0, 0, 0)])


class TestScaledChamfer(unittest.TestCase):
    def test_translation_invariant(self):
        gt = [(-1, 0, 0), (1, 0, 0), (0, 1, 0)]
        pred = [(p[0] + 10, p[1] + 5, p[2] - 3) for p in gt]
        self.assertAlmostEqual(scaled_chamfer_distance(pred, gt), 0.0)

    def test_scale_normalized(self):
        gt = [(-1, 0, 0), (1, 0, 0)]
        # identical shape -> zero regardless of normalization
        self.assertAlmostEqual(scaled_chamfer_distance(gt, gt), 0.0)

    def test_positive_for_different_shapes(self):
        gt = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0)]
        pred = [(-1, 0, 0), (1, 0, 0), (0, 2, 0), (0, -2, 0)]
        self.assertGreater(scaled_chamfer_distance(pred, gt), 0.0)

    def test_zero_scale_raises(self):
        with self.assertRaises(ValueError):
            scaled_chamfer_distance([(0, 0, 0)], [(0, 0, 0)])

    def test_median(self):
        gt = [(-1, 0, 0), (1, 0, 0)]
        pairs = [(gt, gt), (gt, gt), (gt, gt)]
        self.assertEqual(median_scaled_chamfer_distance(pairs), 0.0)


class TestReward(unittest.TestCase):
    def test_below_lower_bound(self):
        self.assertEqual(geometric_reward(0.005), 1.0)

    def test_above_upper_bound(self):
        self.assertEqual(geometric_reward(0.9), 0.0)

    def test_interpolated_midpoint(self):
        # midpoint of [0.01, 0.5] -> reward 0.5
        mid = (0.01 + 0.5) / 2
        self.assertAlmostEqual(geometric_reward(mid), 0.5)

    def test_monotonic_decreasing(self):
        a = geometric_reward(0.1)
        b = geometric_reward(0.2)
        self.assertGreater(a, b)

    def test_bad_thresholds(self):
        with self.assertRaises(ValueError):
            geometric_reward(0.1, delta_low=0.5, delta_high=0.5)

    def test_reward_for_identical_is_one(self):
        gt = [(-1, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertEqual(geometric_reward_for(gt, gt), 1.0)


if __name__ == "__main__":
    unittest.main()
