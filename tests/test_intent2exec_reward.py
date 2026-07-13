import unittest

from harnesscad.data.dataengine.reward.intent2exec_reward import (
    DEFAULT_FAILURE_PENALTY,
    PARAMETRIC_MISASSIGNMENT,
    REFERENCE_FRAME_MISALIGNMENT,
    normalize_score,
    r_eval,
    r_exec,
    r_geom,
    total_reward,
)


class TestExec(unittest.TestCase):
    def test_binary(self):
        self.assertEqual(r_exec(True), 1.0)
        self.assertEqual(r_exec(False), 0.0)


class TestGeom(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(r_geom(3.0, 6.0), 0.5)

    def test_empty_union(self):
        self.assertEqual(r_geom(0.0, 0.0), 0.0)

    def test_full_overlap(self):
        self.assertEqual(r_geom(4.0, 4.0), 1.0)

    def test_intersection_exceeds_union(self):
        with self.assertRaises(ValueError):
            r_geom(5.0, 4.0)

    def test_negative(self):
        with self.assertRaises(ValueError):
            r_geom(-1.0, 2.0)


class TestNormalize(unittest.TestCase):
    def test_midpoint(self):
        self.assertAlmostEqual(normalize_score(5.0, 0.0, 10.0), 0.5)

    def test_clamp(self):
        self.assertEqual(normalize_score(-3.0), 0.0)
        self.assertEqual(normalize_score(99.0), 1.0)

    def test_bad_range(self):
        with self.assertRaises(ValueError):
            normalize_score(1.0, 5.0, 5.0)


class TestEval(unittest.TestCase):
    def test_clean(self):
        self.assertAlmostEqual(r_eval(8.0), 0.8)

    def test_failure_deduction(self):
        base = r_eval(8.0)
        pen = r_eval(8.0, [REFERENCE_FRAME_MISALIGNMENT])
        self.assertAlmostEqual(pen, base - DEFAULT_FAILURE_PENALTY)

    def test_two_failures_clamped(self):
        val = r_eval(8.0, [REFERENCE_FRAME_MISALIGNMENT, PARAMETRIC_MISASSIGNMENT])
        self.assertEqual(val, 0.0)

    def test_unknown_failure(self):
        with self.assertRaises(KeyError):
            r_eval(8.0, ["mystery"])


class TestTotal(unittest.TestCase):
    def test_gate_zero_when_not_executable(self):
        self.assertEqual(total_reward(False, 1.0, 1.0), 0.0)

    def test_convex_combo(self):
        val = total_reward(True, 0.6, 0.4, lambda_geom=1.0, lambda_eval=1.0)
        self.assertAlmostEqual(val, 1.0)

    def test_weights(self):
        val = total_reward(True, 0.5, 0.5, lambda_geom=2.0, lambda_eval=0.0)
        self.assertAlmostEqual(val, 1.0)

    def test_negative_weight(self):
        with self.assertRaises(ValueError):
            total_reward(True, 0.5, 0.5, lambda_geom=-1.0)


if __name__ == "__main__":
    unittest.main()
