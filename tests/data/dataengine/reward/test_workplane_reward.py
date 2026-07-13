import math
import unittest

from harnesscad.data.dataengine.reward import workplane_reward as r


class TestFormatReward(unittest.TestCase):
    def test_valid_format(self):
        text = "<think>reasoning here</think>\n```python\nresult = cq.Workplane()\n```"
        self.assertEqual(r.format_reward(text), 1.0)

    def test_missing_code_block(self):
        self.assertEqual(r.format_reward("<think>only reasoning</think>"), 0.0)

    def test_code_before_reasoning_fails(self):
        text = "```python\ncode\n```\n<think>reasoning</think>"
        self.assertEqual(r.format_reward(text), 0.0)

    def test_empty(self):
        self.assertEqual(r.format_reward(""), 0.0)
        self.assertEqual(r.format_reward(None), 0.0)


class TestExecAndIoU(unittest.TestCase):
    def test_exec(self):
        self.assertEqual(r.exec_reward(True), 1.0)
        self.assertEqual(r.exec_reward(False), 0.0)

    def test_iou_bounds(self):
        self.assertEqual(r.iou_reward(0.5), 0.5)
        with self.assertRaises(ValueError):
            r.iou_reward(1.5)

    def test_jaccard(self):
        self.assertAlmostEqual(r.jaccard_iou(2.0, 8.0), 0.25)
        self.assertEqual(r.jaccard_iou(0.0, 0.0), 1.0)

    def test_jaccard_invalid(self):
        with self.assertRaises(ValueError):
            r.jaccard_iou(5.0, 2.0)
        with self.assertRaises(ValueError):
            r.jaccard_iou(-1.0, 2.0)


class TestWorkPlaneReward(unittest.TestCase):
    def test_perfect_alignment(self):
        rp = r.work_plane_reward([0, 0, 0], [0, 0, 0],
                                 [1, 0, 0], [1, 0, 0],
                                 [0, 1, 0], [0, 1, 0])
        self.assertEqual(rp, 1.0)

    def test_origin_deviation(self):
        d = r.origin_deviation([3, 0, 0], [0, 4, 0])
        self.assertAlmostEqual(d, 5.0)

    def test_axis_deviation_aligned(self):
        d = r.axis_deviation([1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0])
        self.assertAlmostEqual(d, 0.0)

    def test_axis_deviation_anti_parallel(self):
        d = r.axis_deviation([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0])
        self.assertAlmostEqual(d, 2.0)

    def test_axis_deviation_orthogonal(self):
        # x rotated 90 deg (cos=0), y rotated 90 deg (cos=0) => 0.5*(2-0-0)=1.0
        d = r.axis_deviation([1, 0, 0], [0, 1, 0], [0, 1, 0], [-1, 0, 0])
        self.assertAlmostEqual(d, 1.0)

    def test_reward_clamped_low(self):
        rp = r.work_plane_reward([100, 0, 0], [0, 0, 0],
                                 [1, 0, 0], [1, 0, 0],
                                 [0, 1, 0], [0, 1, 0])
        self.assertEqual(rp, 0.0)

    def test_zero_vector_raises(self):
        with self.assertRaises(ValueError):
            r.axis_deviation([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0])


class TestTotalReward(unittest.TestCase):
    def test_gating_blocks_when_format_fails(self):
        self.assertEqual(r.total_reward(0.0, 1.0, 1.0, 1.0), 0.0)

    def test_gating_blocks_when_exec_fails(self):
        self.assertEqual(r.total_reward(1.0, 0.0, 1.0, 1.0), 0.0)

    def test_positive_when_both_gates_pass(self):
        val = r.total_reward(1.0, 1.0, 0.8, 0.5,
                             lambda_iou=1.0, lambda_plane=1.0)
        self.assertAlmostEqual(val, 1.3)

    def test_weights(self):
        val = r.total_reward(1.0, 1.0, 1.0, 1.0, lambda_iou=2.0, lambda_plane=3.0)
        self.assertAlmostEqual(val, 5.0)


class TestRewardComponents(unittest.TestCase):
    def test_full_positive(self):
        text = "<think>x</think>\n```python\ncode\n```"
        comp = r.reward_components(
            text, True, 0.9,
            origin_gen=[0, 0, 0], origin_gt=[0, 0, 0],
            x_gen=[1, 0, 0], x_gt=[1, 0, 0],
            y_gen=[0, 1, 0], y_gt=[0, 1, 0])
        self.assertEqual(comp["r_format"], 1.0)
        self.assertEqual(comp["r_exec"], 1.0)
        self.assertAlmostEqual(comp["r_iou"], 0.9)
        self.assertAlmostEqual(comp["r_plane"], 1.0)
        self.assertAlmostEqual(comp["total"], 1.9)

    def test_non_executable_zeroes_geometry(self):
        text = "<think>x</think>\n```python\ncode\n```"
        comp = r.reward_components(text, False, 0.9)
        self.assertEqual(comp["r_exec"], 0.0)
        self.assertEqual(comp["r_iou"], 0.0)
        self.assertEqual(comp["r_plane"], 0.0)
        self.assertEqual(comp["total"], 0.0)

    def test_missing_pose_defaults_plane_zero(self):
        text = "<think>x</think>\n```python\ncode\n```"
        comp = r.reward_components(text, True, 0.5)
        self.assertEqual(comp["r_plane"], 0.0)
        self.assertAlmostEqual(comp["total"], 0.5)

    def test_determinism(self):
        text = "<think>x</think>\n```python\ncode\n```"
        a = r.reward_components(text, True, 0.7)
        b = r.reward_components(text, True, 0.7)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
