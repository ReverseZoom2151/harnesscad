"""Tests for the keep-in / keep-out interface-match axis."""
import unittest

from harnesscad.eval.bench.geometry.cgb_interface_match import (
    DEFAULT_N_SAMPLES,
    INTERFACE_FULL_SCORE_IOU,
    INTERFACE_ZERO_SCORE_IOU,
    SATURATION_THRESHOLD,
    ZERO_POSE,
    SubVolume,
    SubVolumeNameError,
    best_iou_in_context,
    discover_sub_volumes,
    evaluate_interface,
    feature_passes,
    group_sub_volumes,
    interface_score,
    iou_to_interface_score,
    parse_sub_volume,
    pose_grid,
    score_group,
)


class TestSubVolumeContract(unittest.TestCase):
    def test_parse(self):
        sv = parse_sub_volume("jig_0__2__KOR.step")
        self.assertEqual((sv.group, sv.index, sv.fit_type), (0, 2, "KOR"))
        self.assertEqual(sv.name, "jig_0__2__KOR")

    def test_parse_stp_extension(self):
        self.assertEqual(parse_sub_volume("jig_1__0__KIR.stp").fit_type, "KIR")

    def test_bad_name_raises(self):
        with self.assertRaises(SubVolumeNameError):
            parse_sub_volume("jig_0_1_KOR.step")
        with self.assertRaises(SubVolumeNameError):
            parse_sub_volume("jig_0__1__XYZ.step")

    def test_bad_fit_type_raises(self):
        with self.assertRaises(SubVolumeNameError):
            SubVolume(group=0, index=0, fit_type="MAYBE")

    def test_discover_ignores_other_files(self):
        names = [
            "ground_truth.step",
            "jig_1__0__KIR.step",
            "jig_0__1__KOR.step",
            "jig_0__0__KOR.step",
            "render.png",
        ]
        found = discover_sub_volumes(names)
        self.assertEqual(
            [sv.name for sv in found],
            ["jig_0__0__KOR", "jig_0__1__KOR", "jig_1__0__KIR"],
        )

    def test_grouping(self):
        found = discover_sub_volumes(
            ["jig_1__0__KIR.step", "jig_0__1__KOR.step", "jig_0__0__KOR.step"]
        )
        groups = group_sub_volumes(found)
        self.assertEqual(list(groups), [0, 1])
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(groups[1][0].fit_type, "KIR")


class TestRamp(unittest.TestCase):
    def test_full_pass(self):
        self.assertEqual(iou_to_interface_score(INTERFACE_FULL_SCORE_IOU), 1.0)
        self.assertEqual(iou_to_interface_score(1.0), 1.0)

    def test_clean_fail(self):
        self.assertEqual(iou_to_interface_score(INTERFACE_ZERO_SCORE_IOU), 0.0)
        self.assertEqual(iou_to_interface_score(0.0), 0.0)

    def test_linear_midpoint(self):
        mid = (INTERFACE_FULL_SCORE_IOU + INTERFACE_ZERO_SCORE_IOU) / 2
        self.assertAlmostEqual(iou_to_interface_score(mid), 0.5, places=9)

    def test_sloppy_fit_banks_little(self):
        self.assertLess(iou_to_interface_score(0.85), 0.5)

    def test_feature_passes(self):
        self.assertTrue(feature_passes(0.96))
        self.assertFalse(feature_passes(0.94))


class TestPoseSearch(unittest.TestCase):
    def test_identity_pose_first(self):
        poses = pose_grid(100.0)
        self.assertEqual(poses[0], ZERO_POSE)
        self.assertEqual(len(poses), DEFAULT_N_SAMPLES)

    def test_window_bounds(self):
        poses = pose_grid(200.0, n_samples=16, max_rotation_deg=1.0,
                          translation_fraction=0.01)
        for rx, ry, rz, tx, ty, tz in poses:
            for angle in (rx, ry, rz):
                self.assertLessEqual(abs(angle), 1.0)
            for offset in (tx, ty, tz):
                self.assertLessEqual(abs(offset), 2.0)  # 1% of 200

    def test_deterministic(self):
        self.assertEqual(pose_grid(50.0), pose_grid(50.0))

    def test_best_iou_early_exits_on_saturation(self):
        calls = []

        def iou_at_pose(pose):
            calls.append(pose)
            return 0.995  # saturates immediately

        best, pose, evaluated = best_iou_in_context(iou_at_pose, pose_grid(10.0))
        self.assertGreaterEqual(best, SATURATION_THRESHOLD)
        self.assertEqual(pose, ZERO_POSE)
        self.assertEqual(evaluated, 1)
        self.assertEqual(len(calls), 1)

    def test_best_iou_searches_the_window(self):
        poses = pose_grid(10.0, n_samples=8)
        target = poses[5]

        def iou_at_pose(pose):
            return 0.93 if pose == target else 0.5

        best, pose, evaluated = best_iou_in_context(iou_at_pose, poses)
        self.assertAlmostEqual(best, 0.93)
        self.assertEqual(pose, target)
        self.assertEqual(evaluated, 8)


class TestAggregation(unittest.TestCase):
    def test_group_scores_as_worst_feature(self):
        group = score_group(0, {"a": 0.99, "b": 0.99, "c": 0.10})
        self.assertEqual(group.score, 0.0)
        self.assertEqual(group.worst_feature, "c")

    def test_empty_group_raises(self):
        with self.assertRaises(ValueError):
            score_group(0, {})

    def test_mean_across_groups(self):
        result = interface_score({0: {"a": 0.99}, 1: {"b": 0.10}})
        self.assertAlmostEqual(result.score, 0.5)
        self.assertEqual(len(result.groups), 2)

    def test_no_groups_raises(self):
        with self.assertRaises(ValueError):
            interface_score({})

    def test_doc_example_offset_slot_zeroes_the_axis(self):
        # One mating group: two clean bolt holes plus a slot shifted off spec.
        result = interface_score({0: {"hole_a": 0.98, "hole_b": 0.97, "slot": 0.31}})
        self.assertEqual(result.score, 0.0)

    def test_to_dict_shape(self):
        payload = interface_score({0: {"a": 0.99}}).to_dict()
        self.assertEqual(payload["score"], 1.0)
        self.assertIn("0", payload["contexts"])
        self.assertIn("per_feature_iou", payload["contexts"]["0"])


class TestEndToEnd(unittest.TestCase):
    def test_evaluate_interface_with_stub_kernel(self):
        subs = discover_sub_volumes(
            ["jig_0__0__KOR.step", "jig_0__1__KIR.step", "jig_1__0__KOR.step"]
        )

        def iou_fn(sv, pose):
            # Group 0 fits at the authored pose; group 1's feature never fits.
            if sv.group == 0:
                return 0.995 if pose == ZERO_POSE else 0.9
            return 0.4

        result = evaluate_interface(subs, iou_fn, bbox_diagonal=100.0)
        self.assertAlmostEqual(result.score, 0.5)
        by_group = {g.group: g for g in result.groups}
        self.assertEqual(by_group[0].score, 1.0)
        self.assertEqual(by_group[1].score, 0.0)


if __name__ == "__main__":
    unittest.main()
