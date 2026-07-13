import unittest

from harnesscad.domain.reconstruction.s2cadsig_op_router import (
    ALL_BRANCH_OUTPUTS,
    NUM_OPERATIONS,
    OP_ADD_SUB,
    OP_BEVEL,
    OP_EXTRUSION,
    OP_SPECS,
    OP_SWEEP,
    OperationError,
    classification_accuracy,
    confusion_matrix,
    required_parameters,
    route,
    select_branch_outputs,
    softmax,
    spec_for,
)


class TestVocabulary(unittest.TestCase):
    def test_four_ops(self):
        self.assertEqual(len(OP_SPECS), NUM_OPERATIONS)
        self.assertEqual([s.op_id for s in OP_SPECS], [0, 1, 2, 3])

    def test_spec_lookup(self):
        self.assertEqual(spec_for(OP_BEVEL).name, "bevel")
        self.assertEqual(spec_for("sweep").op_id, OP_SWEEP)
        with self.assertRaises(OperationError):
            spec_for(9)
        with self.assertRaises(OperationError):
            spec_for("chamfer")
        with self.assertRaises(OperationError):
            spec_for(1.5)

    def test_guiding_curves(self):
        self.assertEqual(spec_for(OP_EXTRUSION).guiding_curve, "offset_curve")
        self.assertEqual(spec_for(OP_SWEEP).guiding_curve, "profile_curve")
        self.assertEqual(spec_for(OP_ADD_SUB).guiding_curve, "base_curve")

    def test_bevel_is_heatmap_head_without_offset(self):
        s = spec_for(OP_BEVEL)
        self.assertEqual(s.curve_head, "heatmap")
        self.assertFalse(s.needs_offset)
        self.assertEqual(s.volume_effect, "modify")

    def test_required_parameters(self):
        self.assertEqual(
            required_parameters(OP_EXTRUSION),
            (
                "stitching_face",
                "offset_curve",
                "offset_distance",
                "offset_direction",
                "offset_sign",
            ),
        )
        self.assertEqual(
            required_parameters("bevel"), ("stitching_face", "base_curve")
        )

    def test_all_outputs(self):
        self.assertIn("face_heatmap", ALL_BRANCH_OUTPUTS)
        self.assertEqual(len(ALL_BRANCH_OUTPUTS), 4)


class TestSoftmaxRoute(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        p = softmax([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(sum(p), 1.0, places=9)
        self.assertTrue(all(v > 0 for v in p))

    def test_softmax_shift_invariant(self):
        a = softmax([1.0, 2.0, 3.0])
        b = softmax([101.0, 102.0, 103.0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_softmax_empty(self):
        with self.assertRaises(OperationError):
            softmax([])

    def test_route_picks_argmax(self):
        r = route([0.1, 0.2, 5.0, 0.0])
        self.assertEqual(r.op_id, OP_BEVEL)
        self.assertEqual(r.op_name, "bevel")
        self.assertGreater(r.confidence, 0.9)
        self.assertGreater(r.margin, 0.0)
        self.assertTrue(r.accepted)

    def test_route_tie_breaks_low(self):
        r = route([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(r.op_id, OP_ADD_SUB)
        self.assertAlmostEqual(r.margin, 0.0, places=9)
        self.assertAlmostEqual(r.confidence, 0.25, places=9)

    def test_route_threshold_rejects(self):
        r = route([1.0, 1.0, 1.0, 1.0], threshold=0.5)
        self.assertFalse(r.accepted)

    def test_route_bad_length(self):
        with self.assertRaises(OperationError):
            route([1.0, 2.0])

    def test_used_and_ignored_outputs(self):
        r = route([0.0, 9.0, 0.0, 0.0])
        self.assertEqual(r.used_outputs, ("face_heatmap", "offset_curve"))
        self.assertNotIn("face_heatmap", r.ignored_outputs)
        self.assertIn("profile_curve", r.ignored_outputs)

    def test_select_branch_outputs(self):
        r = route([9.0, 0.0, 0.0, 0.0])
        combined = {
            "face_heatmap": [0.5],
            "base_curve": [1.0],
            "offset_curve": [2.0],
            "profile_curve": [3.0],
        }
        out = select_branch_outputs(r, combined)
        self.assertEqual(sorted(out), ["base_curve", "face_heatmap"])

    def test_select_branch_outputs_missing(self):
        r = route([0.0, 0.0, 0.0, 9.0])
        with self.assertRaises(OperationError):
            select_branch_outputs(r, {"face_heatmap": []})


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix(self):
        m = confusion_matrix([0, 1, 1, 3], [0, 1, 2, 3])
        self.assertEqual(m[0][0], 1)
        self.assertEqual(m[2][1], 1)
        self.assertEqual(m[3][3], 1)
        self.assertEqual(sum(sum(r) for r in m), 4)

    def test_confusion_matrix_errors(self):
        with self.assertRaises(OperationError):
            confusion_matrix([0], [0, 1])
        with self.assertRaises(OperationError):
            confusion_matrix([7], [0])

    def test_accuracy(self):
        self.assertAlmostEqual(classification_accuracy([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
        self.assertAlmostEqual(classification_accuracy([0, 0, 2, 3], [0, 1, 2, 3]), 0.75)
        self.assertEqual(classification_accuracy([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
