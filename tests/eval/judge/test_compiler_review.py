import unittest

from harnesscad.eval.judge.compiler_review import (
    ReviewResult,
    feedback_message,
    review_sequence,
)


def _sketch(points=None):
    if points is None:
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
    return {"type": "sketch", "loops": [{"points": points}]}


def _extrude(depth=1.0, boolean="union"):
    return {"type": "extrude", "depth": depth, "boolean": boolean}


class CompilerReviewTests(unittest.TestCase):
    def test_valid_sketch_extrude_passes(self):
        seq = [_sketch(), _extrude(), {"type": "end"}]
        r = review_sequence(seq)
        self.assertTrue(r.ok)
        self.assertEqual(feedback_message(r), "")

    def test_empty_sequence_is_format_error(self):
        r = review_sequence([])
        self.assertFalse(r.ok)
        self.assertEqual(r.category, "format")

    def test_missing_end_terminator(self):
        r = review_sequence([_sketch(), _extrude()])
        self.assertEqual(r.category, "format")
        self.assertIn("end", r.reason)

    def test_unknown_opcode(self):
        r = review_sequence([{"type": "warp"}, {"type": "end"}])
        self.assertEqual(r.category, "format")
        self.assertEqual(r.op_index, 0)

    def test_sketch_with_no_loop_is_geometry(self):
        r = review_sequence([{"type": "sketch", "loops": []}, {"type": "end"}])
        self.assertEqual(r.category, "geometry")

    def test_degenerate_collinear_loop_is_geometry(self):
        seq = [_sketch([[0, 0], [1, 0], [2, 0]]), _extrude(), {"type": "end"}]
        r = review_sequence(seq)
        self.assertEqual(r.category, "geometry")
        self.assertIn("degenerate", r.reason)

    def test_too_few_vertices(self):
        seq = [_sketch([[0, 0], [1, 0]]), _extrude(), {"type": "end"}]
        r = review_sequence(seq)
        self.assertEqual(r.category, "geometry")

    def test_circle_loop_ok_and_bad_radius(self):
        ok = review_sequence([{"type": "sketch", "loops": [{"radius": 2.0}]}, _extrude(), {"type": "end"}])
        self.assertTrue(ok.ok)
        bad = review_sequence([{"type": "sketch", "loops": [{"radius": 0.0}]}, _extrude(), {"type": "end"}])
        self.assertEqual(bad.category, "geometry")

    def test_extrude_without_sketch(self):
        r = review_sequence([_extrude(), {"type": "end"}])
        self.assertEqual(r.category, "extrusion")

    def test_zero_depth_extrude(self):
        r = review_sequence([_sketch(), _extrude(depth=0.0), {"type": "end"}])
        self.assertEqual(r.category, "extrusion")

    def test_cut_without_base_is_boolean(self):
        r = review_sequence([_sketch(), _extrude(boolean="cut"), {"type": "end"}])
        self.assertEqual(r.category, "boolean")

    def test_second_extrude_cut_ok(self):
        seq = [_sketch(), _extrude(), _sketch(), _extrude(boolean="cut"), {"type": "end"}]
        self.assertTrue(review_sequence(seq).ok)

    def test_no_solid_produced(self):
        r = review_sequence([_sketch(), {"type": "end"}])
        self.assertEqual(r.category, "geometry")
        self.assertIn("empty solid", r.reason)

    def test_feedback_message_is_rendered_for_each_category(self):
        for seq, cat in (
            ([], "format"),
            ([{"type": "sketch", "loops": []}, {"type": "end"}], "geometry"),
            ([_extrude(), {"type": "end"}], "extrusion"),
            ([_sketch(), _extrude(boolean="cut"), {"type": "end"}], "boolean"),
        ):
            r = review_sequence(seq)
            msg = feedback_message(r)
            self.assertTrue(msg)
            self.assertEqual(r.category, cat)

    def test_determinism(self):
        seq = [_sketch(), _extrude(), {"type": "end"}]
        self.assertEqual(review_sequence(seq), review_sequence(seq))

    def test_non_list_input(self):
        r = review_sequence("not a list")
        self.assertEqual(r.category, "format")


if __name__ == "__main__":
    unittest.main()
