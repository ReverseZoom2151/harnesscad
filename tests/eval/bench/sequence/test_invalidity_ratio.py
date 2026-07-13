import unittest

from harnesscad.eval.bench.sequence import invalidity_ratio as ir


class TestCurveValidity(unittest.TestCase):
    def test_degenerate_line(self):
        self.assertTrue(ir.line_is_invalid({"start": (0.0, 0.0), "end": (0.0, 0.0)}))

    def test_valid_line(self):
        self.assertFalse(ir.line_is_invalid({"start": (0.0, 0.0), "end": (1.0, 0.0)}))

    def test_degenerate_arc(self):
        self.assertTrue(ir.arc_is_invalid({"start": (0.2, 0.2), "end": (0.2, 0.2)}))

    def test_valid_arc(self):
        self.assertFalse(ir.arc_is_invalid({"start": (0.0, 0.0), "end": (1.0, 1.0)}))

    def test_curve_dispatch(self):
        self.assertTrue(ir.curve_is_invalid({"type": "line", "start": (0, 0), "end": (0, 0)}))
        self.assertFalse(ir.curve_is_invalid({"type": "circle", "center": (0, 0), "radius": 1}))


class TestExtrusionValidity(unittest.TestCase):
    def test_zero_both_sides_invalid(self):
        self.assertTrue(ir.extrusion_is_invalid({"d_plus": 0.0, "d_minus": 0.0}))

    def test_one_side_nonzero_valid(self):
        self.assertFalse(ir.extrusion_is_invalid({"d_plus": 0.5, "d_minus": 0.0}))
        self.assertFalse(ir.extrusion_is_invalid({"d_plus": 0.0, "d_minus": 0.3}))


class TestSequenceValidity(unittest.TestCase):
    def _valid_seq(self):
        return {
            "curves": [{"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}],
            "extrusion": {"d_plus": 0.5, "d_minus": 0.0},
        }

    def _invalid_curve_seq(self):
        return {
            "curves": [{"type": "line", "start": (0.0, 0.0), "end": (0.0, 0.0)}],
            "extrusion": {"d_plus": 0.5, "d_minus": 0.0},
        }

    def _invalid_extr_seq(self):
        return {
            "curves": [{"type": "line", "start": (0.0, 0.0), "end": (1.0, 0.0)}],
            "extrusion": {"d_plus": 0.0, "d_minus": 0.0},
        }

    def test_valid_sequence(self):
        self.assertFalse(ir.sequence_is_invalid(self._valid_seq()))

    def test_invalid_curve_sequence(self):
        self.assertTrue(ir.sequence_is_invalid(self._invalid_curve_seq()))

    def test_invalid_extrusion_sequence(self):
        self.assertTrue(ir.sequence_is_invalid(self._invalid_extr_seq()))

    def test_report(self):
        rep = ir.inspect_sequence(self._invalid_curve_seq())
        self.assertEqual(rep.invalid_curves, 1)
        self.assertFalse(rep.invalid_extrusion)
        self.assertTrue(rep.is_invalid)

    def test_missing_extrusion_ok(self):
        seq = {"curves": [{"type": "line", "start": (0, 0), "end": (1, 1)}]}
        self.assertFalse(ir.sequence_is_invalid(seq))


class TestInvalidityRatio(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(ir.invalidity_ratio([]), 0.0)

    def test_ratio(self):
        seqs = [
            {"curves": [{"type": "line", "start": (0, 0), "end": (1, 0)}],
             "extrusion": {"d_plus": 1.0, "d_minus": 0.0}},
            {"curves": [{"type": "line", "start": (0, 0), "end": (0, 0)}],
             "extrusion": {"d_plus": 1.0, "d_minus": 0.0}},
            {"curves": [{"type": "line", "start": (0, 0), "end": (1, 0)}],
             "extrusion": {"d_plus": 0.0, "d_minus": 0.0}},
            {"curves": [{"type": "line", "start": (0, 0), "end": (1, 0)}],
             "extrusion": {"d_plus": 1.0, "d_minus": 0.0}},
        ]
        self.assertEqual(ir.invalidity_ratio(seqs), 0.5)
        self.assertEqual(ir.invalidity_percentage(seqs), 50.0)

    def test_one_percent_like(self):
        valid = {"curves": [{"type": "line", "start": (0, 0), "end": (1, 0)}],
                 "extrusion": {"d_plus": 1.0, "d_minus": 0.0}}
        invalid = {"curves": [{"type": "line", "start": (0, 0), "end": (0, 0)}],
                   "extrusion": {"d_plus": 1.0, "d_minus": 0.0}}
        seqs = [valid] * 99 + [invalid]
        self.assertAlmostEqual(ir.invalidity_percentage(seqs), 1.0)


if __name__ == "__main__":
    unittest.main()
