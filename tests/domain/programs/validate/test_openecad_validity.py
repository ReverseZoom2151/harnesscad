"""Tests for OpenECAD loop-closure and profile validity."""

import unittest

from harnesscad.domain.programs.ast import openecad as oe
from harnesscad.domain.programs.validate import openecad_validity as val


def line(a, b):
    return oe.Call(oe.ADD_LINE, (oe.Arg(list(a), "start"), oe.Arg(list(b), "end")))


def arc(a, b, mid):
    return oe.Call(oe.ADD_ARC, (
        oe.Arg(list(a), "start"), oe.Arg(list(b), "end"), oe.Arg(list(mid), "mid")))


def circle(c, r):
    return oe.Call(oe.ADD_CIRCLE, (oe.Arg(list(c), "center"), oe.Arg(r, "radius")))


# A closed unit square (counter-clockwise).
SQUARE = [
    line((0.0, 0.0), (1.0, 0.0)),
    line((1.0, 0.0), (1.0, 1.0)),
    line((1.0, 1.0), (0.0, 1.0)),
    line((0.0, 1.0), (0.0, 0.0)),
]


class TestEndpoints(unittest.TestCase):
    def test_line_endpoints(self):
        self.assertEqual(
            val.curve_endpoints(line((0, 0), (2, 3))), ((0, 0), (2, 3)))

    def test_circle_has_none(self):
        self.assertIsNone(val.curve_endpoints(circle((0, 0), 5)))


class TestClosure(unittest.TestCase):
    def test_square_is_closed(self):
        self.assertTrue(val.is_closed_loop(SQUARE))

    def test_open_loop_not_closed(self):
        broken = SQUARE[:-1]  # missing the closing edge
        self.assertFalse(val.is_closed_loop(broken))

    def test_gap_detected(self):
        loop = list(SQUARE)
        loop[-1] = line((0.0, 1.0), (0.1, 0.0))  # does not return to origin
        self.assertFalse(val.is_closed_loop(loop))
        self.assertGreater(max(val.loop_gaps(loop)), 0.05)

    def test_single_circle_closed(self):
        self.assertTrue(val.is_closed_loop([circle((0, 0), 5)]))

    def test_circle_plus_line_invalid(self):
        self.assertFalse(val.is_closed_loop([circle((0, 0), 5), line((0, 0), (1, 0))]))

    def test_empty_not_closed(self):
        self.assertFalse(val.is_closed_loop([]))

    def test_arc_closes_loop(self):
        loop = [
            line((-1.0, 0.0), (1.0, 0.0)),
            arc((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)),
        ]
        self.assertTrue(val.is_closed_loop(loop))

    def test_tolerance(self):
        loop = list(SQUARE)
        loop[-1] = line((0.0, 1.0), (1e-9, 0.0))
        self.assertTrue(val.is_closed_loop(loop, tol=1e-6))


class TestProfile(unittest.TestCase):
    def test_valid_profile(self):
        self.assertTrue(val.profile_is_valid([SQUARE, [circle((0, 0), 1)]]))

    def test_empty_profile_invalid(self):
        self.assertFalse(val.profile_is_valid([]))

    def test_one_bad_loop_invalidates(self):
        self.assertFalse(val.profile_is_valid([SQUARE, SQUARE[:-1]]))


class TestProgramGrouping(unittest.TestCase):
    def _prog(self):
        code = "\n".join([
            "SketchPlane0 = add_sketchplane(origin=[0.0, 0.0, 0.0], "
            "normal=[0.0, 0.0, 1.0], x_axis=[1.0, 0.0, 0.0])",
            "Loops0, Curves0_0 = [], []",
            "Line0_0_0 = add_line(start=[0.0, 0.0], end=[1.0, 0.0])",
            "Line0_0_1 = add_line(start=[1.0, 0.0], end=[1.0, 1.0])",
            "Line0_0_2 = add_line(start=[1.0, 1.0], end=[0.0, 1.0])",
            "Line0_0_3 = add_line(start=[0.0, 1.0], end=[0.0, 0.0])",
        ])
        return oe.parse(code)

    def test_grouping_orders_curves(self):
        loops = val.loops_from_program(self._prog())
        self.assertEqual(len(loops), 1)
        (key, curves), = loops.items()
        self.assertEqual((key.step, key.loop), (0, 0))
        self.assertEqual(len(curves), 4)

    def test_program_profile_valid(self):
        self.assertTrue(val.program_profiles_valid(self._prog()))


if __name__ == "__main__":
    unittest.main()
