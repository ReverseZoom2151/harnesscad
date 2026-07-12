import math
import unittest

from geometry.gaudi_parametric_profile import (
    ParametricExprError,
    compile_expr,
    dedupe_points,
    ensure_ccw,
    evaluate,
    is_degenerate,
    polygon_signed_area,
    sample_curve,
)


class EvaluateTests(unittest.TestCase):
    def test_variable_and_arithmetic(self):
        self.assertAlmostEqual(evaluate("2 * t + 1", 3), 7.0)

    def test_bare_and_prefixed_math(self):
        self.assertAlmostEqual(evaluate("cos(t)", 0.0), 1.0)
        self.assertAlmostEqual(evaluate("math.cos(t)", 0.0), 1.0)

    def test_constants(self):
        self.assertAlmostEqual(evaluate("pi", 0.0), math.pi)
        self.assertAlmostEqual(evaluate("tau", 0.0), math.tau)

    def test_power_and_unary(self):
        self.assertAlmostEqual(evaluate("-t ** 2", 3), -9.0)

    def test_gaudi_wave_formula(self):
        # from app.py default script
        x = evaluate("4 * math.cos(t) + 2 * math.sin(2*t)", 0.0)
        y = evaluate("4 * math.sin(t) + 2 * math.cos(2*t)", 0.0)
        self.assertAlmostEqual(x, 4.0)
        self.assertAlmostEqual(y, 2.0)

    def test_atan2_two_args(self):
        self.assertAlmostEqual(evaluate("atan2(1, 1)", 0.0), math.pi / 4)


class SafetyTests(unittest.TestCase):
    def test_unknown_name_rejected(self):
        with self.assertRaises(ParametricExprError):
            evaluate("foo + t", 1.0)

    def test_attribute_outside_math_rejected(self):
        with self.assertRaises(ParametricExprError):
            evaluate("os.system", 0.0)

    def test_arbitrary_call_rejected(self):
        with self.assertRaises(ParametricExprError):
            evaluate("open('x')", 0.0)

    def test_subscript_rejected(self):
        with self.assertRaises(ParametricExprError):
            evaluate("t[0]", 0.0)

    def test_syntax_error(self):
        with self.assertRaises(ParametricExprError):
            evaluate("t +", 0.0)

    def test_compile_validates_once(self):
        with self.assertRaises(ParametricExprError):
            compile_expr("bogus(t)")


class SampleCurveTests(unittest.TestCase):
    def test_count_and_half_open(self):
        pts = sample_curve("cos(t)", "sin(t)", 0.0, 2 * math.pi, 8)
        self.assertEqual(len(pts), 8)
        # half-open: last point is NOT the closing repeat of the first
        self.assertFalse(
            math.isclose(pts[0][0], pts[-1][0]) and math.isclose(pts[0][1], pts[-1][1])
        )

    def test_deterministic(self):
        a = sample_curve("t", "t*t", 0.0, 1.0, 5)
        b = sample_curve("t", "t*t", 0.0, 1.0, 5)
        self.assertEqual(a, b)

    def test_unit_circle_on_axes(self):
        pts = sample_curve("cos(t)", "sin(t)", 0.0, 2 * math.pi, 4)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[0][1], 0.0)
        self.assertAlmostEqual(pts[1][0], 0.0, places=6)
        self.assertAlmostEqual(pts[1][1], 1.0, places=6)

    def test_bad_steps(self):
        with self.assertRaises(ParametricExprError):
            sample_curve("t", "t", 0.0, 1.0, 0)


class PolygonTests(unittest.TestCase):
    def _square_ccw(self):
        return [(0, 0), (2, 0), (2, 2), (0, 2)]

    def test_signed_area_ccw_positive(self):
        self.assertAlmostEqual(polygon_signed_area(self._square_ccw()), 4.0)

    def test_signed_area_cw_negative(self):
        self.assertAlmostEqual(
            polygon_signed_area(list(reversed(self._square_ccw()))), -4.0
        )

    def test_ensure_ccw_flips_cw(self):
        cw = list(reversed(self._square_ccw()))
        fixed = ensure_ccw(cw)
        self.assertGreater(polygon_signed_area(fixed), 0.0)

    def test_ensure_ccw_keeps_ccw(self):
        ccw = self._square_ccw()
        self.assertEqual(ensure_ccw(ccw), ccw)

    def test_dedupe_consecutive_and_closing(self):
        pts = [(0, 0), (0, 0), (1, 0), (1, 1), (0, 0)]
        out = dedupe_points(pts)
        self.assertEqual(out, [(0, 0), (1, 0), (1, 1)])

    def test_degenerate_collinear(self):
        self.assertTrue(is_degenerate([(0, 0), (1, 1), (2, 2)]))

    def test_degenerate_too_few(self):
        self.assertTrue(is_degenerate([(0, 0), (1, 0)]))

    def test_non_degenerate(self):
        self.assertFalse(is_degenerate(self._square_ccw()))


if __name__ == "__main__":
    unittest.main()
