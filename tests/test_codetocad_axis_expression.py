import unittest

from geometry.codetocad_axis_expression import (
    AxisExpressionError,
    is_relative,
    resolve_axis_value,
    resolve_point,
    resolve_relative_size,
)
from geometry.codetocad_cardinal_landmark import BoundaryAxis, BoundaryBox


class TestIsRelative(unittest.TestCase):
    def test_keywords(self):
        self.assertTrue(is_relative("min + 2mm"))
        self.assertTrue(is_relative("MAX"))
        self.assertTrue(is_relative("center"))
        self.assertTrue(is_relative("50%"))

    def test_absolute(self):
        self.assertFalse(is_relative("5mm"))
        self.assertFalse(is_relative(5))
        self.assertFalse(is_relative(0.5))


class TestResolveAxisValue(unittest.TestCase):
    def setUp(self):
        self.axis = BoundaryAxis(0.0, 0.1)  # 100mm long

    def test_keywords(self):
        self.assertAlmostEqual(resolve_axis_value(self.axis, "min"), 0.0)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "max"), 0.1)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "center"), 0.05)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "CENTER"), 0.05)

    def test_keyword_offsets(self):
        self.assertAlmostEqual(resolve_axis_value(self.axis, "min + 2mm"), 0.002)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "min+2mm"), 0.002)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "max - 1cm"), 0.09)
        self.assertAlmostEqual(
            resolve_axis_value(self.axis, "center + 5mm"), 0.055
        )

    def test_keyword_percent_offset(self):
        self.assertAlmostEqual(resolve_axis_value(self.axis, "max - 10%"), 0.09)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "min + 25%"), 0.025)

    def test_proportional(self):
        self.assertAlmostEqual(resolve_axis_value(self.axis, "50%"), 0.05)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "0%"), 0.0)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "100%"), 0.1)

    def test_proportional_on_shifted_axis(self):
        axis = BoundaryAxis(1.0, 3.0)
        self.assertAlmostEqual(resolve_axis_value(axis, "50%"), 2.0)
        self.assertAlmostEqual(resolve_axis_value(axis, "min + 50%"), 2.0)

    def test_absolute(self):
        self.assertAlmostEqual(resolve_axis_value(self.axis, "5mm"), 0.005)
        self.assertAlmostEqual(resolve_axis_value(self.axis, 0.02), 0.02)
        self.assertAlmostEqual(resolve_axis_value(self.axis, "1in + 1mm"), 0.0264)

    def test_errors(self):
        with self.assertRaises(AxisExpressionError):
            resolve_axis_value(self.axis, "min 2mm")
        with self.assertRaises(AxisExpressionError):
            resolve_axis_value(self.axis, "")
        with self.assertRaises(AxisExpressionError):
            resolve_axis_value(self.axis, None)
        with self.assertRaises(AxisExpressionError):
            resolve_axis_value(self.axis, "min + 90deg")
        with self.assertRaises(AxisExpressionError):
            resolve_axis_value(self.axis, "90deg")


class TestResolvePoint(unittest.TestCase):
    def test_mixed_expressions(self):
        box = BoundaryBox.from_extents((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
        point = resolve_point(box, x="min + 5mm", y="center", z="max")
        self.assertAlmostEqual(point[0], -0.05 + 0.005)
        self.assertAlmostEqual(point[1], 0.0)
        self.assertAlmostEqual(point[2], 0.15)

    def test_defaults(self):
        box = BoundaryBox.from_extents((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
        self.assertEqual(resolve_point(box), (0.0, 0.0, 0.0))

    def test_all_proportional(self):
        box = BoundaryBox.from_extents((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        self.assertEqual(
            resolve_point(box, x="0%", y="50%", z="100%"), (-1.0, 0.0, 1.0)
        )


class TestResolveRelativeSize(unittest.TestCase):
    def test_percent(self):
        self.assertAlmostEqual(resolve_relative_size("50%", base=0.08), 0.04)
        self.assertAlmostEqual(resolve_relative_size("200%", base=0.01), 0.02)

    def test_absolute(self):
        self.assertAlmostEqual(resolve_relative_size("2mm"), 0.002)
        self.assertAlmostEqual(resolve_relative_size(0.5), 0.5)

    def test_percent_without_base(self):
        with self.assertRaises(AxisExpressionError):
            resolve_relative_size("50%")

    def test_mixed(self):
        self.assertAlmostEqual(
            resolve_relative_size("50% + 1mm", base=0.1), 0.051
        )


if __name__ == "__main__":
    unittest.main()
