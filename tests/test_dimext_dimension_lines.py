import unittest

from harnesscad.domain.drawings.dimension_lines import (
    DimensionMeasurement,
    angle_between,
    detect_dimensions,
    measurements_to_metric,
)
from harnesscad.domain.vision.pixel_calibration import calibrate_from_reference


class TestAngle(unittest.TestCase):
    def test_perpendicular(self):
        h = (0.0, 0.0, 10.0, 0.0)
        v = (0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(angle_between(h, v), 90.0)

    def test_parallel(self):
        a = (0.0, 0.0, 10.0, 0.0)
        b = (0.0, 5.0, 10.0, 5.0)
        self.assertAlmostEqual(angle_between(a, b), 0.0)


class TestDetectDimensions(unittest.TestCase):
    def _horizontal_dim(self):
        # Dimension line from (0,0) to (100,0); extension lines rising at each end.
        dline = (0.0, 0.0, 100.0, 0.0)
        ext_a = (0.0, -5.0, 0.0, 20.0)
        ext_b = (100.0, -5.0, 100.0, 20.0)
        return [dline, ext_a, ext_b]

    def test_detect_single_dimension(self):
        dims = detect_dimensions(self._horizontal_dim())
        self.assertEqual(len(dims), 1)
        self.assertAlmostEqual(dims[0].length_pixels, 100.0)
        self.assertIsNone(dims[0].length_metric)

    def test_metric_via_calibration(self):
        cal = calibrate_from_reference(10.0, 1.0)  # 10 px/mm
        dims = detect_dimensions(self._horizontal_dim(), calibration=cal)
        self.assertAlmostEqual(dims[0].length_metric, 10.0)

    def test_no_extension_lines(self):
        # A lone dimension line with no perpendicular extension lines -> nothing.
        dims = detect_dimensions([(0.0, 0.0, 100.0, 0.0)])
        self.assertEqual(dims, [])

    def test_extension_must_touch_ends(self):
        dline = (0.0, 0.0, 100.0, 0.0)
        ext_a = (0.0, -5.0, 0.0, 20.0)
        far_ext = (60.0, -5.0, 60.0, 20.0)  # touches neither end within tol
        dims = detect_dimensions([dline, ext_a, far_ext])
        self.assertEqual(dims, [])

    def test_sorted_by_length(self):
        # Two dimensions; larger first.
        small = [(0.0, 0.0, 40.0, 0.0),
                 (0.0, -5.0, 0.0, 10.0),
                 (40.0, -5.0, 40.0, 10.0)]
        big = [(0.0, 50.0, 120.0, 50.0),
               (0.0, 45.0, 0.0, 65.0),
               (120.0, 45.0, 120.0, 65.0)]
        dims = detect_dimensions(small + big)
        self.assertEqual(len(dims), 2)
        self.assertAlmostEqual(dims[0].length_pixels, 120.0)
        self.assertAlmostEqual(dims[1].length_pixels, 40.0)

    def test_measurements_to_metric(self):
        cal = calibrate_from_reference(2.0, 1.0)  # 2 px/mm
        dims = detect_dimensions(self._horizontal_dim())
        conv = measurements_to_metric(dims, cal)
        self.assertAlmostEqual(conv[0].length_metric, 50.0)

    def test_vertical_dimension(self):
        dline = (5.0, 0.0, 5.0, 80.0)
        ext_a = (-10.0, 0.0, 15.0, 0.0)
        ext_b = (-10.0, 80.0, 15.0, 80.0)
        dims = detect_dimensions([dline, ext_a, ext_b])
        self.assertEqual(len(dims), 1)
        self.assertAlmostEqual(dims[0].length_pixels, 80.0)


if __name__ == "__main__":
    unittest.main()
