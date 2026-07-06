import math
import unittest

from vision.cvcad_pixel_calibration import (
    Calibration,
    bounding_box,
    bounding_box_pixel_size,
    calibrate_from_reference,
    calibrate_from_reference_contour,
    contour_to_metric,
    distance_metric,
    measure_object_size,
    metric_to_pixels,
    pixels_to_metric,
    point_to_metric,
)


class TestCalibration(unittest.TestCase):
    def test_pixels_per_metric_ratio(self):
        # Reference sticker 100 px wide, known 20 mm -> 5 px/mm.
        cal = calibrate_from_reference(100.0, 20.0)
        self.assertAlmostEqual(cal.pixels_per_metric, 5.0)
        self.assertAlmostEqual(cal.mm_per_pixel, 0.2)

    def test_invalid_reference(self):
        with self.assertRaises(ValueError):
            calibrate_from_reference(0.0, 20.0)
        with self.assertRaises(ValueError):
            calibrate_from_reference(100.0, 0.0)

    def test_pixels_to_metric_roundtrip(self):
        cal = calibrate_from_reference(50.0, 10.0)  # 5 px/mm
        self.assertAlmostEqual(pixels_to_metric(cal, 150.0), 30.0)
        self.assertAlmostEqual(metric_to_pixels(cal, 30.0), 150.0)

    def test_point_and_contour_to_metric(self):
        cal = calibrate_from_reference(10.0, 1.0)  # 10 px/mm
        self.assertEqual(point_to_metric(cal, (100.0, 50.0)), (10.0, 5.0))
        pts = contour_to_metric(cal, [(0.0, 0.0), (20.0, 40.0)])
        self.assertEqual(pts, [(0.0, 0.0), (2.0, 4.0)])

    def test_distance_metric(self):
        cal = calibrate_from_reference(4.0, 1.0)  # 4 px/mm
        # 3-4-5 triangle: 20 px hypotenuse -> 5 mm.
        self.assertAlmostEqual(distance_metric(cal, (0.0, 0.0), (12.0, 16.0)),
                               5.0)

    def test_bounding_box(self):
        pts = [(1.0, 2.0), (5.0, 2.0), (5.0, 9.0), (1.0, 9.0)]
        self.assertEqual(bounding_box(pts), (1.0, 2.0, 5.0, 9.0))
        self.assertEqual(bounding_box_pixel_size(pts), (4.0, 7.0))

    def test_bounding_box_empty(self):
        with self.assertRaises(ValueError):
            bounding_box([])

    def test_measure_object_size(self):
        cal = calibrate_from_reference(2.0, 1.0)  # 2 px/mm
        pts = [(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)]
        w, h = measure_object_size(cal, pts)
        self.assertAlmostEqual(w, 30.0)
        self.assertAlmostEqual(h, 20.0)

    def test_calibrate_from_reference_contour(self):
        # Reference circle bbox width 40 px, known diameter 8 mm -> 5 px/mm.
        circle = [(10.0, 30.0), (50.0, 30.0), (30.0, 10.0), (30.0, 50.0)]
        cal = calibrate_from_reference_contour(circle, 8.0)
        self.assertAlmostEqual(cal.pixels_per_metric, 5.0)

    def test_paper_percentage_case(self):
        # Reproduce the calibrated-measurement magnitude around the cube (29.8mm).
        cal = calibrate_from_reference(100.0, 10.0)  # 10 px/mm
        self.assertAlmostEqual(pixels_to_metric(cal, 298.0), 29.8)


if __name__ == "__main__":
    unittest.main()
