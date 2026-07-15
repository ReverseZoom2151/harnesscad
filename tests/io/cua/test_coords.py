"""Decode model coordinates; refuse to guess the space; expose the drift bug."""

import unittest

from harnesscad.io.cua.coords import (
    CoordError, CoordSpace, Downscale, denormalize, detect_nonuniform_scale,
    geometric_mean_error, midpoint, parse_numbers, parse_point,
)


class TestParsing(unittest.TestCase):
    def test_box_tag_is_unwrapped(self):
        txt = "here <|box_start|>100, 200, 300, 400<|box_end|> ok"
        self.assertEqual(parse_numbers(txt), [100.0, 200.0, 300.0, 400.0])

    def test_bbox_midpoint(self):
        self.assertEqual(midpoint([100, 200, 300, 400]), (200.0, 300.0))

    def test_two_number_point(self):
        self.assertEqual(midpoint([12.5, 34.0]), (12.5, 34.0))

    def test_bad_count_raises(self):
        with self.assertRaises(CoordError):
            midpoint([1.0])
        with self.assertRaises(CoordError):
            midpoint([1.0, 2.0, 3.0])


class TestNoMagnitudeGuessing(unittest.TestCase):
    def test_unit_space(self):
        self.assertEqual(denormalize(0.5, 0.5, CoordSpace.UNIT, 1000, 800),
                         (500.0, 400.0))

    def test_thousand_space(self):
        self.assertEqual(denormalize(500, 250, CoordSpace.THOUSAND, 1000, 800),
                         (500.0, 200.0))

    def test_out_of_declared_range_raises_not_reinterprets(self):
        # 500 is fine as THOUSAND but is NOT reinterpreted when declared UNIT.
        with self.assertRaises(CoordError):
            denormalize(500, 0.5, CoordSpace.UNIT, 1000, 800)

    def test_pixel_outside_image_raises(self):
        with self.assertRaises(CoordError):
            denormalize(2000, 10, CoordSpace.PIXELS, 1000, 800)

    def test_parse_point_end_to_end(self):
        txt = "<|box_start|>250, 250, 750, 750<|box_end|>"
        self.assertEqual(parse_point(txt, CoordSpace.THOUSAND, 1000, 800),
                         (500, 400))


class TestDownscaleAndDrift(unittest.TestCase):
    def test_uniform_scale_has_no_warning(self):
        self.assertIsNone(detect_nonuniform_scale(2560, 1440, 1280, 720))

    def test_fill_resize_is_flagged(self):
        """16:9 squashed into 16:10 -> x and y scales differ."""
        msg = detect_nonuniform_scale(2560, 1440, 1280, 800)
        self.assertIsNotNone(msg)
        self.assertIn("non-uniform", msg)

    def test_per_axis_round_trip_is_exact(self):
        ds = Downscale(2560, 1440, 1280, 800)
        img = ds.to_image(1000, 700)
        back = ds.to_source(*img)
        self.assertAlmostEqual(back[0], 1000)
        self.assertAlmostEqual(back[1], 700)

    def test_geometric_mean_mapping_drifts_off_centre(self):
        """PROVE the reference sqrt(sx*sy) bug: zero error only at the origin,
        growing with distance, and never zero off-axis when scales differ."""
        ex, ey = geometric_mean_error(2560, 1440, 1280, 800, 0, 0)
        self.assertAlmostEqual(ex, 0.0)
        self.assertAlmostEqual(ey, 0.0)
        fx, fy = geometric_mean_error(2560, 1440, 1280, 800, 2000, 1200)
        self.assertGreater(abs(fx) + abs(fy), 1.0)

    def test_uniform_scale_has_zero_geometric_mean_error(self):
        fx, fy = geometric_mean_error(2560, 1440, 1280, 720, 2000, 1200)
        self.assertAlmostEqual(fx, 0.0)
        self.assertAlmostEqual(fy, 0.0)

    def test_degenerate_downscale_raises(self):
        with self.assertRaises(CoordError):
            Downscale(0, 100, 100, 100)


if __name__ == "__main__":
    unittest.main()
