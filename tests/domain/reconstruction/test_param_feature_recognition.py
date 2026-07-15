"""Tests for the IntentForge parameter-aware feature recognizer."""

import unittest

from harnesscad.domain.reconstruction import param_feature_recognition as pfr


def _cyl(axis, cx, cy, radius, length):
    """A cylinder along `axis` centred at (cx, cy) with given radius/length."""
    half = length / 2.0
    if axis == "z":
        bbox = {
            "xmin": cx - radius, "xmax": cx + radius,
            "ymin": cy - radius, "ymax": cy + radius,
            "zmin": -half, "zmax": half,
        }
    else:  # x
        bbox = {
            "xmin": -half, "xmax": half,
            "ymin": cy - radius, "ymax": cy + radius,
            "zmin": -radius, "zmax": radius,
        }
    return pfr.Face(kind="cylinder", bbox=bbox, axis=axis, radius=radius)


class ThroughHoleTest(unittest.TestCase):
    def test_matches_expected_centers_and_count(self):
        centers = [(-20, -10), (20, -10), (20, 10), (-20, 10)]
        faces = [_cyl("z", cx, cy, 2.5, 3.0) for cx, cy in centers]
        res = pfr.recognize_through_holes(
            faces, "z", expected_count=4, expected_diameter=5.0,
            expected_centers=centers, through_length=3.0,
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.recognized_count, 4)
        self.assertEqual(res.confidence, pfr.CONFIDENCE_HIGH)

    def test_wrong_diameter_rejected(self):
        faces = [_cyl("z", 0, 0, 10.0, 3.0)]  # dia 20, expected 5
        res = pfr.recognize_through_holes(
            faces, "z", expected_count=1, expected_diameter=5.0, through_length=3.0
        )
        self.assertFalse(res.passed)
        self.assertEqual(res.recognized_count, 0)
        self.assertEqual(res.confidence, pfr.CONFIDENCE_LOW)

    def test_count_mismatch_warns(self):
        faces = [_cyl("z", 0, 0, 2.5, 3.0)]
        res = pfr.recognize_through_holes(faces, "z", expected_count=2, expected_diameter=5.0)
        self.assertFalse(res.passed)
        self.assertTrue(res.warnings)


class RoundedCornerTest(unittest.TestCase):
    def test_four_corner_cylinders_recognized(self):
        w, h, t, r = 80, 40, 3, 2.0
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        faces = [_cyl("z", cx, cy, r, t) for cx, cy in corners]
        res = pfr.recognize_rounded_corners(
            faces, corner_radius=r, plate_width=w, plate_height=h, plate_thickness=t
        )
        self.assertTrue(res.passed)
        self.assertEqual(res.recognized_count, 4)

    def test_center_hole_not_counted_as_corner(self):
        w, h, t, r = 80, 40, 3, 2.0
        faces = [_cyl("z", 0, 0, r, t)]  # centred -> not a corner
        res = pfr.recognize_rounded_corners(
            faces, corner_radius=r, plate_width=w, plate_height=h, plate_thickness=t
        )
        self.assertFalse(res.passed)


class CenterCutoutTest(unittest.TestCase):
    def test_four_walls_recognized(self):
        cw, ch, t = 20.0, 10.0, 3.0
        # two x-walls at +/-cw/2 (thin in x), two y-walls at +/-ch/2 (thin in y)
        faces = [
            pfr.Face("plane", {"xmin": cw/2, "xmax": cw/2, "ymin": -ch/2, "ymax": ch/2, "zmin": 0, "zmax": t}),
            pfr.Face("plane", {"xmin": -cw/2, "xmax": -cw/2, "ymin": -ch/2, "ymax": ch/2, "zmin": 0, "zmax": t}),
            pfr.Face("plane", {"xmin": -cw/2, "xmax": cw/2, "ymin": ch/2, "ymax": ch/2, "zmin": 0, "zmax": t}),
            pfr.Face("plane", {"xmin": -cw/2, "xmax": cw/2, "ymin": -ch/2, "ymax": -ch/2, "zmin": 0, "zmax": t}),
        ]
        res = pfr.recognize_center_cutout(faces, cutout_width=cw, cutout_height=ch, plate_thickness=t)
        self.assertTrue(res.passed)

    def test_no_walls_low_confidence(self):
        res = pfr.recognize_center_cutout([], cutout_width=20, cutout_height=10, plate_thickness=3)
        self.assertFalse(res.passed)
        self.assertEqual(res.confidence, pfr.CONFIDENCE_LOW)


if __name__ == "__main__":
    unittest.main()
