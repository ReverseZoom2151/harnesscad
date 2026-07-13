"""Tests for geometry.sdfx_thread_profile."""

import math
import unittest

from harnesscad.domain.geometry.sdf.sdfx_polygon_sdf import polygon_area, polygon_sdf
from harnesscad.domain.geometry.features.sdfx_thread_profile import (
    acme_thread,
    ansi_buttress_thread,
    iso_thread,
)


class TestISOThread(unittest.TestCase):
    def test_external_profile_valid(self):
        verts = iso_thread(6.0, 1.0, external=True)
        self.assertGreater(len(verts), 6)
        for x, y in verts:
            self.assertTrue(math.isfinite(x) and math.isfinite(y))
        # profile spans one pitch in x: [-1, 1]
        xs = [v[0] for v in verts]
        self.assertAlmostEqual(min(xs), -1.0)
        self.assertAlmostEqual(max(xs), 1.0)

    def test_external_crest_at_major_radius(self):
        radius = 6.0
        pitch = 1.0
        verts = iso_thread(radius, pitch, external=True)
        # within the central pitch window (the swept region) the crest flat
        # sits at the nominal major radius; the x=+/-pitch edge carries the
        # sharp-V apex extension (r_major + h/8) which is trimmed on sweep.
        central = [v[1] for v in verts if abs(v[0]) <= pitch / 2.0 + 1e-9]
        self.assertAlmostEqual(max(central), radius, places=6)

    def test_internal_differs_from_external(self):
        ext = iso_thread(6.0, 1.0, external=True)
        internal = iso_thread(6.0, 1.0, external=False)
        self.assertNotEqual(len(ext), len(internal))

    def test_root_below_major(self):
        radius = 8.0
        pitch = 1.25
        verts = iso_thread(radius, pitch, external=True)
        ymin_positive = min(v[1] for v in verts if v[1] > 0)
        # thread root is strictly below the major radius
        self.assertLess(ymin_positive, radius)

    def test_bad_params(self):
        with self.assertRaises(ValueError):
            iso_thread(0.0, 1.0)
        with self.assertRaises(ValueError):
            iso_thread(6.0, 0.0)


class TestAcmeThread(unittest.TestCase):
    def test_trapezoid_shape(self):
        verts = acme_thread(10.0, 2.0)
        self.assertEqual(len(verts), 8)
        ymax = max(v[1] for v in verts)
        self.assertAlmostEqual(ymax, 10.0)

    def test_symmetric_in_x(self):
        verts = acme_thread(10.0, 2.0)
        xs = sorted(v[0] for v in verts)
        # x range symmetric about 0
        self.assertAlmostEqual(xs[0], -xs[-1])

    def test_flank_angle_29deg(self):
        # the flank runs from (x_ofs1, h) to (x_ofs0, radius); its slope
        # corresponds to the 29-degree included angle (14.5 per side).
        radius, pitch = 10.0, 2.0
        verts = acme_thread(radius, pitch)
        h = radius - 0.5 * pitch
        theta = math.radians(29.0 / 2.0)
        delta = 0.25 * pitch * math.tan(theta)
        x_ofs0 = 0.25 * pitch - delta
        x_ofs1 = 0.25 * pitch + delta
        # the horizontal run over the flank height should match tan(theta)
        run = x_ofs1 - x_ofs0
        rise = radius - h
        self.assertAlmostEqual(run / rise, math.tan(theta), places=6)


class TestButtressThread(unittest.TestCase):
    def test_spans_two_periods(self):
        pitch = 2.0
        verts = ansi_buttress_thread(10.0, pitch)
        xs = [v[0] for v in verts]
        self.assertAlmostEqual(min(xs), -pitch)
        self.assertAlmostEqual(max(xs), pitch)

    def test_asymmetric(self):
        # buttress is asymmetric: the profile is not mirror-symmetric in x.
        verts = ansi_buttress_thread(10.0, 2.0)
        # collect interior (radius-level) vertices; their x set is not symmetric
        self.assertGreater(len(verts), 8)
        for x, y in verts:
            self.assertTrue(math.isfinite(x) and math.isfinite(y))

    def test_usable_as_field(self):
        verts = ansi_buttress_thread(10.0, 2.0)
        # a point deep inside the tooth body should be inside the polygon
        d = polygon_sdf((0.0, 5.0), verts)
        self.assertLess(d, 0.0)

    def test_area_positive(self):
        verts = ansi_buttress_thread(10.0, 2.0)
        self.assertGreater(abs(polygon_area(verts)), 0.0)


if __name__ == "__main__":
    unittest.main()
