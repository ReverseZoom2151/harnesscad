"""Tests for geometry.nurbgen_hybrid_primitives (analytic fallback + CD)."""

import math
import unittest

from harnesscad.domain.geometry.parametric import nurbgen_hybrid_primitives as hp


class TestLine(unittest.TestCase):
    def test_line_endpoints_and_count(self):
        pts = hp.sample_line((0.0, 0.0, 0.0), (3.0, 0.0, 4.0), samples=8)
        self.assertEqual(len(pts), 9)
        self.assertEqual(pts[0], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(pts[-1][0], 3.0)
        self.assertAlmostEqual(pts[-1][2], 4.0)
        # Midpoint lies on the segment.
        self.assertAlmostEqual(pts[4][0], 1.5)


class TestCircle(unittest.TestCase):
    def test_points_on_circle_in_xy_plane(self):
        pts = hp.sample_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0,
                               samples=40)
        for x, y, z in pts:
            self.assertAlmostEqual(math.hypot(x, y), 2.0, places=12)
            self.assertAlmostEqual(z, 0.0, places=12)

    def test_semicircle_arc(self):
        pts = hp.sample_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0,
                               first=0.0, last=math.pi, samples=2)
        # start (1,0,0), middle (0,1,0), end (-1,0,0)
        self.assertAlmostEqual(pts[0][0], 1.0, places=12)
        self.assertAlmostEqual(pts[1][1], 1.0, places=12)
        self.assertAlmostEqual(pts[2][0], -1.0, places=12)

    def test_circle_in_tilted_plane_radius_preserved(self):
        normal = (1.0, 1.0, 1.0)
        center = (5.0, -2.0, 3.0)
        pts = hp.sample_circle(center, normal, 1.5, samples=24)
        for p in pts:
            d = math.dist(p, center)
            self.assertAlmostEqual(d, 1.5, places=10)
            # Point offset is perpendicular to the normal.
            off = (p[0] - center[0], p[1] - center[1], p[2] - center[2])
            self.assertAlmostEqual(hp._dot(off, hp._normalize(normal)), 0.0,
                                   places=10)


class TestEllipse(unittest.TestCase):
    def test_ellipse_axes(self):
        pts = hp.sample_ellipse((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 3.0, 1.0,
                                samples=4)
        # t=0 -> major axis end; t=pi/2 -> minor axis end.
        self.assertAlmostEqual(pts[0][0], 3.0, places=12)
        self.assertAlmostEqual(pts[1][1], 1.0, places=12)


class TestDispatcher(unittest.TestCase):
    def test_sample_primitive_line(self):
        spec = {"type": "line", "start": [0, 0, 0], "end": [1, 1, 1]}
        pts = hp.sample_primitive(spec, samples=4)
        self.assertEqual(len(pts), 5)

    def test_sample_primitive_circle(self):
        spec = {"type": "circle", "center": [0, 0, 0], "normal": [0, 0, 1],
                "radius": 1.0}
        pts = hp.sample_primitive(spec, samples=8)
        self.assertAlmostEqual(math.hypot(pts[0][0], pts[0][1]), 1.0)

    def test_unknown_type_rejected(self):
        with self.assertRaises(ValueError):
            hp.sample_primitive({"type": "hyperbola"})


class TestChamferDistance(unittest.TestCase):
    def test_identical_clouds_zero(self):
        a = hp.sample_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 8)
        self.assertAlmostEqual(hp.chamfer_distance(a, a), 0.0, places=15)

    def test_symmetric(self):
        a = hp.sample_line((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5)
        b = hp.sample_line((0.0, 0.1, 0.0), (1.0, 0.1, 0.0), 5)
        self.assertAlmostEqual(hp.chamfer_distance(a, b),
                               hp.chamfer_distance(b, a), places=15)

    def test_shifted_cloud_matches_squared_offset(self):
        a = hp.sample_line((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 100)
        # Shift perpendicular by 0.01; nearest distance ~= 0.01 -> CD ~= 1e-4.
        b = [(x, y + 0.01, z) for (x, y, z) in a]
        cd = hp.chamfer_distance(a, b)
        self.assertAlmostEqual(cd, 0.01 ** 2, places=6)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            hp.chamfer_distance([], [(0.0, 0.0, 0.0)])


class TestHybridGate(unittest.TestCase):
    def test_close_reconstruction_keeps_nurbs(self):
        gt = hp.sample_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, samples=64)
        recon = [(x + 1e-4, y, z) for (x, y, z) in gt]  # tiny error
        self.assertTrue(hp.accept_nurbs(recon, gt))
        self.assertEqual(hp.choose_representation(recon, gt), "nurbs")

    def test_far_reconstruction_falls_back(self):
        gt = hp.sample_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0, samples=64)
        recon = [(x + 0.1, y, z) for (x, y, z) in gt]  # large error
        self.assertFalse(hp.accept_nurbs(recon, gt))
        self.assertEqual(hp.choose_representation(recon, gt), "analytic")

    def test_hybrid_stats_fraction(self):
        decisions = ["nurbs"] * 7 + ["analytic"] * 3
        s = hp.hybrid_stats(decisions)
        self.assertEqual(s["n_nurbs"], 7)
        self.assertEqual(s["n_analytic"], 3)
        self.assertAlmostEqual(s["nurbs_fraction"], 0.7)

    def test_hybrid_stats_rejects_bad_label(self):
        with self.assertRaises(ValueError):
            hp.hybrid_stats(["nurbs", "spline"])


if __name__ == "__main__":
    unittest.main()
