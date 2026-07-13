"""Tests for numeric.curv_sphere_trace.

Verifies that sphere tracing recovers the analytic first-hit ``t`` for a sphere,
box and plane; that misses return ``None``; and that the central-difference
normal matches the analytic surface normal.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import curv_sdf_primitives as P
from harnesscad.domain.numeric import curv_sphere_trace as R


class TestSphereTrace(unittest.TestCase):
    def test_hit_sphere_analytic_t(self):
        # sphere radius 1 at origin; ray from (-5,0,0) along +x hits at x=-1
        field = lambda p: P.sphere(p, 2.0)
        d = R.ray_direction((1.0, 0.0, 0.0))
        t = R.sphere_trace(field, (-5.0, 0.0, 0.0), d, epsilon=1e-7)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 4.0, places=5)  # from x=-5 to x=-1

    def test_hit_offset_sphere(self):
        # sphere radius 2 centred at (0,0,10); ray from origin along +z
        field = lambda p: P.sphere((p[0], p[1], p[2] - 10.0), 4.0)
        d = R.ray_direction((0.0, 0.0, 1.0))
        t = R.sphere_trace(field, (0.0, 0.0, 0.0), d, epsilon=1e-7)
        self.assertAlmostEqual(t, 8.0, places=5)  # hits front face at z=8

    def test_hit_box(self):
        field = lambda p: P.box_exact(p, (2.0, 2.0, 2.0))  # half-size 1
        d = R.ray_direction((1.0, 0.0, 0.0))
        t = R.sphere_trace(field, (-5.0, 0.0, 0.0), d, epsilon=1e-7)
        self.assertAlmostEqual(t, 4.0, places=5)  # face at x=-1

    def test_hit_plane(self):
        # plane z=0, normal +z; ray from (0,0,5) downward hits at t=5
        field = lambda p: P.plane(p, (0.0, 0.0, 1.0), 0.0)
        d = R.ray_direction((0.0, 0.0, -1.0))
        t = R.sphere_trace(field, (0.0, 0.0, 5.0), d, epsilon=1e-7)
        self.assertAlmostEqual(t, 5.0, places=5)

    def test_miss_returns_none(self):
        # sphere radius 1 at origin; ray parallel but offset by 5 in y misses
        field = lambda p: P.sphere(p, 2.0)
        d = R.ray_direction((1.0, 0.0, 0.0))
        t = R.sphere_trace(field, (-5.0, 5.0, 0.0), d, max_dist=100.0)
        self.assertIsNone(t)

    def test_lipschitz_scaled_field_still_hits(self):
        # a field scaled by 2 (over-estimating gradient) traces safely with
        # lipschitz=2 and still lands on the analytic surface
        field = lambda p: 2.0 * P.sphere(p, 2.0)
        d = R.ray_direction((1.0, 0.0, 0.0))
        t = R.sphere_trace(field, (-5.0, 0.0, 0.0), d, epsilon=1e-6, lipschitz=2.0)
        self.assertAlmostEqual(t, 4.0, places=4)


class TestNormal(unittest.TestCase):
    def test_sphere_normal(self):
        field = lambda p: P.sphere(p, 2.0)
        # on the +x surface the outward normal is +x
        n = R.estimate_normal(field, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(n[0], 1.0, places=5)
        self.assertAlmostEqual(n[1], 0.0, places=5)
        self.assertAlmostEqual(n[2], 0.0, places=5)

    def test_diagonal_normal(self):
        field = lambda p: P.sphere(p, 2.0)
        s = 1.0 / math.sqrt(3.0)
        n = R.estimate_normal(field, (s, s, s))
        for c in n:
            self.assertAlmostEqual(c, s, places=4)

    def test_plane_normal(self):
        field = lambda p: P.plane(p, (0.0, 0.0, 1.0), 0.0)
        n = R.estimate_normal(field, (3.0, 2.0, 0.0))
        self.assertAlmostEqual(n[2], 1.0, places=6)

    def test_unit_length(self):
        field = lambda p: P.box_exact(p, (2.0, 3.0, 1.0))
        n = R.estimate_normal(field, (1.0, 0.4, 0.2))
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
