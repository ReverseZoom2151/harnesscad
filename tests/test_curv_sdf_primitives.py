"""Tests for geometry.curv_sdf_primitives.

Verifies the defining properties of a signed distance field:

* sign correctness (negative strictly inside, positive strictly outside, ~0 on
  the boundary);
* for *exact* primitives, ``|f(p)|`` equals the true Euclidean distance to the
  boundary at sampled points, computed independently;
* the 1-Lipschitz / Eikonal property ``|grad f| <= 1 + eps`` via central
  differences (mitred and exact fields are both 1-Lipschitz).
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import curv_sdf_primitives as P


def grad_mag(f, p, h=1e-5):
    """Central-difference gradient magnitude of a 3D field ``f`` at ``p``."""
    g = []
    for i in range(len(p)):
        pp = list(p)
        pm = list(p)
        pp[i] += h
        pm[i] -= h
        g.append((f(pp) - f(pm)) / (2.0 * h))
    return math.sqrt(sum(c * c for c in g))


class TestSphere(unittest.TestCase):
    def test_exact_distance(self):
        # sphere diameter 4 -> radius 2
        for p, expected in [
            ((0.0, 0.0, 0.0), -2.0),
            ((2.0, 0.0, 0.0), 0.0),
            ((5.0, 0.0, 0.0), 3.0),
            ((3.0, 4.0, 0.0), 3.0),  # mag 5 - 2
        ]:
            self.assertAlmostEqual(P.sphere(p, 4.0), expected, places=9)

    def test_lipschitz(self):
        for p in [(1.0, 2.0, 3.0), (0.3, -0.2, 0.1), (5.0, 5.0, 5.0)]:
            self.assertLessEqual(grad_mag(lambda q: P.sphere(q, 4.0), p), 1.0 + 1e-6)


class TestBox(unittest.TestCase):
    def test_exact_vs_euclidean(self):
        size = (2.0, 2.0, 2.0)  # half-size 1

        def euclid(p):
            # true signed distance to axis-aligned box, half-size 1
            dx = abs(p[0]) - 1.0
            dy = abs(p[1]) - 1.0
            dz = abs(p[2]) - 1.0
            outside = math.sqrt(max(dx, 0) ** 2 + max(dy, 0) ** 2 + max(dz, 0) ** 2)
            inside = min(max(dx, dy, dz), 0.0)
            return outside + inside

        for p in [(2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (2.0, 2.0, 2.0),
                  (0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (3.0, 0.2, -0.1)]:
            self.assertAlmostEqual(P.box_exact(p, size), euclid(p), places=9)

    def test_exact_corner_distance(self):
        # corner at (1,1,1); point (2,2,2) is sqrt(3) away
        self.assertAlmostEqual(P.box_exact((2.0, 2.0, 2.0), (2.0, 2.0, 2.0)),
                               math.sqrt(3.0), places=9)

    def test_mitred_underestimates_but_agrees_on_faces(self):
        size = (2.0, 2.0, 2.0)
        # on a face normal direction the two agree
        self.assertAlmostEqual(P.box_mitred((2.0, 0.0, 0.0), size),
                               P.box_exact((2.0, 0.0, 0.0), size), places=9)
        # near a corner mitred is a strict underestimate of exact
        self.assertLess(P.box_mitred((2.0, 2.0, 2.0), size),
                        P.box_exact((2.0, 2.0, 2.0), size))

    def test_sign(self):
        size = (2.0, 4.0, 6.0)
        self.assertLess(P.box_exact((0.0, 0.0, 0.0), size), 0.0)
        self.assertGreater(P.box_exact((10.0, 0.0, 0.0), size), 0.0)

    def test_lipschitz(self):
        size = (2.0, 3.0, 1.5)
        for p in [(2.0, 1.0, 0.5), (0.1, 0.2, 0.3), (4.0, 4.0, 4.0)]:
            self.assertLessEqual(grad_mag(lambda q: P.box_exact(q, size), p), 1.0 + 1e-6)
            self.assertLessEqual(grad_mag(lambda q: P.box_mitred(q, size), p), 1.0 + 1e-6)


class TestRoundedBox(unittest.TestCase):
    def test_offset_relation(self):
        size = (2.0, 2.0, 2.0)
        p = (3.0, 0.0, 0.0)
        self.assertAlmostEqual(P.rounded_box(p, size, 0.5),
                               P.box_exact(p, size) - 0.5, places=12)


class TestCylinder(unittest.TestCase):
    def test_exact_distance(self):
        # radius 1, height 2 (hh=1)
        self.assertAlmostEqual(P.cylinder((3.0, 0.0, 0.0), 2.0, 2.0), 2.0, places=9)
        self.assertAlmostEqual(P.cylinder((0.0, 0.0, 3.0), 2.0, 2.0), 2.0, places=9)
        self.assertLess(P.cylinder((0.0, 0.0, 0.0), 2.0, 2.0), 0.0)
        # top rim corner (1,0,1) -> point (2,0,2) is sqrt(2) away
        self.assertAlmostEqual(P.cylinder((2.0, 0.0, 2.0), 2.0, 2.0),
                               math.sqrt(2.0), places=9)

    def test_lipschitz(self):
        for p in [(1.5, 0.0, 0.5), (0.2, 0.3, 2.0), (3.0, 3.0, 3.0)]:
            self.assertLessEqual(grad_mag(lambda q: P.cylinder(q, 2.0, 2.0), p), 1.0 + 1e-6)


class TestCone(unittest.TestCase):
    def test_sign_and_apex(self):
        # base diameter 2 (radius 1), height 2, apex at (0,0,2)
        self.assertLess(P.cone((0.0, 0.0, 1.0), 2.0, 2.0), 0.0)   # interior
        self.assertGreater(P.cone((0.0, 0.0, 3.0), 2.0, 2.0), 0.0)  # above apex
        self.assertGreater(P.cone((2.0, 0.0, 0.0), 2.0, 2.0), 0.0)  # outside base
        # straight above the apex by 1 unit
        self.assertAlmostEqual(P.cone((0.0, 0.0, 3.0), 2.0, 2.0), 1.0, places=9)

    def test_lipschitz(self):
        for p in [(0.5, 0.0, 0.5), (1.5, 0.0, 0.5), (0.0, 0.0, 3.0), (2.0, 2.0, 1.0)]:
            self.assertLessEqual(grad_mag(lambda q: P.cone(q, 2.0, 2.0), p), 1.0 + 1e-6)


class TestCappedCone(unittest.TestCase):
    def test_reduces_to_cylinder_when_equal_caps(self):
        # top==bottom==2 (radius 1), height 2 -> same shape as cylinder
        for p in [(3.0, 0.0, 0.0), (0.0, 0.0, 2.0), (0.5, 0.0, 0.0),
                  (2.0, 0.0, 1.0)]:
            self.assertAlmostEqual(P.capped_cone(p, 2.0, 2.0, 2.0),
                                   P.cylinder(p, 2.0, 2.0), places=9)

    def test_sign(self):
        self.assertLess(P.capped_cone((0.0, 0.0, 0.0), 2.0, 1.0, 3.0), 0.0)


class TestCapsule(unittest.TestCase):
    def test_exact_distance(self):
        a = (-1.0, 0.0, 0.0)
        b = (1.0, 0.0, 0.0)
        # diameter 1 -> r 0.5. point above midpoint at height 2 -> 2 - 0.5
        self.assertAlmostEqual(P.capsule((0.0, 2.0, 0.0), a, b, 1.0), 1.5, places=9)
        # beyond the cap: point (3,0,0) -> dist to b is 2, minus r
        self.assertAlmostEqual(P.capsule((3.0, 0.0, 0.0), a, b, 1.0), 1.5, places=9)
        self.assertLess(P.capsule((0.0, 0.0, 0.0), a, b, 1.0), 0.0)

    def test_lipschitz(self):
        a, b = (-1.0, 0.0, 0.0), (1.0, 0.5, 0.2)
        for p in [(0.0, 2.0, 0.0), (3.0, 0.0, 0.0), (0.2, 0.2, 0.2)]:
            self.assertLessEqual(grad_mag(lambda q: P.capsule(q, a, b, 1.0), p), 1.0 + 1e-6)


class TestTorus(unittest.TestCase):
    def test_exact_distance(self):
        # major diameter 4 (R=2), minor diameter 2 (r=1)
        # point on the +x outer equator at radius 4 -> 4 - 2 - 1 = 1
        self.assertAlmostEqual(P.torus((4.0, 0.0, 0.0), 4.0, 2.0), 1.0, places=9)
        # centre of the hole
        self.assertAlmostEqual(P.torus((0.0, 0.0, 0.0), 4.0, 2.0), 1.0, places=9)
        # on the tube centreline surface
        self.assertLess(P.torus((2.0, 0.0, 0.5), 4.0, 2.0), 0.0)

    def test_lipschitz(self):
        for p in [(4.0, 0.0, 0.0), (2.0, 0.0, 0.5), (1.0, 1.0, 1.0)]:
            self.assertLessEqual(grad_mag(lambda q: P.torus(q, 4.0, 2.0), p), 1.0 + 1e-6)


class TestEllipsoid(unittest.TestCase):
    def test_sign_and_bound(self):
        size = (4.0, 2.0, 2.0)  # semi-axes 2,1,1
        self.assertLess(P.ellipsoid((0.0, 0.0, 0.0), size), 0.0)
        self.assertAlmostEqual(P.ellipsoid((2.0, 0.0, 0.0), size), 0.0, places=9)
        self.assertAlmostEqual(P.ellipsoid((0.0, 1.0, 0.0), size), 0.0, places=9)
        self.assertGreater(P.ellipsoid((3.0, 0.0, 0.0), size), 0.0)

    def test_lipschitz_bound(self):
        size = (4.0, 2.0, 2.0)
        for p in [(3.0, 0.0, 0.0), (0.0, 2.0, 0.0), (1.0, 0.5, 0.5)]:
            self.assertLessEqual(grad_mag(lambda q: P.ellipsoid(q, size), p), 1.0 + 1e-6)


class TestPlane(unittest.TestCase):
    def test_exact(self):
        n = (0.0, 0.0, 1.0)
        self.assertAlmostEqual(P.plane((5.0, 3.0, 2.0), n, 0.0), 2.0, places=9)
        self.assertAlmostEqual(P.plane((0.0, 0.0, -1.0), n, 0.0), -1.0, places=9)

    def test_lipschitz(self):
        n = (0.6, 0.0, 0.8)  # unit
        self.assertAlmostEqual(grad_mag(lambda q: P.plane(q, n, 1.0), (1.0, 1.0, 1.0)),
                               1.0, places=5)


class Test2D(unittest.TestCase):
    def test_circle(self):
        self.assertAlmostEqual(P.circle((3.0, 4.0), 2.0), 4.0, places=9)  # mag5 - 1
        self.assertLess(P.circle((0.0, 0.0), 2.0), 0.0)

    def test_rect_exact(self):
        # half-size (1,1), corner distance
        self.assertAlmostEqual(P.rect_exact((2.0, 2.0), (2.0, 2.0)),
                               math.sqrt(2.0), places=9)
        self.assertLess(P.rect_exact((0.0, 0.0), (2.0, 2.0)), 0.0)

    def test_rect_mitred_underestimates(self):
        self.assertLess(P.rect_mitred((2.0, 2.0), (2.0, 2.0)),
                        P.rect_exact((2.0, 2.0), (2.0, 2.0)))

    def test_half_plane(self):
        n = (1.0, 0.0)
        self.assertAlmostEqual(P.half_plane((3.0, 5.0), n, 1.0), 2.0, places=9)

    def test_regular_polygon_apothem(self):
        # incircle diameter 2 -> apothem 1. bottom edge parallel to X at y=-1.
        # the bottom edge midpoint (0,-1) is on the boundary
        self.assertAlmostEqual(P.regular_polygon((0.0, -1.0), 6, 2.0), 0.0, places=9)
        # centre is inside
        self.assertLess(P.regular_polygon((0.0, 0.0), 6, 2.0), 0.0)
        # far away is outside
        self.assertGreater(P.regular_polygon((5.0, 0.0), 6, 2.0), 0.0)

    def test_regular_polygon_lipschitz(self):
        def f(q):
            return P.regular_polygon((q[0], q[1]), 5, 2.0)
        for p in [(0.5, 0.5), (2.0, 0.0), (0.1, -0.9)]:
            g = []
            for i in range(2):
                pp, pm = list(p), list(p)
                pp[i] += 1e-5
                pm[i] -= 1e-5
                g.append((f(pp) - f(pm)) / 2e-5)
            self.assertLessEqual(math.hypot(*g), 1.0 + 1e-5)


class TestLifts(unittest.TestCase):
    def test_extrude_matches_cylinder(self):
        # extruding a circle radius 1 to height 2 == cylinder d2 h2
        for p in [(3.0, 0.0, 0.0), (0.0, 0.0, 3.0), (2.0, 0.0, 2.0),
                  (0.0, 0.0, 0.0)]:
            d2 = P.circle((p[0], p[1]), 2.0)
            self.assertAlmostEqual(P.extrude(d2, p[2], 2.0),
                                   P.cylinder(p, 2.0, 2.0), places=9)

    def test_revolve_makes_torus(self):
        # revolve a circle of diameter 2 (r=1) centred at radius 2 -> torus
        def cross(x, y):
            return P.circle((x - 2.0, y), 2.0)

        for p in [(4.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.5)]:
            self.assertAlmostEqual(P.revolve(cross, p),
                                   P.torus(p, 4.0, 2.0), places=9)


if __name__ == "__main__":
    unittest.main()
