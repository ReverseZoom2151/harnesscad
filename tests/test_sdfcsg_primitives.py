"""Tests for geometry.sdfcsg_primitives.

Verifies sign correctness (inside < 0, outside > 0, boundary ~ 0) and the
approximate 1-Lipschitz property for the additional SDF primitives ported from
sdf-csg.  Reference points are chosen analytically for each shape.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import sdfcsg_primitives as P


def grad_mag(f, p, h=1e-5):
    g = []
    for i in range(3):
        a = list(p)
        b = list(p)
        a[i] -= h
        b[i] += h
        g.append((f(b) - f(a)) / (2.0 * h))
    return math.sqrt(sum(c * c for c in g))


class TestBoxFrame(unittest.TestCase):
    def test_sign(self):
        f = lambda q: P.box_frame(q, (1.0, 1.0, 1.0), 0.1)
        # Just inside a corner strut (edge runs along z at x=y=1).
        self.assertLess(f((0.95, 0.95, 0.5)), 0.0)
        # Centre of a face: hollow -> outside.
        self.assertGreater(f((0.0, 0.0, 0.0)), 0.0)
        # Far away: outside and large.
        self.assertGreater(f((5.0, 5.0, 5.0)), 3.0)

    def test_lipschitz(self):
        f = lambda q: P.box_frame(q, (1.0, 1.0, 1.0), 0.1)
        for p in [(1.0, 1.0, 0.5), (0.5, 1.05, 1.0), (2.0, 0.0, 0.0)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-3)


class TestCappedTorus(unittest.TestCase):
    def test_full_torus_matches_ring(self):
        # angle == pi -> full torus of major R, minor r.
        R, r = 2.0, 0.5
        f = lambda q: P.capped_torus(q, math.pi, R, r)
        # On the tube centre ring (x=R, z=0): distance ~ -r.
        self.assertAlmostEqual(f((R, 0.0, 0.0)), -r, places=6)
        # On the surface of the tube.
        self.assertAlmostEqual(f((R + r, 0.0, 0.0)), 0.0, places=6)

    def test_sign(self):
        f = lambda q: P.capped_torus(q, math.pi / 2.0, 2.0, 0.5)
        self.assertLess(f((0.0, 2.0, 0.0)), 0.0)  # inside tube on the arc
        self.assertGreater(f((0.0, 0.0, 0.0)), 0.0)  # centre hole


class TestLink(unittest.TestCase):
    def test_sign_and_distance(self):
        le, R, r = 1.0, 1.0, 0.3
        f = lambda q: P.link(q, le, R, r)
        # Point on the straight-tube centre line (y within +/-le, x=R).
        self.assertAlmostEqual(f((R, 0.0, 0.0)), -r, places=6)
        self.assertAlmostEqual(f((R + r, 0.0, 0.0)), 0.0, places=6)
        self.assertGreater(f((0.0, 0.0, 0.0)), 0.0)

    def test_lipschitz(self):
        f = lambda q: P.link(q, 1.0, 1.0, 0.3)
        for p in [(1.4, 0.2, 0.1), (0.6, 1.5, 0.0), (2.0, 0.0, 0.5)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-3)


class TestHexagonalPrism(unittest.TestCase):
    def test_sign(self):
        f = lambda q: P.hexagonal_prism(q, 1.0, 2.0)
        self.assertLess(f((0.0, 0.0, 0.0)), 0.0)
        self.assertGreater(f((0.0, 0.0, 3.0)), 0.0)  # past the z cap
        self.assertGreater(f((3.0, 0.0, 0.0)), 0.0)  # outside radially

    def test_apothem_boundary(self):
        # Apothem direction is +y for this hexagon orientation.
        f = lambda q: P.hexagonal_prism(q, 1.0, 2.0)
        self.assertAlmostEqual(f((0.0, 1.0, 0.0)), 0.0, places=6)

    def test_lipschitz(self):
        f = lambda q: P.hexagonal_prism(q, 1.0, 2.0)
        for p in [(0.3, 0.4, 0.5), (0.0, 1.1, 0.0), (0.5, 0.0, 2.1)]:
            self.assertLessEqual(grad_mag(f, p), 1.0 + 1e-3)


class TestTriangularPrism(unittest.TestCase):
    def test_sign(self):
        f = lambda q: P.triangular_prism(q, 1.0, 2.0)
        self.assertLess(f((0.0, 0.0, 0.0)), 0.0)
        self.assertGreater(f((0.0, 0.0, 3.0)), 0.0)
        self.assertGreater(f((2.0, 0.0, 0.0)), 0.0)


class TestSolidAngle(unittest.TestCase):
    def test_sign(self):
        # Cone half-aperture 45deg about +y, sphere radius 1.
        f = lambda q: P.solid_angle(q, math.pi / 4.0, 1.0)
        # Inside the wedge, within the sphere, close to +y axis.
        self.assertLess(f((0.0, 0.5, 0.0)), 0.0)
        # Outside the sphere along +y.
        self.assertGreater(f((0.0, 2.0, 0.0)), 0.0)
        # Below the apex (opposite the aperture).
        self.assertGreater(f((0.0, -0.5, 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
