"""Tests for geometry.sdf.sweep (ImplicitCAD ExtrudeM twist/taper linear extrude)."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import sweep as W
from harnesscad.domain.geometry.sdf import primitives as P


def square2d(x, y):
    """2D exact square field, full side 2 (half-extent 1)."""
    return P.rect_exact((x, y), (2.0, 2.0))


class TestPlainExtrude(unittest.TestCase):
    def test_reduces_to_prism(self):
        # no twist, unit scale: straight prism over 0 <= z <= 4.
        f = lambda p: W.linear_extrude(square2d, p, 4.0)
        self.assertLess(f((0.0, 0.0, 2.0)), 0.0)          # centre inside
        self.assertGreater(f((0.0, 0.0, 5.0)), 0.0)       # above top cap
        self.assertGreater(f((0.0, 0.0, -1.0)), 0.0)      # below base
        self.assertGreater(f((3.0, 0.0, 2.0)), 0.0)       # outside laterally

    def test_base_and_top_positions(self):
        f = lambda p: W.linear_extrude(square2d, p, 4.0)
        # near the top cap at z=4, still inside just below.
        self.assertLess(f((0.0, 0.0, 3.9)), 0.0)
        self.assertGreater(f((0.0, 0.0, 4.2)), 0.0)

    def test_height_must_be_positive(self):
        with self.assertRaises(ValueError):
            W.linear_extrude(square2d, (0.0, 0.0, 0.0), 0.0)


class TestTwist(unittest.TestCase):
    def test_cross_section_rotates_with_height(self):
        # square side 2 twisted 90deg over height 4. A point on the +x edge
        # midline of the base stays inside near the base; at the top the section
        # has rotated 90deg so the same world point sits over what was the edge.
        f = lambda p: W.twist_extrude(square2d, p, 4.0, 90.0)
        # A corner of the base square at (0.9, 0.9) is inside just above the base.
        self.assertLess(f((0.9, 0.9, 0.1)), 0.0)
        # Near the top the section has rotated ~90deg; a point rotated by +90
        # from a base-interior point is interior there.
        # Base interior point (0.9, 0.0); rotate +90 about z -> (0.0, 0.9).
        self.assertLess(f((0.0, 0.9, 3.9)), 0.0)

    def test_zero_twist_matches_plain(self):
        f0 = lambda p: W.twist_extrude(square2d, p, 3.0, 0.0)
        f1 = lambda p: W.linear_extrude(square2d, p, 3.0)
        for p in [(0.0, 0.0, 1.5), (0.5, 0.3, 0.5), (2.0, 0.0, 1.5)]:
            self.assertAlmostEqual(f0(p), f1(p), places=12)


class TestTaper(unittest.TestCase):
    def test_frustum_narrows_upward(self):
        # square side 2 at base tapering to factor 0.25 at top (side 0.5).
        f = lambda p: W.taper_extrude(square2d, p, 4.0, (0.25, 0.25))
        # a point at x=0.8 is inside near the base (half-width ~1)...
        self.assertLess(f((0.8, 0.0, 0.2)), 0.0)
        # ...but outside near the top where the half-width has shrunk to ~0.125.
        self.assertGreater(f((0.8, 0.0, 3.8)), 0.0)

    def test_sign_correct_inside_axis(self):
        f = lambda p: W.taper_extrude(square2d, p, 4.0, (0.5, 0.5))
        for z in (0.5, 2.0, 3.5):
            self.assertLess(f((0.0, 0.0, z)), 0.0)


if __name__ == "__main__":
    unittest.main()
