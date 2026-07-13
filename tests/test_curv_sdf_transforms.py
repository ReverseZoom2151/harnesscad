"""Tests for geometry.curv_sdf_transforms.

Checks level-set transforms (offset/shell/round/morph), that scale
compensation preserves the exact-distance property, that isometries (translate,
rotate, mirror) preserve field magnitude, and that repetition tiles space
correctly (each cell reproduces the base field about its own centre).
"""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import curv_sdf_primitives as P
from harnesscad.domain.geometry.sdf import curv_sdf_transforms as T


def sphere(r):
    return lambda p: P.sphere(p, 2.0 * r)


def grad_mag(f, p, h=1e-5):
    g = []
    for i in range(len(p)):
        pp, pm = list(p), list(p)
        pp[i] += h
        pm[i] -= h
        g.append((f(tuple(pp)) - f(tuple(pm))) / (2.0 * h))
    return math.sqrt(sum(c * c for c in g))


class TestLevelSet(unittest.TestCase):
    def test_offset(self):
        d = P.sphere((3.0, 0.0, 0.0), 2.0)  # radius 1 -> value 2
        self.assertAlmostEqual(T.offset(d, 0.5), 1.5)   # inflate
        self.assertAlmostEqual(T.offset(d, -0.5), 2.5)  # deflate

    def test_shell(self):
        # a shell of thickness 0.4 around a sphere: boundary at |d| = 0.2
        f = sphere(1.0)
        sh = lambda p: T.shell(f(p), 0.4)
        # a point exactly on the original surface is mid-shell -> -0.2
        self.assertAlmostEqual(sh((1.0, 0.0, 0.0)), -0.2, places=9)
        # inner and outer shell walls
        self.assertAlmostEqual(sh((0.8, 0.0, 0.0)), 0.0, places=9)
        self.assertAlmostEqual(sh((1.2, 0.0, 0.0)), 0.0, places=9)

    def test_morph(self):
        self.assertAlmostEqual(T.morph(2.0, 4.0, 0.0), 2.0)
        self.assertAlmostEqual(T.morph(2.0, 4.0, 1.0), 4.0)
        self.assertAlmostEqual(T.morph(2.0, 4.0, 0.25), 2.5)


class TestScaleCompensation(unittest.TestCase):
    def test_scale_preserves_exact_distance(self):
        # scaling a unit sphere by 3 -> sphere of radius 3; field stays exact
        f = sphere(1.0)
        g = T.scale(f, 3.0)
        # point at radius 5 should read distance 2
        self.assertAlmostEqual(g((5.0, 0.0, 0.0)), 2.0, places=9)
        # gradient magnitude stays 1 (Eikonal preserved)
        self.assertAlmostEqual(grad_mag(g, (5.0, 0.0, 0.0)), 1.0, places=5)

    def test_scale_without_compensation_would_break(self):
        # sanity: the compensated value differs from naive f(p/s)
        f = sphere(1.0)
        g = T.scale(f, 3.0)
        naive = f((5.0 / 3.0, 0.0, 0.0))
        self.assertAlmostEqual(g((5.0, 0.0, 0.0)), naive * 3.0, places=9)

    def test_stretch_is_lipschitz_bound(self):
        f = sphere(1.0)
        g = T.stretch(f, (2.0, 1.0, 1.0))
        for p in [(2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.5, 0.5)]:
            self.assertLessEqual(grad_mag(g, p), 1.0 + 1e-6)


class TestIsometries(unittest.TestCase):
    def test_translate(self):
        f = sphere(1.0)
        g = T.translate(f, (5.0, 0.0, 0.0))
        # centre moved to (5,0,0); at (5,0,0) distance is -1
        self.assertAlmostEqual(g((5.0, 0.0, 0.0)), -1.0, places=9)
        self.assertAlmostEqual(g((7.0, 0.0, 0.0)), 1.0, places=9)

    def test_rotate_z_preserves_box(self):
        # rotating an exact box by 90 deg maps (x,y)->(-y,x); distance preserved
        f = lambda p: P.box_exact(p, (2.0, 4.0, 2.0))
        g = T.rotate_z(f, math.pi / 2.0)
        # a probe far in +x for the rotated box equals probe in the pre-image
        p = (3.0, 1.0, 0.0)
        # rotate point by -angle to get pre-image
        self.assertAlmostEqual(g(p), grad_ref(f, p, math.pi / 2.0), places=9)

    def test_mirror(self):
        f = T.translate(sphere(1.0), (2.0, 0.0, 0.0))  # sphere at +x
        mg = T.mirror_x(f)
        # mirror makes a copy at -x too; both sides symmetric
        self.assertAlmostEqual(mg((2.0, 0.0, 0.0)), mg((-2.0, 0.0, 0.0)), places=9)
        self.assertAlmostEqual(mg((-2.0, 0.0, 0.0)), -1.0, places=9)


def grad_ref(f, p, angle):
    # helper: analytic reference for rotate_z (apply forward rotation to point)
    ca, sa = math.cos(-angle), math.sin(-angle)
    x = p[0] * ca - p[1] * sa
    y = p[0] * sa + p[1] * ca
    return f((x, y, p[2]))


class TestRepetition(unittest.TestCase):
    def test_repeat_x_tiles(self):
        # small sphere repeated every 4 units in X
        f = sphere(0.5)
        g = T.repeat_x(f, 4.0)
        # every cell centre (0, +/-4, +/-8, ...) reproduces the base field
        for cx in (-8.0, -4.0, 0.0, 4.0, 8.0):
            self.assertAlmostEqual(g((cx, 0.0, 0.0)), f((0.0, 0.0, 0.0)), places=9)
            self.assertAlmostEqual(g((cx + 0.3, 0.0, 0.0)),
                                   f((0.3, 0.0, 0.0)), places=9)

    def test_repeat_x_cell_boundary(self):
        # midway between two cells (x=2 for width 4) folds to the cell edge
        f = sphere(0.5)
        g = T.repeat_x(f, 4.0)
        self.assertAlmostEqual(g((2.0, 0.0, 0.0)), f((-2.0, 0.0, 0.0)), places=9)

    def test_repeat_xyz(self):
        f = sphere(0.5)
        g = T.repeat_xyz(f, (4.0, 4.0, 4.0))
        for c in [(0.0, 0.0, 0.0), (4.0, 4.0, 4.0), (-4.0, 0.0, 4.0)]:
            self.assertAlmostEqual(g(c), f((0.0, 0.0, 0.0)), places=9)
            off = (c[0] + 0.2, c[1] - 0.1, c[2] + 0.3)
            self.assertAlmostEqual(g(off), f((0.2, -0.1, 0.3)), places=9)

    def test_repeat_finite_clamps(self):
        # 3 copies along X spaced 4 apart, centred at x=0,4,8
        f = sphere(0.5)
        g = T.repeat_finite(f, (4.0, 4.0, 4.0), (3, 1, 1))
        # cells 0,1,2 exist
        for cx in (0.0, 4.0, 8.0):
            self.assertAlmostEqual(g((cx, 0.0, 0.0)), f((0.0, 0.0, 0.0)), places=9)
        # beyond the last cell there is no new copy: at x=12 the point is
        # clamped to cell 2 (x=8), so distance grows (not a fresh centre)
        self.assertGreater(g((12.0, 0.0, 0.0)), f((0.0, 0.0, 0.0)))
        self.assertAlmostEqual(g((12.0, 0.0, 0.0)), f((4.0, 0.0, 0.0)), places=9)


if __name__ == "__main__":
    unittest.main()
