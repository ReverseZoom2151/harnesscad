"""Tests for geometry.sdf.symmetry (ImplicitCAD arbitrary mirror + geo-mean scale)."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import symmetry as S
from harnesscad.domain.geometry.sdf import primitives as P
from harnesscad.domain.geometry.sdf import field_transforms as T


class TestReflectPoint(unittest.TestCase):
    def test_axis_normal_flips_that_axis(self):
        self.assertEqual(S.reflect_point((1.0, 0.0, 0.0), (3.0, 4.0, 5.0)),
                         (-3.0, 4.0, 5.0))
        self.assertEqual(S.reflect_point((0.0, 1.0, 0.0), (3.0, 4.0, 5.0)),
                         (3.0, -4.0, 5.0))

    def test_non_unit_normal(self):
        # scaling the normal must not change the reflection.
        a = S.reflect_point((2.0, 0.0, 0.0), (3.0, 4.0, 5.0))
        self.assertAlmostEqual(a[0], -3.0, places=12)
        self.assertAlmostEqual(a[1], 4.0, places=12)

    def test_diagonal_swaps(self):
        # mirror across the plane x = y (normal (1,-1,0)) swaps x and y.
        r = S.reflect_point((1.0, -1.0, 0.0), (2.0, 5.0, 1.0))
        self.assertAlmostEqual(r[0], 5.0, places=12)
        self.assertAlmostEqual(r[1], 2.0, places=12)
        self.assertAlmostEqual(r[2], 1.0, places=12)

    def test_involution(self):
        n, p = (0.3, 0.7, -0.2), (1.0, -2.0, 0.5)
        back = S.reflect_point(n, S.reflect_point(n, p))
        for i in range(3):
            self.assertAlmostEqual(back[i], p[i], places=12)

    def test_zero_normal_rejected(self):
        with self.assertRaises(ValueError):
            S.reflect_point((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


class TestMirrorField(unittest.TestCase):
    def test_mirror_of_translated_sphere(self):
        # sphere of diameter 2 centred at (2,0,0); mirror across x=0 -> centre (-2,0,0).
        f = T.translate(lambda q: P.sphere(q, 2.0), (2.0, 0.0, 0.0))
        g = S.mirror(f, (1.0, 0.0, 0.0))
        # centre of the mirrored sphere is inside.
        self.assertAlmostEqual(g((-2.0, 0.0, 0.0)), -1.0, places=9)
        # original centre now outside the mirrored copy.
        self.assertGreater(g((2.0, 0.0, 0.0)), 0.0)

    def test_mirror_isometry_preserves_distance_class(self):
        # a reflection must not change |f| magnitude at reflected points.
        f = T.translate(lambda q: P.sphere(q, 2.0), (2.0, 0.0, 0.0))
        g = S.mirror(f, (1.0, 0.0, 0.0))
        p = (0.5, 0.3, -0.1)
        self.assertAlmostEqual(g(p), f(S.reflect_point((1.0, 0.0, 0.0), p)), places=12)


class TestGeometricMeanScale(unittest.TestCase):
    def test_geo_mean_isotropic(self):
        self.assertAlmostEqual(S.geometric_mean_scale((2.0, 2.0, 2.0)), 2.0, places=12)

    def test_geo_mean_anisotropic(self):
        # cube-root of the product.
        self.assertAlmostEqual(S.geometric_mean_scale((1.0, 2.0, 4.0)),
                               8.0 ** (1.0 / 3.0), places=12)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            S.geometric_mean_scale(())

    def test_isotropic_scale_is_exact(self):
        # scaling a sphere isotropically by 2 keeps an exact field: radius doubles.
        f = lambda q: P.sphere(q, 2.0)  # radius 1
        g = S.scale_geometric(f, (2.0, 2.0, 2.0))
        # surface now at radius 2 along any axis.
        self.assertAlmostEqual(g((2.0, 0.0, 0.0)), 0.0, places=9)
        self.assertAlmostEqual(g((0.0, 0.0, 0.0)), -2.0, places=9)

    def test_zero_factor_rejected(self):
        with self.assertRaises(ValueError):
            S.scale_geometric(lambda q: 0.0, (0.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
