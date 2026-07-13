"""Tests for drawings.vitruvion_hand_drawn_noise."""

import math
import unittest

from harnesscad.domain.drawings.vitruvion_hand_drawn_noise import (
    HandDrawnNoise,
    cholesky,
    matern_kernel,
)
from harnesscad.domain.geometry.sketch.vitruvion_sketch_norm import VArc, VCircle, VPoint, entity_from_params

RES = 32  # small kernel keeps the pure-Python Cholesky fast in tests


def _noise(entities, seed=0, **kwargs):
    return HandDrawnNoise(entities, seed=seed, resolution=RES, **kwargs)


class TestMaternKernel(unittest.TestCase):
    def test_symmetric_and_nugget_on_diagonal(self):
        k = matern_kernel([0.0, 0.25, 0.5], length_scale=0.05, amplitude=0.002)
        self.assertAlmostEqual(k[0][1], k[1][0])
        # Diagonal is amp^2 * (1 + nugget).
        self.assertAlmostEqual(k[0][0], 0.002 ** 2 * (1.0 + 1e-6))

    def test_correlation_decays_with_distance(self):
        k = matern_kernel([0.0, 0.02, 0.2], length_scale=0.05, amplitude=0.002)
        self.assertGreater(k[0][1], k[0][2])

    def test_nu5_is_smoother_than_nu3(self):
        k3 = matern_kernel([0.0, 0.05], nu=3)
        k5 = matern_kernel([0.0, 0.05], nu=5)
        self.assertGreater(k5[0][1], k3[0][1])

    def test_bad_nu(self):
        with self.assertRaises(ValueError):
            matern_kernel([0.0, 1.0], nu=4)


class TestCholesky(unittest.TestCase):
    def test_factor_reproduces_the_matrix(self):
        matrix = matern_kernel([0.0, 0.03, 0.06, 0.09])
        lower = cholesky(matrix)
        n = len(matrix)
        for i in range(n):
            for j in range(n):
                got = sum(lower[i][k] * lower[j][k] for k in range(n))
                self.assertAlmostEqual(got, matrix[i][j], places=12)

    def test_lower_triangular(self):
        lower = cholesky(matern_kernel([0.0, 0.05, 0.1]))
        self.assertEqual(lower[0][1], 0.0)
        self.assertEqual(lower[0][2], 0.0)

    def test_non_pd_raises(self):
        with self.assertRaises(ValueError):
            cholesky([[1.0, 2.0], [2.0, 1.0]])


class TestHandDrawnNoise(unittest.TestCase):
    def setUp(self):
        self.entities = [
            entity_from_params([-0.4, 0.0, 0.4, 0.0]),
            VCircle(xCenter=0.0, yCenter=0.0, radius=0.3),
            VPoint(x=0.1, y=-0.2),
        ]

    def test_deterministic_in_seed(self):
        a = _noise(self.entities, seed=5).render()
        b = _noise(self.entities, seed=5).render()
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        a = _noise(self.entities, seed=1).render()
        b = _noise(self.entities, seed=2).render()
        self.assertNotEqual(a, b)

    def test_scale_is_ten_diagonals(self):
        noise = _noise([VPoint(x=0.0, y=0.0), VPoint(x=3.0, y=4.0)])
        self.assertAlmostEqual(noise.scale, 50.0)

    def test_reference_extent_quirk(self):
        # Circle at (2, 0) with r = 1: the reference's extent is the union of
        # [-1, 1] (radius about the ORIGIN) with the centre, so x spans [-1, 2].
        quirk = _noise([VCircle(xCenter=2.0, yCenter=0.0, radius=1.0)])
        self.assertAlmostEqual(quirk.min_x, -1.0)
        self.assertAlmostEqual(quirk.max_x, 2.0)
        # The corrected version is the true bounding box, x spanning [1, 3].
        fixed = _noise([VCircle(xCenter=2.0, yCenter=0.0, radius=1.0)], bbox_extent=True)
        self.assertAlmostEqual(fixed.min_x, 1.0)
        self.assertAlmostEqual(fixed.max_x, 3.0)

    def test_line_stays_near_the_ideal_stroke(self):
        noise = _noise(self.entities, seed=3)
        polyline = noise.line((-0.4, 0.0), (0.4, 0.0))
        self.assertGreater(len(polyline), 1)
        # The wobble is small: every sample stays close to the y = 0 axis.
        for _, y in polyline:
            self.assertLess(abs(y), 0.2)
        # ... but it is not exactly straight.
        self.assertTrue(any(abs(y) > 0 for _, y in polyline))

    def test_line_starts_at_the_start_point(self):
        noise = _noise(self.entities, seed=4)
        polyline = noise.line((-0.4, 0.1), (0.4, 0.1))
        # Station 0 has zero arclength, so only the (small) normal offset moves it.
        self.assertAlmostEqual(polyline[0][0], -0.4, places=6)

    def test_circle_has_a_gap(self):
        noise = _noise(self.entities, seed=6)
        polyline = noise.circle((0.0, 0.0), 0.3)
        gap = math.hypot(
            polyline[0][0] - polyline[-1][0], polyline[0][1] - polyline[-1][1]
        )
        self.assertGreater(gap, 1e-4)

    def test_arc_radius_wobbles_around_the_ideal(self):
        noise = _noise(self.entities, seed=7)
        polyline = noise.arc((0.0, 0.0), 0.3, 0.0, 90.0)
        radii = [math.hypot(x, y) for x, y in polyline]
        self.assertTrue(all(abs(r - 0.3) < 0.15 for r in radii))

    def test_arc_always_yields_at_least_one_sample(self):
        noise = _noise(self.entities, seed=8)
        self.assertEqual(len(noise.arc((0.0, 0.0), 1e-9, 0.0, 1.0)), 1)

    def test_point_is_displaced(self):
        noise = _noise(self.entities, seed=9)
        x, y = noise.point((0.5, -0.5))
        self.assertNotEqual((x, y), (0.5, -0.5))
        self.assertLess(abs(x - 0.5), 0.1)

    def test_render_covers_every_entity(self):
        strokes = _noise(self.entities + [VArc(radius=0.2, endParam=1.0)], seed=2).render()
        self.assertEqual(len(strokes), 4)
        self.assertEqual(len(strokes[2]), 1)  # the point

    def test_unsupported_entity_raises(self):
        with self.assertRaises(ValueError):
            _noise([object()]).render()

    def test_resolution_guard(self):
        with self.assertRaises(ValueError):
            HandDrawnNoise([VPoint()], resolution=1)


if __name__ == "__main__":
    unittest.main()
