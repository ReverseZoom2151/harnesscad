"""Tests for GeoCAD sketch-level symmetric editing (appendix H)."""

import unittest

from geometry import geocad_sketch_symmetry as sym


class CentroidTest(unittest.TestCase):
    def test_centroid(self):
        self.assertEqual(sym.centroid([(0, 0), (2, 0), (2, 2), (0, 2)]), (1.0, 1.0))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            sym.centroid([])


class AxisTest(unittest.TestCase):
    def test_axis_between_horizontal_centers(self):
        # Centers at (-2,0) and (2,0): axis is the vertical line x=0.
        ax = sym.symmetry_axis((-2, 0), (2, 0))
        self.assertEqual(ax.point, (0.0, 0.0))
        # Direction should be vertical (0, +/-1).
        self.assertAlmostEqual(abs(ax.direction[0]), 0.0)
        self.assertAlmostEqual(abs(ax.direction[1]), 1.0)

    def test_coincident_rejected(self):
        with self.assertRaises(ValueError):
            sym.symmetry_axis((1, 1), (1, 1))


class ReflectTest(unittest.TestCase):
    def test_reflect_point_across_vertical(self):
        ax = sym.symmetry_axis((-2, 0), (2, 0))  # x = 0
        self.assertAlmostEqual(sym.reflect_point((3, 5), ax)[0], -3.0)
        self.assertAlmostEqual(sym.reflect_point((3, 5), ax)[1], 5.0)

    def test_reflect_involution(self):
        ax = sym.symmetry_axis((0, -3), (0, 3))  # horizontal axis y=0
        p = (4.0, 7.0)
        back = sym.reflect_point(sym.reflect_point(p, ax), ax)
        self.assertAlmostEqual(back[0], p[0])
        self.assertAlmostEqual(back[1], p[1])

    def test_synthesise_pair(self):
        ax = sym.symmetry_axis((-2, 0), (2, 0))
        new_loop = [(1, 0), (3, 0), (3, 2)]
        a, b = sym.synthesise_symmetric_pair(new_loop, ax)
        self.assertEqual(a, new_loop)
        self.assertAlmostEqual(b[0][0], -1.0)
        self.assertAlmostEqual(b[1][0], -3.0)

    def test_reflected_centroid_matches_partner(self):
        # A loop and its reflection should have mirror-image centroids.
        ax = sym.symmetry_axis((-5, 0), (5, 0))
        loop_a = [(3, 1), (5, 1), (5, 3), (3, 3)]
        loop_b = sym.reflect_loop(loop_a, ax)
        ca, cb = sym.centroid(loop_a), sym.centroid(loop_b)
        self.assertAlmostEqual(ca[0], -cb[0])
        self.assertAlmostEqual(ca[1], cb[1])


if __name__ == "__main__":
    unittest.main()
