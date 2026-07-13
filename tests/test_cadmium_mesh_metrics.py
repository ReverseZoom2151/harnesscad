"""Tests for bench.cadmium_mesh_metrics."""

import math
import unittest

from harnesscad.eval.bench.cadmium_mesh_metrics import (
    Mesh,
    compare,
    discrete_mean_curvature_difference,
    euler_characteristic,
    euler_characteristic_match,
    is_watertight,
    mean_curvature,
    sphericity,
    sphericity_discrepancy,
    surface_area,
    undirected_edges,
    volume,
)


def unit_cube():
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    # Outward-oriented quad faces (CCW seen from outside).
    faces = [
        (0, 3, 2, 1),  # bottom z=0
        (4, 5, 6, 7),  # top z=1
        (0, 1, 5, 4),  # y=0
        (2, 3, 7, 6),  # y=1
        (1, 2, 6, 5),  # x=1
        (0, 4, 7, 3),  # x=0
    ]
    return Mesh.of(v, faces)


def scaled_cube(s):
    v = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
         (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3),
    ]
    return Mesh.of(v, faces)


def open_box():
    """Cube missing its top face -> boundary edges, not watertight."""
    v = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    faces = [
        (0, 3, 2, 1), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3),
    ]
    return Mesh.of(v, faces)


class MeshBasicsTest(unittest.TestCase):
    def test_cube_surface_area_and_volume(self):
        c = unit_cube()
        self.assertAlmostEqual(surface_area(c), 6.0)
        self.assertAlmostEqual(volume(c), 1.0)

    def test_scaled_volume(self):
        self.assertAlmostEqual(volume(scaled_cube(2.0)), 8.0)
        self.assertAlmostEqual(surface_area(scaled_cube(2.0)), 24.0)

    def test_edges_each_shared_twice(self):
        counts = undirected_edges(unit_cube())
        self.assertEqual(len(counts), 12)
        self.assertTrue(all(c == 2 for c in counts.values()))

    def test_validation(self):
        with self.assertRaises(ValueError):
            Mesh.of([(0, 0)], [(0, 0, 0)])
        with self.assertRaises(ValueError):
            Mesh.of([(0, 0, 0), (1, 0, 0)], [(0, 1)])
        with self.assertRaises(ValueError):
            Mesh.of([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 9)])


class WatertightTest(unittest.TestCase):
    def test_cube_is_watertight(self):
        self.assertTrue(is_watertight(unit_cube()))

    def test_open_box_not_watertight(self):
        self.assertFalse(is_watertight(open_box()))

    def test_empty_not_watertight(self):
        self.assertFalse(is_watertight(Mesh.of([(0, 0, 0)], [])))


class EulerTest(unittest.TestCase):
    def test_cube_euler_is_two(self):
        self.assertEqual(euler_characteristic(unit_cube()), 2)

    def test_match_and_mismatch(self):
        self.assertEqual(euler_characteristic_match(unit_cube(), scaled_cube(3.0)), 1)
        self.assertEqual(euler_characteristic_match(unit_cube(), open_box()), 0)


class SphericityTest(unittest.TestCase):
    def test_range_and_scale_invariance(self):
        s1 = sphericity(unit_cube())
        s2 = sphericity(scaled_cube(5.0))
        self.assertTrue(0.0 < s1 <= 1.0)
        # Sphericity is a dimensionless shape descriptor: scale-invariant.
        self.assertAlmostEqual(s1, s2)

    def test_cube_sphericity_value(self):
        # Known closed form for a cube: pi^(1/3) * 6^(2/3) / 6 ~= 0.8060.
        self.assertAlmostEqual(sphericity(unit_cube()),
                               math.pi ** (1 / 3) * 6 ** (2 / 3) / 6.0)

    def test_discrepancy_zero_for_congruent(self):
        self.assertAlmostEqual(
            sphericity_discrepancy(unit_cube(), scaled_cube(2.0)), 0.0)

    def test_needs_positive_volume(self):
        flat = Mesh.of([(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                       [(0, 1, 2), (0, 2, 1)])
        with self.assertRaises(ValueError):
            sphericity(flat)


class CurvatureTest(unittest.TestCase):
    def test_congruent_cubes_zero_dmcd(self):
        self.assertAlmostEqual(
            discrete_mean_curvature_difference(unit_cube(), scaled_cube(1.0), 5.0),
            0.0)

    def test_cube_edges_are_convex(self):
        # Large radius captures every edge; a convex cube has positive total.
        self.assertGreater(mean_curvature(unit_cube(), 10.0), 0.0)

    def test_radius_must_be_positive(self):
        with self.assertRaises(ValueError):
            mean_curvature(unit_cube(), 0.0)


class CompareTest(unittest.TestCase):
    def test_gating_when_both_watertight(self):
        r = compare(unit_cube(), scaled_cube(2.0), radius=5.0)
        self.assertTrue(r.both_watertight)
        self.assertEqual(r.euler_match, 1)
        self.assertAlmostEqual(r.sphericity_discrepancy, 0.0)
        self.assertAlmostEqual(r.dmcd, 0.0)

    def test_gating_suppressed_when_not_watertight(self):
        r = compare(open_box(), unit_cube())
        self.assertFalse(r.both_watertight)
        self.assertIsNone(r.euler_match)
        self.assertIsNone(r.sphericity_discrepancy)
        self.assertIsNone(r.dmcd)
        # Euler characteristics are still reported unconditionally.
        self.assertEqual(r.gt_euler, 2)


if __name__ == "__main__":
    unittest.main()
