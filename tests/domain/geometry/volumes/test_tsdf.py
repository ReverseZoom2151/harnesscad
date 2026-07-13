"""Tests for geometry.cadmorph_tsdf (CADMorph tSDF + Boolean algebra)."""
import math
import unittest

from harnesscad.domain.geometry.volumes.tsdf import (
    TSDFGrid, l2_distance, occupancy_hamming, voxel_iou,
)


class TSDFConstructionTests(unittest.TestCase):
    def test_validates_dims_and_length(self):
        with self.assertRaises(ValueError):
            TSDFGrid((0, 1, 1), (), 0.2)
        with self.assertRaises(ValueError):
            TSDFGrid((2, 1, 1), (0.0,), 0.2)  # wrong length
        with self.assertRaises(ValueError):
            TSDFGrid((1, 1, 1), (0.0,), 0.0)  # non-positive truncation

    def test_from_sdf_truncates(self):
        # A plane far below the surface everywhere -> all clamp to -tau.
        g = TSDFGrid.from_sdf((2, 2, 2), lambda x, y, z: -100.0, truncation=0.2)
        self.assertTrue(all(v == -0.2 for v in g.values))
        self.assertEqual(g.occupied_count(), 8)
        self.assertAlmostEqual(g.occupancy_fraction(), 1.0)

    def test_sphere_inside_outside(self):
        g = TSDFGrid.sphere((5, 5, 5), center=(2, 2, 2), radius=1.5,
                            truncation=0.5)
        # Centre cell is well inside (negative), a corner is outside (positive).
        centre = 2 + 5 * (2 + 5 * 2)
        corner = 0
        self.assertTrue(g.is_inside(centre))
        self.assertFalse(g.is_inside(corner))


class BooleanAlgebraTests(unittest.TestCase):
    """Paper Eqs. 5-7: union=min, difference=max(a,-b), intersection=max(a,b)."""

    def setUp(self):
        self.a = TSDFGrid((3, 1, 1), (-0.2, -0.2, 0.2), 0.2)  # inside,inside,out
        self.b = TSDFGrid((3, 1, 1), (0.2, -0.2, -0.2), 0.2)  # out,inside,inside

    def test_union_is_min(self):
        u = self.a.union(self.b)
        self.assertEqual(u.values, (-0.2, -0.2, -0.2))  # occupied everywhere
        self.assertEqual(u.occupied_count(), 3)

    def test_intersection_is_max(self):
        i = self.a.intersection(self.b)
        # Only the middle cell is inside both.
        self.assertEqual(i.occupancy(), (False, True, False))

    def test_difference_is_max_neg(self):
        d = self.a.difference(self.b)
        # a minus b: middle cell (inside b) is carved away; first stays inside.
        self.assertEqual(d.occupancy(), (True, False, False))

    def test_difference_matches_formula(self):
        d = self.a.difference(self.b)
        for va, vb, vd in zip(self.a.values, self.b.values, d.values):
            self.assertAlmostEqual(vd, max(va, -vb))

    def test_result_is_truncated(self):
        # Build fields whose max(a,-b) would exceed tau before clamping.
        a = TSDFGrid((1, 1, 1), (0.2,), 0.2)
        b = TSDFGrid((1, 1, 1), (-0.2,), 0.2)
        d = a.difference(b)  # max(0.2, 0.2) = 0.2, still within tau
        self.assertLessEqual(abs(d.values[0]), 0.2 + 1e-9)

    def test_incompatible_dims_raise(self):
        other = TSDFGrid((2, 1, 1), (-0.2, 0.2), 0.2)
        with self.assertRaises(ValueError):
            self.a.union(other)

    def test_incompatible_truncation_raises(self):
        other = TSDFGrid((3, 1, 1), (0.0, 0.0, 0.0), 0.5)
        with self.assertRaises(ValueError):
            self.a.union(other)

    def test_boolean_composition_builds_shape(self):
        # A box with a spherical bite removed (sketch/extrude then cut).
        block = TSDFGrid.box((6, 6, 6), lo=(1, 1, 1), hi=(4, 4, 4),
                             truncation=0.5)
        hole = TSDFGrid.sphere((6, 6, 6), center=(2, 2, 2), radius=1.0,
                               truncation=0.5)
        cut = block.difference(hole)
        # Cutting removes material, so the result has <= the block's occupancy.
        self.assertLessEqual(cut.occupied_count(), block.occupied_count())


class DistanceMetricTests(unittest.TestCase):
    def test_l2_zero_for_identical(self):
        g = TSDFGrid.sphere((4, 4, 4), (2, 2, 2), 1.5, 0.4)
        self.assertEqual(l2_distance(g, g), 0.0)

    def test_l2_matches_euclidean(self):
        a = TSDFGrid((2, 1, 1), (0.1, -0.1), 0.2)
        b = TSDFGrid((2, 1, 1), (0.2, 0.2), 0.2)
        expect = math.sqrt((0.1 - 0.2) ** 2 + (-0.1 - 0.2) ** 2)
        self.assertAlmostEqual(l2_distance(a, b), expect)

    def test_voxel_iou_identical_is_one(self):
        g = TSDFGrid((3, 1, 1), (-0.2, -0.2, 0.2), 0.2)
        self.assertEqual(voxel_iou(g, g), 1.0)

    def test_voxel_iou_empty_shapes_is_one(self):
        a = TSDFGrid((2, 1, 1), (0.2, 0.2), 0.2)
        b = TSDFGrid((2, 1, 1), (0.2, 0.2), 0.2)
        self.assertEqual(voxel_iou(a, b), 1.0)

    def test_voxel_iou_partial(self):
        a = TSDFGrid((3, 1, 1), (-0.2, -0.2, 0.2), 0.2)  # inside {0,1}
        b = TSDFGrid((3, 1, 1), (-0.2, 0.2, 0.2), 0.2)   # inside {0}
        # intersection {0}=1, union {0,1}=2 -> 0.5
        self.assertEqual(voxel_iou(a, b), 0.5)

    def test_occupancy_hamming(self):
        a = TSDFGrid((3, 1, 1), (-0.2, -0.2, 0.2), 0.2)
        b = TSDFGrid((3, 1, 1), (-0.2, 0.2, 0.2), 0.2)
        self.assertEqual(occupancy_hamming(a, b), 1)


if __name__ == "__main__":
    unittest.main()
