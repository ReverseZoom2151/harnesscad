"""Tests for geometry.dontmesh_halfspace_csg."""

import unittest

from geometry.dontmesh_halfspace_csg import (
    Cell,
    CSGModel,
    Cylinder,
    HalfSpace,
    Plane,
    all_cells_bounded,
    any_cells_overlap,
    axis_box_cell,
    axis_cylinder_cell,
    bounding_box,
    cell_is_bounded,
    cells_overlap,
    grid_points,
    iou,
    occupancy,
    random_points,
    volume_fraction,
)


class TestSurfaces(unittest.TestCase):
    def test_plane_evaluate_sign(self):
        p = Plane.axis_aligned("x", 2.0)
        self.assertLess(p.evaluate((1.0, 0.0, 0.0)), 0)
        self.assertGreater(p.evaluate((3.0, 0.0, 0.0)), 0)
        self.assertAlmostEqual(p.evaluate((2.0, 5.0, 9.0)), 0.0)

    def test_cylinder_inside_negative(self):
        c = Cylinder("z", 0.0, 0.0, 1.0)
        self.assertLess(c.evaluate((0.0, 0.0, 5.0)), 0)  # on axis, inside
        self.assertGreater(c.evaluate((2.0, 0.0, 0.0)), 0)  # outside
        self.assertAlmostEqual(c.evaluate((1.0, 0.0, 0.0)), 0.0)  # on surface

    def test_halfspace_sense(self):
        p = Plane.axis_aligned("x", 0.0)
        hs_pos = HalfSpace(p, +1)
        hs_neg = HalfSpace(p, -1)
        self.assertTrue(hs_pos.contains((1.0, 0.0, 0.0)))
        self.assertFalse(hs_pos.contains((-1.0, 0.0, 0.0)))
        self.assertTrue(hs_neg.contains((-1.0, 0.0, 0.0)))


class TestCellMembership(unittest.TestCase):
    def test_box_cell_membership(self):
        cell = axis_box_cell((0, 0, 0), (2, 2, 2))
        self.assertTrue(cell.contains((1, 1, 1)))
        self.assertFalse(cell.contains((3, 1, 1)))
        self.assertFalse(cell.contains((-0.5, 1, 1)))

    def test_cylinder_cell_membership(self):
        cell = axis_cylinder_cell("z", 0.0, 0.0, 1.0, 0.0, 4.0)
        self.assertTrue(cell.contains((0, 0, 2)))
        self.assertFalse(cell.contains((0, 0, 5)))  # above cap
        self.assertFalse(cell.contains((2, 0, 2)))  # outside radius

    def test_model_union(self):
        a = axis_box_cell((0, 0, 0), (1, 1, 1))
        b = axis_box_cell((5, 5, 5), (6, 6, 6))
        m = CSGModel((a, b))
        self.assertTrue(m.contains((0.5, 0.5, 0.5)))
        self.assertTrue(m.contains((5.5, 5.5, 5.5)))
        self.assertFalse(m.contains((3, 3, 3)))


class TestSampling(unittest.TestCase):
    def test_grid_points_count(self):
        pts = grid_points(((0, 0, 0), (1, 1, 1)), 4)
        self.assertEqual(len(pts), 64)

    def test_occupancy_and_volume(self):
        cell = axis_box_cell((0, 0, 0), (1, 1, 1))
        m = CSGModel((cell,))
        box = ((0, 0, 0), (2, 2, 2))
        vf = volume_fraction(m, box, 8)
        # Box occupies 1/8 of the domain volume.
        self.assertAlmostEqual(vf, 1.0 / 8.0, places=2)
        mask = occupancy(m, box, 8)
        self.assertEqual(len(mask), 512)

    def test_iou_self_is_one(self):
        m = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        box = ((-1, -1, -1), (2, 2, 2))
        self.assertEqual(iou(m, m, box, 6), 1.0)

    def test_iou_disjoint_is_zero(self):
        a = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        b = CSGModel((axis_box_cell((3, 3, 3), (4, 4, 4)),))
        box = ((-1, -1, -1), (5, 5, 5))
        self.assertEqual(iou(a, b, box, 8), 0.0)

    def test_iou_empty_both(self):
        # Cell that is empty over the probe (contradictory half-spaces).
        empty = Cell((HalfSpace(Plane.axis_aligned("x", 0.0), +1),
                      HalfSpace(Plane.axis_aligned("x", 0.0), -1)))
        # Actually this is the plane x=0 (measure zero); use two disjoint planes.
        empty = Cell((HalfSpace(Plane.axis_aligned("x", 1.0), +1),
                      HalfSpace(Plane.axis_aligned("x", -1.0), -1)))
        m = CSGModel((empty,))
        box = ((0, 0, 0), (0.5, 0.5, 0.5))
        self.assertEqual(iou(m, m, box, 4), 1.0)

    def test_bounding_box(self):
        m = CSGModel((axis_box_cell((1, 1, 1), (2, 2, 2)),))
        bb = bounding_box(m, ((0, 0, 0), (3, 3, 3)), 12)
        self.assertIsNotNone(bb)
        (lo, hi) = bb
        self.assertTrue(all(0.9 <= v <= 1.3 for v in lo))
        self.assertTrue(all(1.7 <= v <= 2.1 for v in hi))

    def test_random_points_deterministic(self):
        box = ((0, 0, 0), (1, 1, 1))
        p1 = random_points(box, 10, seed=7)
        p2 = random_points(box, 10, seed=7)
        p3 = random_points(box, 10, seed=8)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)


class TestValidityOverlap(unittest.TestCase):
    def test_bounded_box_is_bounded(self):
        cell = axis_box_cell((0, 0, 0), (2, 2, 2))
        probe = ((-2, -2, -2), (4, 4, 4))
        self.assertTrue(cell_is_bounded(cell, probe, 12))

    def test_half_open_is_unbounded(self):
        # Only one plane -> half space, open on many sides.
        cell = Cell((HalfSpace(Plane.axis_aligned("x", 0.0), +1),))
        probe = ((-2, -2, -2), (4, 4, 4))
        self.assertFalse(cell_is_bounded(cell, probe, 10))

    def test_all_cells_bounded(self):
        m = CSGModel((
            axis_box_cell((0, 0, 0), (1, 1, 1)),
            axis_box_cell((2, 2, 2), (3, 3, 3)),
        ))
        probe = ((-1, -1, -1), (4, 4, 4))
        self.assertTrue(all_cells_bounded(m, probe, 12))

    def test_overlap_detection(self):
        a = axis_box_cell((0, 0, 0), (2, 2, 2))
        b = axis_box_cell((1, 1, 1), (3, 3, 3))
        box = ((0, 0, 0), (3, 3, 3))
        self.assertTrue(cells_overlap(a, b, box, 12))
        c = axis_box_cell((5, 5, 5), (6, 6, 6))
        self.assertFalse(cells_overlap(a, c, box, 12))

    def test_any_cells_overlap(self):
        disjoint = CSGModel((
            axis_box_cell((0, 0, 0), (1, 1, 1)),
            axis_box_cell((2, 2, 2), (3, 3, 3)),
        ))
        box = ((-1, -1, -1), (4, 4, 4))
        self.assertFalse(any_cells_overlap(disjoint, box, 12))
        overlapping = CSGModel((
            axis_box_cell((0, 0, 0), (2, 2, 2)),
            axis_box_cell((1, 1, 1), (3, 3, 3)),
        ))
        self.assertTrue(any_cells_overlap(overlapping, box, 12))


if __name__ == "__main__":
    unittest.main()
