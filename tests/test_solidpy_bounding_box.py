"""Tests for geometry.solidpy_bounding_box."""

import unittest

from harnesscad.domain.geometry.sdf.scadlm_csg_eval import bounds, contains, evaluate_source
from harnesscad.domain.geometry.assembly.solidpy_bounding_box import (
    X,
    Y,
    Z,
    BoundingBox,
    bounding_box,
    distribute_in_grid,
    grid_positions,
    section_cut,
    split_body_planar,
)
from harnesscad.domain.programs.ast.scadlm_ast import parse
from harnesscad.domain.programs.emit.solidpy_scad_emit import cube, scad_render, sphere


class TestBoundingBoxOfPoints(unittest.TestCase):
    def test_3d_points(self):
        lo, hi = bounding_box([(0, 0, 0), (1, 2, 3), (-1, 5, 1)])
        self.assertEqual(lo, (-1.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 5.0, 3.0))

    def test_2d_points_are_z_zero(self):
        lo, hi = bounding_box([(0, 0), (4, 2)])
        self.assertEqual(lo[2], 0.0)
        self.assertEqual(hi[2], 0.0)

    def test_empty(self):
        with self.assertRaises(ValueError):
            bounding_box([])


class TestBoundingBox(unittest.TestCase):
    def test_from_points(self):
        bb = BoundingBox.from_points([(0, 0, 0), (10, 4, 2)])
        self.assertEqual(bb.size, [10.0, 4.0, 2.0])
        self.assertEqual(bb.position, [5.0, 2.0, 1.0])
        self.assertEqual(bb.min_corner(), (0.0, 0.0, 0.0))
        self.assertEqual(bb.max_corner(), (10.0, 4.0, 2.0))

    def test_volume_contains_intersects_union(self):
        a = BoundingBox([2, 2, 2])
        b = BoundingBox([2, 2, 2], [1, 0, 0])
        far = BoundingBox([2, 2, 2], [10, 0, 0])
        self.assertEqual(a.volume(), 8.0)
        self.assertTrue(a.contains((0.5, 0.5, 0.5)))
        self.assertFalse(a.contains((2, 0, 0)))
        self.assertTrue(a.intersects(b))
        self.assertFalse(a.intersects(far))
        merged = a.union(far)
        self.assertEqual(merged.min_corner()[0], -1.0)
        self.assertEqual(merged.max_corner()[0], 11.0)

    def test_equality(self):
        self.assertEqual(BoundingBox([1, 2, 3]), BoundingBox([1, 2, 3]))
        self.assertNotEqual(BoundingBox([1, 2, 3]), BoundingBox([1, 2, 4]))

    def test_bad_construction(self):
        with self.assertRaises(ValueError):
            BoundingBox([1, 2])
        with self.assertRaises(ValueError):
            BoundingBox([1, -2, 3])
        with self.assertRaises(ValueError):
            BoundingBox([1, 2, 3], [0, 0])

    def test_split_planar_halves(self):
        bb = BoundingBox([10, 10, 10])  # centred on the origin
        lower, upper = bb.split_planar(Z, 0.5)
        self.assertEqual(lower.size, [10.0, 10.0, 5.0])
        self.assertEqual(lower.position, [0.0, 0.0, -2.5])
        self.assertEqual(upper.position, [0.0, 0.0, 2.5])
        self.assertAlmostEqual(lower.volume() + upper.volume(), bb.volume())

    def test_split_planar_uneven_and_axis_by_name(self):
        bb = BoundingBox([10, 10, 10])
        a, b = bb.split_planar("x", 0.25)
        self.assertEqual(a.size[X], 2.5)
        self.assertEqual(b.size[X], 7.5)
        self.assertEqual(a.max_corner()[X], b.min_corner()[X])

    def test_split_planar_wall_thickness(self):
        bb = BoundingBox([10, 10, 10])
        a, _ = bb.split_planar(Z, 0.5, add_wall_thickness=1)
        self.assertEqual(a.size, [12.0, 12.0, 6.0])  # +2 on free axes, +1 on the cut

    def test_split_planar_validation(self):
        with self.assertRaises(ValueError):
            BoundingBox([1, 1, 1]).split_planar(Z, 0.0)
        with self.assertRaises(ValueError):
            BoundingBox([1, 1, 1]).split_planar("w")

    def test_cube_node(self):
        node = BoundingBox([2, 2, 2], [1, 0, 0]).cube()
        src = scad_render(node)
        lo, hi = bounds(evaluate_source(src))
        self.assertAlmostEqual(lo[0], 0.0)
        self.assertAlmostEqual(hi[0], 2.0)

    def test_cube_larger(self):
        node = BoundingBox([2, 2, 2]).cube(larger=True)
        lo, hi = bounds(evaluate_source(scad_render(node)))
        self.assertGreater(hi[0], 1.0)


class TestSplitBody(unittest.TestCase):
    def setUp(self):
        self.body = sphere(r=10, segments=16)
        self.bb = BoundingBox([20, 20, 20])

    def test_pieces_occupy_their_halves(self):
        a, box_a, b, box_b = split_body_planar(self.body, self.bb, Z, 0.5)
        tree_a = evaluate_source(scad_render(a))
        tree_b = evaluate_source(scad_render(b))
        self.assertTrue(contains(tree_a, (0, 0, -5)))
        self.assertFalse(contains(tree_a, (0, 0, 5)))
        self.assertTrue(contains(tree_b, (0, 0, 5)))
        self.assertFalse(contains(tree_b, (0, 0, -5)))
        self.assertEqual(box_a.position[Z], -5.0)
        self.assertEqual(box_b.position[Z], 5.0)

    def test_dowel_holes_are_subtracted_from_both(self):
        a, _, b, _ = split_body_planar(self.body, self.bb, Z, 0.5,
                                       dowel_holes=True, dowel_rad=2)
        for node in (a, b):
            src = scad_render(node)
            self.assertIn("difference()", src)
            self.assertEqual(src.count("cylinder("), 2)
            tree = evaluate_source(src)
            # the dowel holes sit either side of the cut on the X axis
            self.assertFalse(contains(tree, (-4, 0, 0)))
            self.assertFalse(contains(tree, (4, 0, 0)))

    def test_dowels_rotate_for_other_axes(self):
        a, _, _, _ = split_body_planar(self.body, self.bb, X, 0.5,
                                       dowel_holes=True, dowel_rad=2)
        src = scad_render(a)
        self.assertIn("rotate(a = 90", src)
        self.assertEqual(len(parse(src)), 1)

    def test_body_is_not_mutated(self):
        before = scad_render(self.body)
        split_body_planar(self.body, self.bb, Z, 0.5, dowel_holes=True)
        self.assertEqual(scad_render(self.body), before)

    def test_determinism(self):
        a1, _, _, _ = split_body_planar(self.body, self.bb, Z, 0.4,
                                        dowel_holes=True)
        a2, _, _, _ = split_body_planar(self.body, self.bb, Z, 0.4,
                                        dowel_holes=True)
        self.assertEqual(scad_render(a1), scad_render(a2))


class TestSectionCut(unittest.TestCase):
    def test_slice_keeps_only_the_cut_plane(self):
        node = section_cut(sphere(r=10, segments=16), Y, cut_point=0.0,
                           thickness=2, extent=100)
        tree = evaluate_source(scad_render(node))
        self.assertTrue(contains(tree, (0, -0.5, 0)))
        self.assertFalse(contains(tree, (0, 5, 0)))


class TestGrid(unittest.TestCase):
    def test_grid_positions_default_square(self):
        pos = grid_positions(4, (10, 20))
        self.assertEqual(pos, [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0),
                               (0.0, 20.0, 0.0), (10.0, 20.0, 0.0)])

    def test_grid_positions_scalar_cell_and_partial_row(self):
        pos = grid_positions(3, 5)
        self.assertEqual(len(pos), 3)
        self.assertEqual(pos[2], (0.0, 5.0, 0.0))

    def test_grid_positions_explicit_shape(self):
        pos = grid_positions(4, (1, 1), rows_and_cols=(1, 4))
        self.assertEqual([p[0] for p in pos], [0.0, 1.0, 2.0, 3.0])

    def test_grid_too_small(self):
        with self.assertRaises(ValueError):
            grid_positions(5, (1, 1), rows_and_cols=(2, 2))

    def test_grid_empty(self):
        self.assertEqual(grid_positions(0, (1, 1)), [])

    def test_distribute_in_grid_translates_each(self):
        node = distribute_in_grid([cube(1), cube(1)], (10, 10))
        src = scad_render(node)
        self.assertEqual(src.count("translate"), 2)
        self.assertIn("[10, 0, 0]", src)
        lo, hi = bounds(evaluate_source(src))
        self.assertAlmostEqual(hi[0], 11.0)


if __name__ == "__main__":
    unittest.main()
