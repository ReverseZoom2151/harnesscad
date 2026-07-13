"""Tests for geometry.angelcad_csg_boundingbox."""

import math
import unittest

from harnesscad.domain.geometry.sdf.angelcad_csg_boundingbox import (
    BBox3,
    BoundingBoxError,
    bounding_box,
    fits_within,
    is_provably_empty,
)
from harnesscad.domain.programs.ast.angelcad_typed_csg import (
    Node,
    circle,
    cone,
    cube,
    cuboid,
    cylinder,
    difference3d,
    intersection3d,
    linear_extrude,
    minkowski3d,
    offset2d,
    polygon,
    polyhedron,
    projection2d,
    rectangle,
    rotate_extrude,
    rotate_z,
    sphere,
    square,
    sweep,
    transform,
    translate,
    union3d,
)


class TestBBox3(unittest.TestCase):
    def test_empty(self):
        b = BBox3()
        self.assertTrue(b.is_empty())
        with self.assertRaises(BoundingBoxError):
            b.dx()

    def test_enclose_and_metrics(self):
        b = BBox3().enclose((0, 0, 0)).enclose((3, 4, 12))
        self.assertEqual(b.size(), (3.0, 4.0, 12.0))
        self.assertAlmostEqual(b.diagonal(), 13.0)
        self.assertEqual(b.center(), (1.5, 2.0, 6.0))
        self.assertEqual(b.p1(), (0.0, 0.0, 0.0))
        self.assertEqual(b.p2(), (3.0, 4.0, 12.0))

    def test_corners(self):
        self.assertEqual(len(BBox3((0, 0, 0), (1, 1, 1)).corners()), 8)

    def test_transformed_rotation(self):
        b = BBox3((0, 0, 0), (2, 1, 1)).transformed(rotate_z(deg=90))
        self.assertAlmostEqual(b.p1()[0], -1.0)
        self.assertAlmostEqual(b.p2()[1], 2.0)

    def test_union_intersect_minkowski(self):
        a = BBox3((0, 0, 0), (2, 2, 2))
        c = BBox3((1, 1, 1), (3, 3, 3))
        self.assertEqual(a.united(c), BBox3((0, 0, 0), (3, 3, 3)))
        self.assertEqual(a.intersected(c), BBox3((1, 1, 1), (2, 2, 2)))
        self.assertEqual(a.minkowski_sum(c), BBox3((1, 1, 1), (5, 5, 5)))

    def test_disjoint_intersection_is_empty(self):
        a = BBox3((0, 0, 0), (1, 1, 1))
        c = BBox3((5, 5, 5), (6, 6, 6))
        self.assertTrue(a.intersected(c).is_empty())


class TestPrimitives(unittest.TestCase):
    def test_sphere(self):
        self.assertEqual(bounding_box(sphere(2)), BBox3((-2, -2, -2), (2, 2, 2)))

    def test_cube_centered_or_not(self):
        self.assertEqual(bounding_box(cube(2)), BBox3((0, 0, 0), (2, 2, 2)))
        self.assertEqual(bounding_box(cube(2, center=True)), BBox3((-1, -1, -1), (1, 1, 1)))

    def test_cuboid(self):
        self.assertEqual(bounding_box(cuboid(2, 4, 6)), BBox3((0, 0, 0), (2, 4, 6)))

    def test_cylinder_and_cone_use_max_radius(self):
        self.assertEqual(bounding_box(cylinder(10, 3)), BBox3((-3, -3, 0), (3, 3, 10)))
        self.assertEqual(
            bounding_box(cone(10, 1, 5, center=True)), BBox3((-5, -5, -5), (5, 5, 5))
        )

    def test_2d_primitives_are_flat(self):
        for node in (circle(2), square(2), rectangle(2, 3), polygon([(0, 0), (1, 0), (0, 4)])):
            self.assertEqual(bounding_box(node).dz(), 0.0)
        self.assertEqual(bounding_box(circle(2)), BBox3((-2, -2, 0), (2, 2, 0)))

    def test_polyhedron_from_points(self):
        node = polyhedron([(0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)], [(0, 2, 1)])
        self.assertEqual(bounding_box(node), BBox3((0, 0, 0), (1, 2, 3)))

    def test_missing_param(self):
        with self.assertRaises(BoundingBoxError):
            bounding_box(Node("sphere"))


class TestOperators(unittest.TestCase):
    def test_union_encloses(self):
        box = bounding_box(union3d(cube(2), transform(translate(5, 0, 0), sphere(1))))
        self.assertEqual(box, BBox3((0, -1, -1), (6, 2, 2)))

    def test_difference_is_bounded_by_first_operand(self):
        box = bounding_box(difference3d(cube(2), sphere(10)))
        self.assertEqual(box, BBox3((0, 0, 0), (2, 2, 2)))

    def test_intersection_is_the_overlap(self):
        box = bounding_box(
            intersection3d(cuboid(4, 4, 4), transform(translate(3, 0, 0), cuboid(4, 4, 4)))
        )
        self.assertEqual(box, BBox3((3, 0, 0), (4, 4, 4)))

    def test_disjoint_intersection_is_provably_empty(self):
        model = intersection3d(cube(1), transform(translate(10, 0, 0), cube(1)))
        self.assertTrue(is_provably_empty(model))
        self.assertFalse(is_provably_empty(union3d(cube(1), sphere(1))))

    def test_minkowski_adds_boxes(self):
        box = bounding_box(minkowski3d(cube(2), sphere(1)))
        self.assertEqual(box, BBox3((-1, -1, -1), (3, 3, 3)))

    def test_offset2d_grows(self):
        self.assertEqual(bounding_box(offset2d(circle(2), 1.0)), BBox3((-3, -3, 0), (3, 3, 0)))
        self.assertEqual(bounding_box(offset2d(circle(2), -1.0)), BBox3((-3, -3, 0), (3, 3, 0)))

    def test_linear_extrude_lifts_the_profile(self):
        self.assertEqual(
            bounding_box(linear_extrude(rectangle(2, 3), dz=5)), BBox3((0, 0, 0), (2, 3, 5))
        )

    def test_rotate_extrude_revolves_x_into_a_radius(self):
        profile = transform(translate(4, 0, 0), rectangle(1, 2))
        box = bounding_box(rotate_extrude(profile, angle=360.0))
        self.assertEqual(box, BBox3((-5, -5, 0), (5, 5, 2)))

    def test_projection_flattens_z(self):
        box = bounding_box(projection2d(sphere(3)))
        self.assertEqual(box, BBox3((-3, -3, 0), (3, 3, 0)))

    def test_sweep_bounds_the_path(self):
        box = bounding_box(sweep(circle(1), [(0, 0, 0), (0, 0, 10)]))
        self.assertEqual(box, BBox3((-1, -1, -1), (1, 1, 11)))

    def test_transform_of_a_boolean(self):
        model = transform(translate(0, 0, 100), union3d(cube(2), sphere(1)))
        self.assertEqual(bounding_box(model).p1()[2], 99.0)

    def test_nested_transform_composition(self):
        model = transform(translate(10, 0, 0), transform(rotate_z(deg=90), cuboid(4, 1, 1)))
        box = bounding_box(model)
        self.assertAlmostEqual(box.p1()[0], 9.0)
        self.assertAlmostEqual(box.p2()[1], 4.0)

    def test_unknown_op(self):
        with self.assertRaises(BoundingBoxError):
            bounding_box(Node("frobnicate", {}, (cube(1),)))


class TestFits(unittest.TestCase):
    def test_fits_within_build_volume(self):
        model = cuboid(100, 200, 50)
        self.assertTrue(fits_within(model, (250, 210, 210)))
        self.assertFalse(fits_within(model, (100, 100, 100)))

    def test_orientation_independent(self):
        model = cuboid(10, 200, 10)
        self.assertTrue(fits_within(model, (200, 20, 20)))

    def test_exact_fit(self):
        self.assertTrue(fits_within(cube(10), (10, 10, 10)))


if __name__ == "__main__":
    unittest.main()
