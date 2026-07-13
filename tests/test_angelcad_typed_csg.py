"""Tests for programs.angelcad_typed_csg."""

import math
import unittest

from harnesscad.domain.programs.ast.angelcad_typed_csg import (
    SHAPE2D,
    SOLID,
    CsgTypeError,
    Node,
    check,
    circle,
    cone,
    cube,
    cuboid,
    cylinder,
    hmatrix,
    identity,
    linear_extrude,
    mirror,
    polygon,
    polyhedron,
    projection2d,
    rectangle,
    result_type,
    rotate_extrude,
    rotate_z,
    scale,
    sphere,
    square,
    sweep,
    transform,
    transform_extrude,
    translate,
    type_check,
    union2d,
    union3d,
)


class TestTMatrix(unittest.TestCase):
    def test_identity_and_composition(self):
        i = identity()
        self.assertTrue(i.is_identity())
        t = translate(1, 2, 3)
        self.assertEqual((t * i).rows, t.rows)
        self.assertEqual(t.origin(), (1.0, 2.0, 3.0))

    def test_composition_order_is_right_first(self):
        # rotate first, then translate:  (T * R) * p
        m = translate(10, 0, 0) * rotate_z(deg=90)
        p = m.apply_pos((1, 0, 0))
        self.assertAlmostEqual(p[0], 10.0)
        self.assertAlmostEqual(p[1], 1.0)
        self.assertAlmostEqual(p[2], 0.0)
        # the other order translates before rotating
        m2 = rotate_z(deg=90) * translate(10, 0, 0)
        p2 = m2.apply_pos((1, 0, 0))
        self.assertAlmostEqual(p2[0], 0.0)
        self.assertAlmostEqual(p2[1], 11.0)

    def test_rotate_deg_or_rad_exclusive(self):
        a = rotate_z(deg=90)
        b = rotate_z(rad=math.pi / 2)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(a.rows[i][j], b.rows[i][j])
        with self.assertRaises(ValueError):
            rotate_z()
        with self.assertRaises(ValueError):
            rotate_z(deg=90, rad=1.0)

    def test_axes_accessors(self):
        m = rotate_z(deg=90)
        x = m.xdir()
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(x[1], 1.0)
        self.assertAlmostEqual(m.ydir()[0], -1.0)
        self.assertAlmostEqual(m.zdir()[2], 1.0)

    def test_scale_and_mirror(self):
        s = scale(2)
        self.assertEqual(s.apply_pos((1, 1, 1)), (2.0, 2.0, 2.0))
        mx = mirror(1, 0, 0)
        self.assertEqual(mx.apply_pos((3, 4, 5)), (-3.0, 4.0, 5.0))
        with self.assertRaises(ValueError):
            mirror(0, 0, 0)

    def test_hmatrix_orthonormalises(self):
        m = hmatrix((2, 0, 0), (1, 3, 0), pos=(5, 5, 5))
        self.assertAlmostEqual(m.xdir()[0], 1.0)
        self.assertAlmostEqual(m.ydir()[0], 0.0)
        self.assertAlmostEqual(m.ydir()[1], 1.0)
        self.assertAlmostEqual(m.zdir()[2], 1.0)
        self.assertEqual(m.origin(), (5.0, 5.0, 5.0))
        with self.assertRaises(ValueError):
            hmatrix((0, 0, 0), (0, 1, 0))

    def test_apply_vec_ignores_translation(self):
        m = translate(9, 9, 9)
        self.assertEqual(m.apply_vec((1, 0, 0)), (1.0, 0.0, 0.0))


class TestTypes(unittest.TestCase):
    def test_result_types(self):
        self.assertEqual(result_type(circle(2)), SHAPE2D)
        self.assertEqual(result_type(sphere(2)), SOLID)
        self.assertEqual(result_type(linear_extrude(circle(2), 5)), SOLID)
        self.assertEqual(result_type(projection2d(sphere(2))), SHAPE2D)
        self.assertEqual(result_type(transform(translate(1), cube(2))), SOLID)
        self.assertEqual(result_type(transform(translate(1), circle(2))), SHAPE2D)

    def test_operators_pick_the_right_dimension(self):
        self.assertEqual((cube(1) + sphere(1)).op, "union3d")
        self.assertEqual((circle(1) + square(1)).op, "union2d")
        self.assertEqual((cube(1) - sphere(1)).op, "difference3d")
        self.assertEqual((circle(1) & square(1)).op, "intersection2d")
        self.assertEqual((translate(1, 0, 0) * cube(1)).op, "transform")

    def test_valid_program_type_checks(self):
        model = union3d(
            transform(translate(0, 0, 10), cylinder(h=5, r=2)),
            linear_extrude(rectangle(4, 4, center=True), dz=3),
            cube(2, center=True) - sphere(1.2),
        )
        self.assertEqual(check(model), [])
        self.assertEqual(type_check(model), SOLID)


class TestDimensionErrors(unittest.TestCase):
    def test_2d_shape_in_3d_union(self):
        bad = union3d(cube(1), circle(1))
        diags = check(bad)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].code, "dim-mismatch")
        self.assertIn("union3d expects a solid child", diags[0].message)
        with self.assertRaises(CsgTypeError):
            type_check(bad)

    def test_solid_in_2d_union(self):
        diags = check(union2d(circle(1), cube(1)))
        self.assertEqual([d.code for d in diags], ["dim-mismatch"])

    def test_extrude_of_a_solid_is_an_error(self):
        diags = check(linear_extrude(cube(1), 3))
        self.assertEqual([d.code for d in diags], ["dim-mismatch"])

    def test_projection_of_a_2d_shape_is_an_error(self):
        diags = check(projection2d(circle(1)))
        self.assertEqual([d.code for d in diags], ["dim-mismatch"])

    def test_diagnostic_path_locates_the_error(self):
        bad = union3d(cube(1), union3d(sphere(1), square(2)))
        diags = check(bad)
        self.assertEqual(len(diags), 1)
        self.assertEqual(diags[0].path, "/union3d[1]")

    def test_sweep_and_transform_extrude_require_2d(self):
        self.assertEqual(check(sweep(circle(1), [(0, 0, 0), (0, 0, 5)])), [])
        self.assertTrue(check(sweep(sphere(1), [(0, 0, 0), (0, 0, 5)])))
        self.assertEqual(check(transform_extrude(circle(1), square(2))), [])
        self.assertTrue(check(transform_extrude(circle(1), cube(2))))


class TestParamErrors(unittest.TestCase):
    def test_negative_radius(self):
        diags = check(sphere(-1))
        self.assertEqual([d.code for d in diags], ["param-range"])

    def test_missing_and_unknown_params(self):
        diags = check(Node("cylinder", {"h": 3, "bogus": 1}))
        codes = sorted(d.code for d in diags)
        self.assertEqual(codes, ["param-missing", "param-missing", "param-unknown"])

    def test_bool_param_type(self):
        diags = check(Node("cube", {"size": 2, "center": 1}))
        self.assertEqual([d.code for d in diags], ["param-type"])

    def test_arity(self):
        self.assertTrue(any(d.code == "arity" for d in check(Node("union3d"))))
        self.assertTrue(
            any(d.code == "arity" for d in check(Node("cube", {"size": 1, "center": False}, (sphere(1),))))
        )
        self.assertTrue(
            any(
                d.code == "arity"
                for d in check(Node("fill2d", {}, (circle(1), circle(2))))
            )
        )

    def test_unknown_operator(self):
        diags = check(Node("frobnicate"))
        self.assertEqual([d.code for d in diags], ["unknown-op"])

    def test_rotate_extrude_angle_range(self):
        self.assertEqual(check(rotate_extrude(circle(1), angle=90.0)), [])
        self.assertTrue(
            any(d.code == "param-range" for d in check(rotate_extrude(circle(1), angle=400.0)))
        )

    def test_cone_radii(self):
        self.assertEqual(check(cone(4, 2, 0)), [])
        self.assertTrue(any(d.code == "param-range" for d in check(cone(4, 0, 0))))
        self.assertTrue(any(d.code == "param-range" for d in check(cone(4, -1, 2))))

    def test_polygon_point_dimension(self):
        self.assertEqual(check(polygon([(0, 0), (1, 0), (1, 1)])), [])
        diags = check(polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0)]))
        self.assertEqual(len(diags), 3)
        self.assertTrue(all(d.code == "dim-mismatch" for d in diags))

    def test_polyhedron_index_validation(self):
        tet_pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        good = polyhedron(tet_pts, [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)])
        self.assertEqual(check(good), [])
        bad = polyhedron(tet_pts, [(0, 1, 9)])
        self.assertEqual([d.code for d in check(bad)], ["index-range"])
        dup = polyhedron(tet_pts, [(0, 1, 1)])
        self.assertEqual([d.code for d in check(dup)], ["face-degenerate"])
        short = polyhedron(tet_pts, [(0, 1)])
        self.assertEqual([d.code for d in check(short)], ["face-degenerate"])

    def test_transform_requires_matrix(self):
        diags = check(Node("transform", {"matrix": "nope"}, (cube(1),)))
        self.assertEqual([d.code for d in diags], ["param-type"])

    def test_cuboid_and_square_ok(self):
        self.assertEqual(check(cuboid(1, 2, 3, center=True)), [])
        self.assertEqual(check(square(2)), [])


class TestWalk(unittest.TestCase):
    def test_preorder_is_deterministic(self):
        tree = union3d(cube(1), transform(translate(1), sphere(2)))
        ops = [n.op for _, n in tree.walk()]
        self.assertEqual(ops, ["union3d", "cube", "transform", "sphere"])
        self.assertEqual(ops, [n.op for _, n in tree.walk()])


if __name__ == "__main__":
    unittest.main()
