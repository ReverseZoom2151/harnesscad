"""Tests for geometry.scadlm_csg_eval (deterministic OpenSCAD CSG evaluation)."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf.csg_eval import (
    BoolNode,
    Primitive,
    ScadEvalError,
    bounds,
    contains,
    evaluate_source,
    flatten,
    identity,
    invert,
    matmul,
    transform_point,
    volume,
    voxelize,
)


class TestMatrices(unittest.TestCase):
    def test_identity_is_neutral(self):
        m = ((1.0, 2.0, 3.0, 4.0), (0.0, 1.0, 0.0, 5.0),
             (0.0, 0.0, 1.0, 6.0), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(matmul(m, identity()), m)

    def test_invert_roundtrip(self):
        tree = evaluate_source("translate([1,2,3]) rotate([0,0,30]) cube(2);")
        m = tree.matrix
        back = matmul(m, invert(m))
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(back[i][j], 1.0 if i == j else 0.0, places=9)

    def test_transform_point(self):
        tree = evaluate_source("translate([1,2,3]) cube(1);")
        self.assertEqual(transform_point(tree.matrix, (0, 0, 0)), (1.0, 2.0, 3.0))

    def test_singular_matrix_raises(self):
        tree = evaluate_source("scale([1,1,0]) cube(1);")
        with self.assertRaises(ScadEvalError):
            contains(tree, (0.5, 0.5, 0.0))


class TestEvaluatePrimitives(unittest.TestCase):
    def test_cube_defaults(self):
        p = evaluate_source("cube();")
        self.assertIsInstance(p, Primitive)
        self.assertEqual(p.params["sx"], 1.0)
        self.assertTrue(contains(p, (0.5, 0.5, 0.5)))
        self.assertFalse(contains(p, (1.5, 0.5, 0.5)))

    def test_cube_vector_and_center(self):
        p = evaluate_source("cube([2,4,6], center=true);")
        self.assertTrue(contains(p, (0, 0, 0)))
        self.assertTrue(contains(p, (-1, -2, -3)))
        self.assertFalse(contains(p, (1.1, 0, 0)))

    def test_sphere_r_and_d(self):
        a = evaluate_source("sphere(r=2);")
        b = evaluate_source("sphere(d=4);")
        self.assertEqual(a.params["r"], b.params["r"])
        self.assertTrue(contains(a, (0, 0, 1.99)))
        self.assertFalse(contains(a, (0, 0, 2.01)))

    def test_cylinder_named_and_cone(self):
        c = evaluate_source("cylinder(h=10, r=3);")
        self.assertTrue(contains(c, (0, 0, 5)))
        self.assertFalse(contains(c, (3.5, 0, 5)))
        cone = evaluate_source("cylinder(h=10, r1=5, r2=0);")
        self.assertTrue(contains(cone, (4.0, 0, 1)))
        self.assertFalse(contains(cone, (4.0, 0, 9)))

    def test_cylinder_centered(self):
        c = evaluate_source("cylinder(h=4, r=1, center=true);")
        self.assertTrue(contains(c, (0, 0, -1.9)))
        self.assertFalse(contains(c, (0, 0, 2.1)))

    def test_unsupported_primitive_raises(self):
        with self.assertRaises(ScadEvalError):
            evaluate_source("hull() { cube(1); sphere(1); }")
        with self.assertRaises(ScadEvalError):
            evaluate_source("linear_extrude(3) square(2);")


class TestTransforms(unittest.TestCase):
    def test_translate(self):
        t = evaluate_source("translate([10,0,0]) cube(2);")
        self.assertTrue(contains(t, (11, 1, 1)))
        self.assertFalse(contains(t, (1, 1, 1)))

    def test_rotate_vector(self):
        t = evaluate_source("rotate([0,0,90]) translate([5,0,0]) cube(1, center=true);")
        self.assertTrue(contains(t, (0, 5, 0)))
        self.assertFalse(contains(t, (5, 0, 0)))

    def test_rotate_axis_angle(self):
        t = evaluate_source("rotate(90, [0,1,0]) translate([0,0,4]) sphere(1);")
        self.assertTrue(contains(t, (4, 0, 0)))

    def test_scale(self):
        t = evaluate_source("scale([2,1,1]) cube(1);")
        self.assertTrue(contains(t, (1.9, 0.5, 0.5)))
        self.assertFalse(contains(t, (2.1, 0.5, 0.5)))

    def test_mirror(self):
        t = evaluate_source("mirror([1,0,0]) translate([2,0,0]) cube(1);")
        self.assertTrue(contains(t, (-2.5, 0.5, 0.5)))

    def test_multmatrix(self):
        t = evaluate_source(
            "multmatrix([[1,0,0,5],[0,1,0,0],[0,0,1,0],[0,0,0,1]]) cube(1);")
        self.assertTrue(contains(t, (5.5, 0.5, 0.5)))

    def test_color_is_transparent_to_geometry(self):
        t = evaluate_source('color("red") cube(2);')
        self.assertTrue(contains(t, (1, 1, 1)))

    def test_nested_transform_composition(self):
        t = evaluate_source("translate([1,0,0]) translate([2,0,0]) cube(1);")
        self.assertTrue(contains(t, (3.5, 0.5, 0.5)))


class TestBooleans(unittest.TestCase):
    def test_union(self):
        t = evaluate_source("union() { cube(1); translate([5,0,0]) cube(1); }")
        self.assertIsInstance(t, BoolNode)
        self.assertTrue(contains(t, (0.5, 0.5, 0.5)))
        self.assertTrue(contains(t, (5.5, 0.5, 0.5)))

    def test_difference(self):
        t = evaluate_source(
            "difference() { cube(10, center=true); cylinder(h=20, r=2, center=true); }")
        self.assertFalse(contains(t, (0, 0, 0)))
        self.assertTrue(contains(t, (4, 4, 0)))

    def test_intersection(self):
        t = evaluate_source(
            "intersection() { cube(10); translate([5,5,5]) cube(10); }")
        self.assertTrue(contains(t, (7, 7, 7)))
        self.assertFalse(contains(t, (2, 2, 2)))

    def test_implicit_top_level_union(self):
        t = evaluate_source("cube(1); translate([9,0,0]) cube(1);")
        self.assertEqual(t.op, "union")
        self.assertTrue(contains(t, (9.5, 0.5, 0.5)))

    def test_flatten_signs(self):
        t = evaluate_source("difference() { cube(4); sphere(1); }")
        leaves = flatten(t)
        self.assertEqual([s for _, s in leaves], [1, -1])
        self.assertEqual([p.kind for p, _ in leaves], ["cube", "sphere"])

    def test_disabled_modifier_removes_geometry(self):
        t = evaluate_source("union() { cube(1); *sphere(50); }")
        self.assertFalse(contains(t, (0, 0, 40)))


class TestLanguageFeatures(unittest.TestCase):
    def test_variables_and_expressions(self):
        t = evaluate_source("w = 4; h = w * 2; cube([w, w, h]);")
        self.assertEqual(t.params["sz"], 8.0)

    def test_for_loop_unions(self):
        t = evaluate_source("for (i = [0:2]) translate([i*10, 0, 0]) cube(1);")
        self.assertEqual(len(flatten(t)), 3)
        self.assertTrue(contains(t, (20.5, 0.5, 0.5)))

    def test_nested_for_bindings(self):
        t = evaluate_source("for (i=[0,1], j=[0,1]) translate([i*5, j*5, 0]) cube(1);")
        self.assertEqual(len(flatten(t)), 4)

    def test_if_statement(self):
        t = evaluate_source("n = 3; if (n > 2) cube(2); else sphere(9);")
        self.assertEqual(t.kind, "cube")

    def test_user_module_with_defaults(self):
        src = """
        module post(h = 5, r = 1) { cylinder(h = h, r = r); }
        post();
        translate([10,0,0]) post(h = 20, r = 2);
        """
        t = evaluate_source(src)
        leaves = [p for p, _ in flatten(t)]
        self.assertEqual([p.params["h"] for p in leaves], [5.0, 20.0])

    def test_user_function(self):
        t = evaluate_source("function dbl(x) = x * 2; cube(dbl(3));")
        self.assertEqual(t.params["sx"], 6.0)

    def test_builtin_math_functions(self):
        t = evaluate_source("cube([max(2,5), sqrt(16), abs(-3)]);")
        self.assertEqual((t.params["sx"], t.params["sy"], t.params["sz"]),
                         (5.0, 4.0, 3.0))

    def test_trig_is_in_degrees(self):
        t = evaluate_source("cube([1, 1, sin(90) * 4]);")
        self.assertAlmostEqual(t.params["sz"], 4.0, places=9)

    def test_children_forwarding(self):
        src = """
        module shift() { translate([7,0,0]) children(); }
        shift() cube(1);
        """
        t = evaluate_source(src)
        self.assertTrue(contains(t, (7.5, 0.5, 0.5)))

    def test_list_comprehension_in_argument(self):
        t = evaluate_source("v = [for (i = [1:3]) i * 2]; cube([v[0], v[1], v[2]]);")
        self.assertEqual((t.params["sx"], t.params["sy"], t.params["sz"]),
                         (2.0, 4.0, 6.0))

    def test_let_statement(self):
        t = evaluate_source("let (s = 3) cube(s);")
        self.assertEqual(t.params["sx"], 3.0)

    def test_unknown_module_raises(self):
        with self.assertRaises(ScadEvalError):
            evaluate_source("frobnicate(3);")

    def test_recursion_limit(self):
        with self.assertRaises(ScadEvalError):
            evaluate_source("module r(n) { cube(1); r(n+1); } r(0);")


class TestGeometryQueries(unittest.TestCase):
    def test_bounds_of_translated_cube(self):
        lo, hi = bounds(evaluate_source("translate([1,2,3]) cube([2,2,2]);"))
        self.assertEqual(lo, (1.0, 2.0, 3.0))
        self.assertEqual(hi, (3.0, 4.0, 5.0))

    def test_bounds_of_union(self):
        lo, hi = bounds(evaluate_source("cube(1); translate([9,0,0]) cube(1);"))
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (10.0, 1.0, 1.0))

    def test_bounds_of_difference_keeps_minuend(self):
        box = bounds(evaluate_source(
            "difference() { cube(10); translate([-50,0,0]) cube(1); }"))
        self.assertEqual(box[0], (0.0, 0.0, 0.0))

    def test_bounds_of_disjoint_intersection_is_none(self):
        box = bounds(evaluate_source(
            "intersection() { cube(1); translate([50,0,0]) cube(1); }"))
        self.assertIsNone(box)

    def test_bounds_of_sphere(self):
        lo, hi = bounds(evaluate_source("sphere(r=3);"))
        self.assertEqual(lo, (-3.0, -3.0, -3.0))
        self.assertEqual(hi, (3.0, 3.0, 3.0))

    def test_voxelize_shape_and_fullness(self):
        grid = voxelize(evaluate_source("cube(4);"), resolution=4)
        self.assertEqual(len(grid), 4)
        self.assertTrue(all(v for plane in grid for row in plane for v in row))

    def test_volume_of_cube_is_exact(self):
        self.assertAlmostEqual(volume(evaluate_source("cube([2,3,4]);"), 8),
                               24.0, places=6)

    def test_volume_of_sphere_approximates_analytic(self):
        v = volume(evaluate_source("sphere(r=1);"), resolution=40)
        self.assertAlmostEqual(v, 4.0 / 3.0 * math.pi, delta=0.05)

    def test_volume_of_difference(self):
        src = "difference() { cube([10,10,10]); cube([10,10,5]); }"
        self.assertAlmostEqual(volume(evaluate_source(src), 10), 500.0, places=6)

    def test_volume_is_deterministic(self):
        src = "difference() { cube(6, center=true); sphere(r=3); }"
        a = volume(evaluate_source(src), 12)
        b = volume(evaluate_source(src), 12)
        self.assertEqual(a, b)

    def test_empty_program(self):
        self.assertIsNone(evaluate_source("a = 1;"))
        self.assertIsNone(bounds(None))
        self.assertEqual(volume(None), 0.0)
        self.assertFalse(contains(None, (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
