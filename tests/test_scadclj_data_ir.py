"""Tests for programs.scadclj_data_ir."""

import math
import unittest

from programs.scadclj_data_ir import (
    background,
    call,
    circle,
    cube,
    cylinder,
    define_module,
    deg_to_rad,
    difference,
    excise,
    extrude_linear,
    extrude_rotate,
    format_number,
    fn,
    include,
    intersection,
    mirror,
    polygon,
    polyhedron,
    postwalk,
    rad_to_deg,
    render,
    rotate,
    scale,
    sphere,
    square,
    translate,
    union,
    with_center,
    with_fn,
    write_scad,
)


class NumberFormatTest(unittest.TestCase):
    def test_int_preserved(self):
        self.assertEqual(format_number(3), "3")

    def test_float_trailing_zeros_trimmed(self):
        self.assertEqual(format_number(1.5000), "1.5")
        self.assertEqual(format_number(2.0), "2")

    def test_bool(self):
        self.assertEqual(format_number(True), "true")

    def test_negative_zero(self):
        self.assertEqual(format_number(-0.0), "0")

    def test_deterministic(self):
        self.assertEqual(format_number(1.0 / 3.0), format_number(1.0 / 3.0))


class PrimitiveEmitTest(unittest.TestCase):
    def test_cube(self):
        self.assertEqual(write_scad(cube(1, 2, 3, center=False)),
                         "cube ([1, 2, 3]);\n")

    def test_cube_center(self):
        self.assertEqual(write_scad(cube(1, 2, 3, center=True)),
                         "cube ([1, 2, 3], center=true);\n")

    def test_sphere(self):
        self.assertEqual(write_scad(sphere(5)), "sphere (r=5);\n")

    def test_cylinder_single_r(self):
        self.assertEqual(write_scad(cylinder(2, 10, center=False)),
                         "cylinder (h=10, r=2);\n")

    def test_cylinder_cone(self):
        self.assertEqual(write_scad(cylinder([2, 4], 10, center=False)),
                         "cylinder (h=10, r1=2, r2=4);\n")

    def test_circle(self):
        self.assertEqual(write_scad(circle(4)), "circle (r=4);\n")

    def test_square(self):
        self.assertEqual(write_scad(square(3, 4, center=False)),
                         "square ([3, 4]);\n")

    def test_polygon(self):
        out = write_scad(polygon([[0, 0], [1, 0], [1, 1]]))
        self.assertEqual(out, "polygon (points=[[0, 0], [1, 0], [1, 1]]);\n")

    def test_polyhedron(self):
        out = write_scad(polyhedron([[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                                    [[0, 1, 2]]))
        self.assertIn("points=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]", out)
        self.assertIn("faces=[[0, 1, 2]]", out)


class TransformEmitTest(unittest.TestCase):
    def test_translate_block(self):
        out = write_scad(translate([10, 0, 0], sphere(2)))
        self.assertEqual(
            out,
            "translate ([10, 0, 0]) {\n  sphere (r=2);\n}\n")

    def test_rotate_radians_to_degrees(self):
        # pi/2 radians -> 90 degrees on the z axis.
        out = write_scad(rotate(math.pi / 2, [0, 0, 1], cube(1, 1, 1, center=False)))
        self.assertIn("rotate (a=90, v=[0, 0, 1])", out)

    def test_rotatec_vector_degrees(self):
        out = write_scad(rotate([deg_to_rad(90), 0, 0], cube(1, 1, 1, center=False)))
        self.assertIn("rotate ([90,0,0])", out)

    def test_scale_mirror(self):
        self.assertIn("scale ([2, 2, 2])", write_scad(scale([2, 2, 2], sphere(1))))
        self.assertIn("mirror ([1, 0, 0])", write_scad(mirror([1, 0, 0], sphere(1))))


class BooleanTest(unittest.TestCase):
    def test_union_flattens_list(self):
        shapes = [sphere(1), cube(1, 1, 1, center=False)]
        out = write_scad(union(shapes))
        self.assertIn("sphere (r=1)", out)
        self.assertIn("cube ([1, 1, 1])", out)
        self.assertTrue(out.startswith("union () {\n"))

    def test_difference(self):
        out = write_scad(difference(cube(2, 2, 2, center=False), sphere(1)))
        self.assertTrue(out.startswith("difference () {\n"))

    def test_excise_subtracts_from_last(self):
        # excise(a, target) == target - a
        target = cube(4, 4, 4, center=False)
        tool = sphere(1)
        node = excise(tool, target)
        self.assertEqual(node[0], ":difference")
        # first child of the difference is the target (last arg to excise)
        self.assertEqual(node[1], target)
        self.assertEqual(node[2], tool)

    def test_intersection(self):
        out = write_scad(intersection(cube(2, 2, 2, center=False), sphere(1.5)))
        self.assertTrue(out.startswith("intersection () {\n"))


class SpecialVariableTest(unittest.TestCase):
    def test_with_fn_resolves_on_circle(self):
        with with_fn(64):
            c = circle(5)
        self.assertEqual(write_scad(c), "circle ($fn=64, r=5);\n")

    def test_with_fn_resolves_on_sphere(self):
        with with_fn(32):
            s = sphere(3)
        self.assertIn("$fn=32", write_scad(s))

    def test_binding_restored_after_block(self):
        with with_fn(64):
            pass
        self.assertEqual(write_scad(circle(1)), "circle (r=1);\n")

    def test_with_center(self):
        with with_center(False):
            out = write_scad(cube(1, 1, 1))
        self.assertEqual(out, "cube ([1, 1, 1]);\n")

    def test_fn_literal_statement(self):
        self.assertEqual(write_scad(fn(100)), "$fn = 100;\n")


class LibraryAndModuleTest(unittest.TestCase):
    def test_include(self):
        self.assertEqual(write_scad(include("MCAD/gears.scad")),
                         "include <MCAD/gears.scad>\n")

    def test_call(self):
        self.assertEqual(write_scad(call("gear", 5, 20)), "gear(5, 20);\n")

    def test_define_module(self):
        out = write_scad(define_module("widget", union(sphere(1))))
        self.assertTrue(out.startswith("module widget() {\n"))
        self.assertIn("sphere (r=1)", out)


class ExtrusionTest(unittest.TestCase):
    def test_linear_extrude(self):
        out = write_scad(extrude_linear({"height": 10, "center": False}, circle(3)))
        self.assertIn("linear_extrude (height=10)", out)
        self.assertIn("circle (r=3)", out)

    def test_linear_extrude_twist_radians(self):
        out = write_scad(extrude_linear({"height": 10, "twist": math.pi}, square(2, 2, center=False)))
        self.assertIn("twist=180", out)

    def test_rotate_extrude(self):
        out = write_scad(extrude_rotate({"angle": 270}, translate([5, 0, 0], circle(1))))
        self.assertIn("rotate_extrude (angle=270)", out)

    def test_render(self):
        self.assertIn("render (convexity=2)", write_scad(render(2, sphere(1))))


class ModifierTest(unittest.TestCase):
    def test_background_modifier(self):
        out = write_scad(background(sphere(1)))
        self.assertTrue(out.startswith("%union () {\n"))


class PostwalkTest(unittest.TestCase):
    def test_postwalk_scales_all_radii(self):
        tree = union(sphere(2), translate([1, 0, 0], sphere(4)))

        def double_radius(node):
            if isinstance(node, dict) and "r" in node:
                d = dict(node)
                d["r"] = d["r"] * 2
                return d
            return node

        walked = postwalk(double_radius, tree)
        out = write_scad(walked)
        self.assertIn("sphere (r=4)", out)
        self.assertIn("sphere (r=8)", out)

    def test_postwalk_identity(self):
        tree = union(sphere(1), cube(1, 1, 1, center=False))
        self.assertEqual(write_scad(postwalk(lambda x: x, tree)),
                         write_scad(tree))


class RadDegTest(unittest.TestCase):
    def test_round_trip(self):
        self.assertAlmostEqual(rad_to_deg(deg_to_rad(57.0)), 57.0)

    def test_known(self):
        self.assertAlmostEqual(rad_to_deg(math.pi), 180.0)


if __name__ == "__main__":
    unittest.main()
