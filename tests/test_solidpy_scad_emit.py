"""Tests for programs.solidpy_scad_emit."""

import unittest

from harnesscad.domain.geometry.sdf.scadlm_csg_eval import bounds, contains, evaluate_source
from harnesscad.domain.programs.ast.scadlm_ast import parse, unparse
from harnesscad.domain.programs.emit.solidpy_scad_emit import (
    ScadNode,
    background,
    cube,
    cylinder,
    debug,
    difference,
    format_value,
    hole,
    intersection,
    linear_extrude,
    part,
    polygon,
    rotate,
    scad_render,
    sphere,
    translate,
    union,
    up,
)


class TestFormatValue(unittest.TestCase):
    def test_scalars(self):
        self.assertEqual(format_value(True), "true")
        self.assertEqual(format_value(False), "false")
        self.assertEqual(format_value(3), "3")
        self.assertEqual(format_value(2.5), "2.5")
        self.assertEqual(format_value(2.0), "2")
        self.assertEqual(format_value(-0.0), "0")
        self.assertEqual(format_value("a"), '"a"')

    def test_nested_lists(self):
        self.assertEqual(format_value([1, [2, 3.5]]), "[1, [2, 3.5]]")

    def test_string_escaping(self):
        self.assertEqual(format_value('a"b'), '"a\\"b"')


class TestEmit(unittest.TestCase):
    def test_leaf(self):
        self.assertEqual(scad_render(cube(2)), "cube(size = 2);\n")

    def test_param_order_is_deterministic(self):
        a = scad_render(cylinder(h=2, r=1, center=True))
        b = scad_render(cylinder(center=True, r=1, h=2))
        self.assertEqual(a, b)
        self.assertEqual(a, "cylinder(center = true, h = 2, r = 1);\n")

    def test_segments_becomes_fn(self):
        self.assertIn("$fn = 12", scad_render(sphere(r=1, segments=12)))

    def test_none_params_dropped(self):
        self.assertEqual(scad_render(sphere(r=1)), "sphere(r = 1);\n")

    def test_children_indented(self):
        src = scad_render(translate((1, 0, 0))(cube(2)))
        self.assertEqual(
            src, "translate(v = [1, 0, 0]) {\n    cube(size = 2);\n}\n"
        )

    def test_operator_sugar(self):
        src = scad_render(cube(2) - sphere(r=1))
        self.assertTrue(src.startswith("difference() {"))
        self.assertIn("cube(size = 2);", src)
        self.assertIn("sphere(r = 1);", src)

    def test_intersection_operator(self):
        self.assertTrue(scad_render(cube(2) * sphere(r=1)).startswith("intersection()"))

    def test_sum_of_nodes(self):
        src = scad_render(sum([cube(1), cube(2)]))
        self.assertTrue(src.startswith("union() {"))

    def test_modifiers(self):
        self.assertTrue(scad_render(debug(cube(1))).startswith("#cube"))
        self.assertTrue(scad_render(background(cube(1))).startswith("%cube"))

    def test_copy_is_deep(self):
        a = translate((1, 2, 3))(cube(1))
        b = a.copy()
        b.children[0].params["size"] = 9
        self.assertEqual(a.children[0].params["size"], 1)

    def test_polygon_paths(self):
        src = scad_render(polygon([(0, 0), (1, 0), (1, 1)], paths=[[0, 1, 2]]))
        self.assertIn("paths = [[0, 1, 2]]", src)
        self.assertIn("points = [[0, 0], [1, 0], [1, 1]]", src)

    def test_reserved_word_param(self):
        node = ScadNode("offset", {"r": 1, "for_": 2})
        self.assertIn("for = 2", scad_render(node))

    def test_bad_child(self):
        with self.assertRaises(TypeError):
            cube(1).add(3)


class TestHoles(unittest.TestCase):
    def test_hole_is_lifted_to_root(self):
        body = cube(10, center=True) + hole()(cylinder(r=1, h=20, center=True))
        src = scad_render(body)
        self.assertTrue(src.startswith("difference() {"))
        # The cylinder appears exactly once, in the subtracted branch
        self.assertEqual(src.count("cylinder"), 1)
        self.assertEqual(src.count("cube"), 1)

    def test_hole_survives_later_union(self):
        # The whole point of holes: a later union must not fill the void.
        shell = cube(10, center=True) + hole()(cylinder(r=1, h=20, center=True))
        model = union()(shell, up(20)(cube(1)))
        tree = evaluate_source(scad_render(model))
        self.assertFalse(contains(tree, (0, 0, 0)))
        self.assertTrue(contains(tree, (4, 4, 0)))

    def test_boolean_on_hole_path_becomes_union(self):
        inner = difference()(cube(10), hole()(sphere(r=1)))
        src = scad_render(union()(inner))
        # The hole branch (everything after the positive cube) must not re-emit
        # the enclosing difference: an intersection/difference cannot shrink a void.
        after_cube = src.split("cube", 1)[1]
        self.assertNotIn("difference", after_cube.split("sphere")[0])

    def test_part_resolves_holes_locally(self):
        p = part()(cube(10, center=True) + hole()(cylinder(r=1, h=20, center=True)))
        model = union()(p, translate((20, 0, 0))(cube(2)))
        src = scad_render(model)
        # The outermost call must be the union, not a root-level difference
        self.assertTrue(src.startswith("union() {"))
        self.assertIn("difference()", src)

    def test_no_holes_no_difference(self):
        self.assertNotIn("difference", scad_render(union()(cube(1))))


class TestRoundTripWithScadlm(unittest.TestCase):
    def test_parse_emitted_source(self):
        model = difference()(
            translate((1, 2, 3))(cube([4, 5, 6], center=True)),
            sphere(r=2, segments=16),
        )
        src = scad_render(model)
        nodes = parse(src)
        self.assertEqual(unparse(parse(unparse(nodes))), unparse(nodes))

    def test_evaluate_emitted_geometry(self):
        model = cube([2, 2, 2], center=True) - sphere(r=0.5)
        tree = evaluate_source(scad_render(model))
        self.assertFalse(contains(tree, (0, 0, 0)))
        self.assertTrue(contains(tree, (0.9, 0.9, 0.9)))

    def test_bounds_of_emitted_translate(self):
        tree = evaluate_source(scad_render(translate((5, 0, 0))(cube(2))))
        lo, hi = bounds(tree)
        self.assertAlmostEqual(lo[0], 5.0)
        self.assertAlmostEqual(hi[0], 7.0)

    def test_rotate_round_trip(self):
        src = scad_render(rotate(a=90, v=(0, 0, 1))(cube([2, 1, 1])))
        tree = evaluate_source(src)
        lo, hi = bounds(tree)
        self.assertAlmostEqual(hi[1], 2.0, places=6)

    def test_unsupported_geometry_still_parses(self):
        src = scad_render(linear_extrude(height=3)(polygon([(0, 0), (1, 0), (1, 1)])))
        self.assertEqual(len(parse(src)), 1)

    def test_render_is_stable(self):
        model = intersection()(cube(3), sphere(r=2))
        self.assertEqual(scad_render(model), scad_render(model))


if __name__ == "__main__":
    unittest.main()
