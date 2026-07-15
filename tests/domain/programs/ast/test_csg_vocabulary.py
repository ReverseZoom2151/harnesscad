"""Tests for the cross-family CSG vocabulary superset + hull/minkowski algorithms."""

import unittest

from harnesscad.domain.programs.ast.csg_vocabulary import (
    DIALECTS,
    FAMILIES,
    CsgOp,
    canonicalise,
    convex_hull_2d,
    coverage,
    families_supporting,
    minkowski_sum_2d,
    parse_call,
    polygon_area,
    spelling,
)


class TestVocabulary(unittest.TestCase):
    def test_boolean_aliases_canonicalise_across_families(self):
        self.assertIs(canonicalise("union", "openscad"), CsgOp.UNION)
        self.assertIs(canonicalise("fuse", "replicad"), CsgOp.UNION)
        self.assertIs(canonicalise("subtract", "openjscad"), CsgOp.DIFFERENCE)
        self.assertIs(canonicalise("cut", "replicad"), CsgOp.DIFFERENCE)
        self.assertIs(canonicalise("intersect", "openjscad"), CsgOp.INTERSECTION)

    def test_operator_spelling_for_angelcad(self):
        self.assertIs(canonicalise("+", "angelcad"), CsgOp.UNION)
        self.assertIs(canonicalise("-", "angelcad"), CsgOp.DIFFERENCE)
        self.assertIs(canonicalise("*", "angelcad"), CsgOp.INTERSECTION)

    def test_unknown_name_returns_none(self):
        self.assertIsNone(canonicalise("frobnicate", "openscad"))

    def test_unknown_family_raises(self):
        with self.assertRaises(KeyError):
            canonicalise("union", "solidpython")

    def test_spelling_inverse(self):
        self.assertEqual(spelling(CsgOp.UNION, "replicad"), ("fuse",))
        self.assertIn("cuboid", spelling(CsgOp.CUBE, "openjscad"))

    def test_family_without_op_has_empty_spelling(self):
        # replicad has no polyhedron primitive in our table.
        self.assertEqual(spelling(CsgOp.POLYHEDRON, "replicad"), ())

    def test_hull_supported_everywhere(self):
        self.assertEqual(set(families_supporting(CsgOp.HULL)), set(FAMILIES))

    def test_minkowski_supported_by_subset(self):
        fams = set(families_supporting(CsgOp.MINKOWSKI))
        self.assertIn("openscad", fams)
        self.assertNotIn("replicad", fams)  # OCCT B-rep API has no minkowski

    def test_coverage_counts_consistent(self):
        cov = coverage()
        self.assertEqual(set(cov), set(FAMILIES))
        # every family covers at least the booleans + transforms.
        for fam in FAMILIES:
            self.assertGreaterEqual(cov[fam], 8)

    def test_reverse_index_matches_forward_table(self):
        for op, per_family in DIALECTS.items():
            for family, spellings in per_family.items():
                for name in spellings:
                    self.assertIs(canonicalise(name, family), op)


class TestParseCall(unittest.TestCase):
    def test_parses_call_with_args(self):
        res = parse_call("translate([1,2,3])", "openscad")
        self.assertIsNotNone(res)
        op, args = res
        self.assertIs(op, CsgOp.TRANSLATE)
        self.assertEqual(args, ["[1,2,3]"])

    def test_top_level_comma_split_respects_nesting(self):
        op, args = parse_call("hull(cube(2), sphere(1))", "openscad")
        self.assertIs(op, CsgOp.HULL)
        self.assertEqual(args, ["cube(2)", "sphere(1)"])

    def test_empty_args(self):
        op, args = parse_call("union()", "openjscad")
        self.assertIs(op, CsgOp.UNION)
        self.assertEqual(args, [])

    def test_replicad_method_name(self):
        op, args = parse_call("fuse(other);", "replicad")
        self.assertIs(op, CsgOp.UNION)
        self.assertEqual(args, ["other"])

    def test_non_call_returns_none(self):
        self.assertIsNone(parse_call("x = 5", "openscad"))

    def test_unknown_op_returns_none(self):
        self.assertIsNone(parse_call("wibble(1)", "openscad"))


class TestConvexHull(unittest.TestCase):
    def test_square_with_interior_point(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4), (2, 2)]  # centre must be dropped
        hull = convex_hull_2d(pts)
        self.assertEqual(set(hull), {(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)})
        self.assertEqual(len(hull), 4)

    def test_hull_is_ccw(self):
        hull = convex_hull_2d([(0, 0), (4, 0), (4, 4), (0, 4)])
        self.assertGreater(polygon_area(hull), 0.0)  # CCW => positive area
        self.assertAlmostEqual(polygon_area(hull), 16.0)

    def test_collinear_points_dropped(self):
        pts = [(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]  # (1,0) is on an edge
        hull = convex_hull_2d(pts)
        self.assertNotIn((1.0, 0.0), hull)
        self.assertEqual(len(hull), 4)

    def test_degenerate_two_points(self):
        self.assertEqual(convex_hull_2d([(1, 1), (0, 0)]), [(0.0, 0.0), (1.0, 1.0)])

    def test_deterministic(self):
        pts = [(3, 1), (0, 0), (4, 4), (1, 3), (2, 2)]
        self.assertEqual(convex_hull_2d(pts), convex_hull_2d(pts))


class TestMinkowski(unittest.TestCase):
    def test_square_plus_square_doubles_extent(self):
        # unit square (+) unit square = 2x2 square.
        a = [(0, 0), (1, 0), (1, 1), (0, 1)]
        b = [(0, 0), (1, 0), (1, 1), (0, 1)]
        s = minkowski_sum_2d(a, b)
        self.assertEqual(set(s), {(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)})
        self.assertAlmostEqual(polygon_area(s), 4.0)

    def test_translated_operand_shifts_result(self):
        a = [(0, 0), (2, 0), (2, 2), (0, 2)]
        b = [(10, 10)]  # a single point => pure translation
        s = minkowski_sum_2d(a, b)
        self.assertEqual(set(s), {(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)})

    def test_triangle_plus_segment(self):
        tri = [(0, 0), (2, 0), (0, 2)]
        seg = [(0, 0), (1, 0)]
        s = minkowski_sum_2d(tri, seg)
        # area grows by the swept band; result is convex and area > triangle's 2.0.
        self.assertGreater(polygon_area(s), 2.0)

    def test_empty_operand(self):
        self.assertEqual(minkowski_sum_2d([], [(0, 0)]), [])

    def test_deterministic(self):
        a = [(0, 0), (1, 0), (0, 1)]
        b = [(0, 0), (1, 1)]
        self.assertEqual(minkowski_sum_2d(a, b), minkowski_sum_2d(a, b))


if __name__ == "__main__":
    unittest.main()
