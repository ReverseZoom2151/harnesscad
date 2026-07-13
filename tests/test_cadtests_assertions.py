"""Tests for bench.cadtests_assertions (CADTEST assertion primitives)."""

import unittest

from harnesscad.eval.bench.protocols.cadtests_assertions import (
    CATEGORIES,
    CadTest,
    GEOMETRIC_TYPES,
    SPATIAL,
    TestResult,
    assert_aspect_ratio,
    assert_bbox_dimension,
    assert_center_of_mass,
    assert_coaxial,
    assert_edge_count,
    assert_face_count,
    assert_fill_factor,
    assert_has_geometry_type,
    assert_largest_axis,
    assert_more_faces_than,
    assert_no_geometry_type,
    assert_num_solids,
    assert_symmetry,
    assert_typed_face_count,
    assert_valid_solid,
    assert_volume,
)
from harnesscad.eval.bench.data.cadtests_model import CADModel, Edge, Face


def _box():
    faces = tuple(Face("plane", a) for a in (2.0, 2.0, 3.0, 3.0, 6.0, 6.0))
    edges = tuple(Edge("line", 1.0) for _ in range(12))
    return CADModel(faces, edges, 8, (0.0, 0.0, 0.0), (2.0, 3.0, 1.0), 6.0,
                    (1.0, 1.5, 0.5))


def _cyl():
    faces = (Face("plane", 4.0), Face("plane", 4.0), Face("cylinder", 3.0),
             Face("cylinder", 3.0))
    edges = (Edge("circle", 3.14), Edge("circle", 3.14))
    return CADModel(faces, edges, 8, (0.0, 0.0, 0.0), (4.0, 4.0, 2.0), 28.0,
                    (2.0, 2.0, 1.0))


class TestValidityAsserts(unittest.TestCase):
    def test_valid_solid_pass_fail(self):
        self.assertTrue(assert_valid_solid().evaluate(_box()).passed)
        two = CADModel(_box().faces, _box().edges, 8, (0, 0, 0), (2, 3, 1), 6.0,
                       (1, 1.5, 0.5), solids=2)
        r = assert_valid_solid().evaluate(two)
        self.assertFalse(r.passed)
        self.assertIn("not a valid solid", r.message)

    def test_num_solids_exact_and_range(self):
        self.assertTrue(assert_num_solids(1).evaluate(_box()).passed)
        self.assertFalse(assert_num_solids(2).evaluate(_box()).passed)
        self.assertTrue(assert_num_solids(at_least=1, at_most=3)
                        .evaluate(_box()).passed)


class TestTopologyAsserts(unittest.TestCase):
    def test_face_edge_counts(self):
        self.assertTrue(assert_face_count(6).evaluate(_box()).passed)
        self.assertTrue(assert_edge_count(12).evaluate(_box()).passed)
        self.assertFalse(assert_face_count(5).evaluate(_box()).passed)

    def test_range_count(self):
        self.assertTrue(assert_face_count(at_least=4, at_most=8)
                        .evaluate(_box()).passed)

    def test_typed_face_count(self):
        self.assertTrue(assert_typed_face_count("cylinder", 2)
                        .evaluate(_cyl()).passed)
        self.assertFalse(assert_typed_face_count("cylinder", 1)
                         .evaluate(_cyl()).passed)

    def test_more_faces_than(self):
        self.assertTrue(assert_more_faces_than(4).evaluate(_box()).passed)
        self.assertFalse(assert_more_faces_than(6).evaluate(_box()).passed)


class TestGeometricTypeAsserts(unittest.TestCase):
    def test_presence(self):
        self.assertTrue(assert_has_geometry_type("cylinder")
                        .evaluate(_cyl()).passed)
        self.assertTrue(assert_has_geometry_type("circle", edge=True)
                        .evaluate(_cyl()).passed)
        self.assertFalse(assert_has_geometry_type("sphere")
                         .evaluate(_cyl()).passed)

    def test_absence(self):
        self.assertTrue(assert_no_geometry_type("sphere").evaluate(_box()).passed)
        self.assertFalse(assert_no_geometry_type("plane").evaluate(_box()).passed)

    def test_category(self):
        self.assertEqual(assert_has_geometry_type("plane").category,
                         GEOMETRIC_TYPES)


class TestDimensionAsserts(unittest.TestCase):
    def test_bbox_dimension(self):
        self.assertTrue(assert_bbox_dimension("y", 3.0).evaluate(_box()).passed)
        self.assertFalse(assert_bbox_dimension("y", 5.0).evaluate(_box()).passed)

    def test_tolerance(self):
        self.assertTrue(assert_bbox_dimension("y", 3.05, abs_tol=0.1)
                        .evaluate(_box()).passed)

    def test_aspect_ratio(self):
        self.assertTrue(assert_aspect_ratio("y", "x", 1.5)
                        .evaluate(_box()).passed)

    def test_largest_axis(self):
        self.assertTrue(assert_largest_axis("y").evaluate(_box()).passed)
        self.assertFalse(assert_largest_axis("z").evaluate(_box()).passed)


class TestVolumetricAsserts(unittest.TestCase):
    def test_volume(self):
        self.assertTrue(assert_volume(6.0).evaluate(_box()).passed)
        self.assertFalse(assert_volume(10.0).evaluate(_box()).passed)

    def test_rel_tol(self):
        self.assertTrue(assert_volume(6.3, rel_tol=0.1).evaluate(_box()).passed)

    def test_fill_factor(self):
        self.assertTrue(assert_fill_factor(1.0).evaluate(_box()).passed)
        self.assertTrue(assert_fill_factor(28.0 / 32.0).evaluate(_cyl()).passed)


class TestSpatialAsserts(unittest.TestCase):
    def test_center_of_mass(self):
        self.assertTrue(assert_center_of_mass((1.0, 1.5, 0.5))
                        .evaluate(_box()).passed)
        self.assertFalse(assert_center_of_mass((0.0, 0.0, 0.0))
                         .evaluate(_box()).passed)

    def test_coaxial(self):
        # box com equals bbox center -> concentric
        self.assertTrue(assert_coaxial().evaluate(_box()).passed)

    def test_symmetry(self):
        self.assertTrue(assert_symmetry("x").evaluate(_box()).passed)
        # shift com off-center along x
        off = CADModel(_box().faces, _box().edges, 8, (0, 0, 0), (2, 3, 1), 6.0,
                       (1.9, 1.5, 0.5))
        self.assertFalse(assert_symmetry("x").evaluate(off).passed)

    def test_category(self):
        self.assertEqual(assert_coaxial().category, SPATIAL)


class TestRuntimeErrorInvalidity(unittest.TestCase):
    def test_runtime_error_marks_invalid(self):
        # aspect ratio with a zero-extent denominator raises -> invalid test
        flat = CADModel(_box().faces, _box().edges, 8, (0, 0, 0), (2, 3, 0.0),
                        0.0, (1, 1.5, 0.0))
        r = assert_aspect_ratio("x", "z", 1.0).evaluate(flat)
        self.assertFalse(r.passed)
        self.assertFalse(r.valid)
        self.assertIsNotNone(r.error)

    def test_valid_test_has_no_error(self):
        r = assert_volume(6.0).evaluate(_box())
        self.assertTrue(r.valid)
        self.assertIsNone(r.error)


class TestSchema(unittest.TestCase):
    def test_bad_category_raises(self):
        with self.assertRaises(ValueError):
            CadTest("x", "bogus", "d", lambda m: True)

    def test_categories_complete(self):
        self.assertEqual(len(CATEGORIES), 6)

    def test_requirement_propagates(self):
        t = assert_volume(6.0, requirement="R1")
        self.assertEqual(t.requirement, "R1")
        self.assertEqual(t.evaluate(_box()).requirement, "R1")

    def test_bool_only_predicate_default_message(self):
        t = CadTest("t", SPATIAL, "always true", lambda m: True)
        r = t.evaluate(_box())
        self.assertTrue(r.passed)
        self.assertIn("PASS", r.message)


if __name__ == "__main__":
    unittest.main()
