"""Tests for reconstruction.brepground_consistency."""

import unittest

from harnesscad.domain.reconstruction.translate.brep_grounding import BRepPrimitive
from harnesscad.domain.reconstruction.translate.grounding_consistency import (
    ConsistencyReport,
    FeatureStep,
    check_program,
    check_step,
    query_specificity,
    uses_grounding,
)


def _brep():
    return [
        BRepPrimitive(0, "face", "planar", (0.0, 0.0, 5.0), 100.0),   # top
        BRepPrimitive(1, "face", "planar", (0.0, 0.0, 0.0), 100.0),   # bottom
        BRepPrimitive(2, "edge", "line", (0.0, 3.0, 2.5), 10.0),      # back edge
        BRepPrimitive(3, "edge", "line", (0.0, -3.0, 2.5), 10.0),     # front edge
        BRepPrimitive(4, "edge", "circle", (1.0, 1.0, 5.0), 6.0,
                      is_hole=True),                                   # hole edge
    ]


class TestCheckStep(unittest.TestCase):
    def setUp(self):
        self.brep = _brep()

    def test_sketch_without_query_ok(self):
        r = check_step(FeatureStep("sketch"))
        self.assertTrue(r.ok)

    def test_extrude_without_query_ok(self):
        self.assertTrue(check_step(FeatureStep("extrude")).ok)

    def test_fillet_without_query_fails(self):
        r = check_step(FeatureStep("fillet", query=None, available=self.brep))
        self.assertFalse(r.ok)
        self.assertIn("requires an operand", r.reason)

    def test_fillet_on_edges_ok(self):
        r = check_step(
            FeatureStep("fillet", query="all straight edges", available=self.brep)
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.grounded, (2, 3))

    def test_fillet_grounding_wrong_kind_fails(self):
        # fillet needs edges; a query resolving to a face is inconsistent.
        r = check_step(
            FeatureStep("fillet", query="the top face", available=self.brep)
        )
        self.assertFalse(r.ok)
        self.assertIn("expects edge", r.reason)

    def test_shell_on_face_ok(self):
        r = check_step(
            FeatureStep("shell", query="the top face", available=self.brep)
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.grounded, (0,))

    def test_dangling_query_fails(self):
        r = check_step(
            FeatureStep("fillet", query="all spherical edges", available=self.brep)
        )
        self.assertFalse(r.ok)
        self.assertIn("no primitive", r.reason)

    def test_singular_overselection_fails(self):
        r = check_step(
            FeatureStep("shell", query="all faces", available=self.brep,
                        singular=True)
        )
        self.assertFalse(r.ok)
        self.assertIn("ambiguous", r.reason)

    def test_singular_unique_ok(self):
        r = check_step(
            FeatureStep("shell", query="the top face", available=self.brep,
                        singular=True)
        )
        self.assertTrue(r.ok)


class TestCheckProgram(unittest.TestCase):
    def test_valid_program(self):
        brep = _brep()
        steps = [
            FeatureStep("sketch"),
            FeatureStep("extrude"),
            FeatureStep("fillet", query="all straight edges", available=brep),
        ]
        rep = check_program(steps)
        self.assertIsInstance(rep, ConsistencyReport)
        self.assertTrue(rep.ok)
        self.assertEqual(rep.failures, ())

    def test_invalid_program_reports_failure(self):
        brep = _brep()
        steps = [
            FeatureStep("sketch"),
            FeatureStep("fillet", query="the top face", available=brep),
        ]
        rep = check_program(steps)
        self.assertFalse(rep.ok)
        self.assertEqual(len(rep.failures), 1)
        self.assertEqual(rep.failures[0].feature, "fillet")

    def test_empty_program_ok(self):
        self.assertTrue(check_program([]).ok)


class TestSpecificity(unittest.TestCase):
    def setUp(self):
        self.brep = _brep()

    def test_no_query_is_one(self):
        self.assertEqual(query_specificity(FeatureStep("extrude")), 1.0)

    def test_unique_is_one(self):
        s = FeatureStep("shell", query="the top face", available=self.brep)
        self.assertEqual(query_specificity(s), 1.0)

    def test_broad_query_lower(self):
        s = FeatureStep("fillet", query="all edges", available=self.brep)
        self.assertLess(query_specificity(s), 1.0)

    def test_dangling_is_zero(self):
        s = FeatureStep("fillet", query="spherical edges", available=self.brep)
        self.assertEqual(query_specificity(s), 0.0)

    def test_empty_brep_zero(self):
        s = FeatureStep("fillet", query="all edges", available=[])
        self.assertEqual(query_specificity(s), 0.0)


class TestHelpers(unittest.TestCase):
    def test_uses_grounding(self):
        self.assertTrue(uses_grounding("fillet"))
        self.assertTrue(uses_grounding("chamfer"))
        self.assertTrue(uses_grounding("shell"))
        self.assertFalse(uses_grounding("extrude"))


if __name__ == "__main__":
    unittest.main()
