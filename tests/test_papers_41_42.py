"""Tests for explicit CISP context handles, code-error normalisation, the CAD
API knowledge base, correction trajectories, primitive relations/intersections/
stitching, view coverage, and primitive reconstruction metrics.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import unittest

from bench.correction_trajectory import score as trajectory_score
from bench.primitive_reconstruction_metrics import metrics as reconstruction_metrics
from cisp.explicit_context import Context
from quality.view_coverage import audit as view_audit
from rag.cad_api_knowledge import API, chunks, validate as validate_apis
from reconstruction.primitive_intersections import assemble
from reconstruction.primitive_relations import Primitive, infer, project
from reconstruction.primitive_stitch import stitch
from reliability.code_error import normalize


class ExplicitContextTest(unittest.TestCase):
    def test_require_succeeds_for_a_live_handle(self):
        context = Context()
        handle = context.bind("x", "face")
        context.require(handle, "face")

    def test_rollback_invalidates_handles_bound_after_the_snapshot(self):
        context = Context()
        handle = context.bind("x", "face")
        snapshot = context.snapshot()
        context.require(handle, "face")
        context.rollback(snapshot)
        with self.assertRaises(ValueError):
            context.require(handle, "face")


class CodeErrorTest(unittest.TestCase):
    def test_type_error_normalizes_to_the_type_category(self):
        self.assertEqual(normalize(TypeError(), "x").category, "type")


class CADAPIKnowledgeTest(unittest.TestCase):
    def test_self_consistent_api_passes_validation(self):
        self.assertFalse(validate_apis((API("x", "x()", "face", ("x",)),)))

    def test_chunks_are_emitted_for_an_api(self):
        self.assertTrue(chunks((API("x", "x()", "face", ()),)))


class CorrectionTrajectoryTest(unittest.TestCase):
    def test_invalid_then_valid_trajectory_counts_as_recovered(self):
        self.assertTrue(trajectory_score(({"valid": False}, {"valid": True}))["recovered"])


class PrimitiveRelationsTest(unittest.TestCase):
    def test_shared_axis_is_inferred_as_parallel(self):
        a = Primitive("a", (1, 0, 0))
        b = Primitive("b", (1, 0, 0))
        self.assertEqual(infer(a, b), "parallel")

    def test_projecting_a_parallel_relation_snaps_the_axis(self):
        a = Primitive("a", (1, 0, 0))
        b = Primitive("b", (1, 0, 0))
        self.assertEqual(project(a, b, "parallel")[1].axis, a.axis)


class PrimitiveIntersectionsTest(unittest.TestCase):
    def test_fully_connected_triple_assembles_three_edges(self):
        primitives = (Primitive("a", (1, 0, 0)),
                      Primitive("b", (1, 0, 0)),
                      Primitive("c", (0, 1, 0)))
        result = assemble(primitives,
                          {("a", "b"), ("b", "c"), ("a", "c")},
                          lambda a, b: (a.id, b.id),
                          lambda a, b, c: (0, 0, 0))
        self.assertEqual(len(result["edges"]), 3)


class PrimitiveStitchTest(unittest.TestCase):
    def test_stitching_converges_to_a_negligible_residual(self):
        self.assertLess(stitch(4, lambda x: x / 2, abs)["residual"], 1e-5)


class ViewCoverageTest(unittest.TestCase):
    def test_view_exposing_the_missing_entity_is_recommended(self):
        result = view_audit({"a", "b"},
                            ({"id": "v", "visible": {"a"}, "potential": {"b"}},))
        self.assertEqual(result["recommendation"], "v")


class PrimitiveReconstructionMetricsTest(unittest.TestCase):
    def test_matching_normals_score_perfect_consistency(self):
        result = reconstruction_metrics((((1, 0, 0), (1, 0, 0)),), (), (True,), 0, 1)
        self.assertEqual(result["normal_consistency"], 1)


if __name__ == "__main__":
    unittest.main()
