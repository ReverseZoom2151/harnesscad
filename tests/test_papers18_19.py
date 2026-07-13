import unittest

from harnesscad.eval.bench.data.brep_robustness import mask_cases, evaluate_masking
from harnesscad.eval.bench.data.brep_splits import complexity, grouped_split
from harnesscad.eval.bench.harness.resource_tradeoff import ResourceResult, pareto_frontier
from harnesscad.eval.bench.vision.segmentation_metrics import face_segmentation_metrics
from harnesscad.io.ingest.bezier_contracts import bezier_curve, bezier_triangle
from harnesscad.io.ingest.brep_hierarchy import (
    BRepHierarchy, Coedge, Edge, Face, Loop, Shell, Vertex,
)
from harnesscad.io.ingest.brep_tokens import canonical_cycle, loop_tokens
from harnesscad.io.ingest.spatial_order import morton2, patch_order
from harnesscad.io.ingest.tokenization_audit import audit_tokenization
from harnesscad.eval.quality.graph.brep_descriptors import aggregate, hierarchy_descriptors
from harnesscad.governance.research.model_promotion import promotion_gate
from harnesscad.governance.research.resource_profile import profile
from harnesscad.domain.vision.embedding_cache import EmbeddingCache, embedding_key
from harnesscad.domain.vision.residual_guard import guard_residual


def hierarchy():
    vertices = (
        Vertex("v0", (0, 0, 0)), Vertex("v1", (1, 0, 0)),
        Vertex("v2", (0, 1, 0)),
    )
    edges = (Edge("e0", "v0", "v1", (1,)), Edge("e1", "v1", "v2", (2,)),
             Edge("e2", "v2", "v0", (3,)))
    coedges = tuple(
        [Coedge(f"ca{i}", f"e{i}", True) for i in range(3)]
        + [Coedge(f"cb{i}", f"e{i}", False) for i in reversed(range(3))]
    )
    loops = (Loop("la", ("ca0", "ca1", "ca2"), True),
             Loop("lb", ("cb2", "cb1", "cb0"), True))
    faces = (Face("fa", ("la",), (1,)), Face("fb", ("lb",), (2,)))
    return BRepHierarchy(vertices, edges, coedges, loops, faces,
                         (Shell("s", ("fa", "fb")),))


class HierarchyTests(unittest.TestCase):
    def test_integrity_manifold_and_neighbors(self):
        value = hierarchy()
        self.assertEqual(value.validate(manifold=True), ())
        self.assertEqual(value.face_neighbors(), {"fa": ("fb",), "fb": ("fa",)})

    def test_integrity_catches_broken_loop_and_outer_contract(self):
        value = hierarchy()
        broken = BRepHierarchy(
            value.vertices, value.edges, value.coedges,
            (Loop("la", ("ca1", "ca0"), False),), (Face("f", ("la",)),),
            (Shell("s", ("missing",)),))
        issues = broken.validate()
        self.assertIn("loop-la-not-closed", issues)
        self.assertIn("face-f-outer-loop-count", issues)
        self.assertIn("shell-s-unknown-face", issues)

    def test_canonical_cycle_and_wrap_are_reproducible(self):
        self.assertEqual(canonical_cycle(("b", "c", "a")), ("a", "b", "c"))
        self.assertEqual(canonical_cycle(("c", "b", "a"), orientation_semantic=False),
                         ("a", "b", "c"))
        self.assertEqual(loop_tokens(("b", "c", "a")), ("c", "a", "b", "c", "a"))

    def test_hierarchical_descriptors_and_aggregation(self):
        self.assertEqual(aggregate(((1, 4), (3, 2))), (2, 3, 3, 4))
        result = hierarchy_descriptors(hierarchy(), {})
        self.assertEqual(set(result), {"edges", "loops", "faces"})
        self.assertEqual(set(result["faces"]), {"fa", "fb"})
        self.assertTrue(result["faces"]["fa"])


class GeometryTokenTests(unittest.TestCase):
    def test_morton_and_patch_order(self):
        self.assertEqual([morton2(x, y, bits=2)
                          for x, y in ((0, 0), (0, 1), (1, 0), (1, 1))],
                         [0, 1, 2, 3])
        patches = [{"depth": 2, "x": 1, "y": 0, "triangle": 1},
                   {"depth": 2, "x": 0, "y": 1, "triangle": 0}]
        self.assertEqual(patch_order(reversed(patches)), tuple(reversed(patches)))

    def test_bezier_analytic_utilities(self):
        self.assertEqual(bezier_curve(((0, 0), (2, 2)), 0.25), (0.5, 0.5))
        control = {(0, 0): (0, 0), (1, 0): (1, 0), (0, 1): (0, 1)}
        self.assertEqual(bezier_triangle(control, 1, .25, .5), (.25, .5))

    def test_tokenization_audit_never_silently_truncates(self):
        report = audit_tokenization(
            reference_points=((0, 0),), encoded_points=((0.1, 0),),
            joins=(((0, 0), (0.2, 0)),), trim_deviation=.3,
            segment_count=101, max_segments=100, tolerance=.01)
        self.assertEqual(set(report.issues), {
            "sequence-overflow", "geometry-deviation", "trim-deviation",
            "continuity-gap",
        })


class EvaluationTests(unittest.TestCase):
    def test_seeded_masking(self):
        self.assertEqual(mask_cases(("a", "b", "c", "d"), seed=7),
                         mask_cases(("d", "c", "b", "a"), seed=7))
        report = evaluate_masking(("a", "b"), lambda kept: bool(kept), seed=1)
        self.assertTrue(report["cases"][1]["stable"])

    def test_segmentation_accuracy_iou_and_id_gate(self):
        result = face_segmentation_metrics(
            {"f1": "side", "f2": "end"}, {"f1": "side", "f2": "side"})
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["classes"]["side"]["iou"], .5)
        self.assertFalse(face_segmentation_metrics({"a": "x"}, {})["available"])

    def test_complexity_and_grouped_split_no_family_leak(self):
        self.assertEqual(complexity(face_count=10, planar_faces=10,
                                    trimmed_faces=0, curve_segments=(2,)).stratum,
                         "simple")
        self.assertEqual(complexity(face_count=10, planar_faces=5,
                                    trimmed_faces=6, curve_segments=()).stratum,
                         "hard")
        records = ({"id": "a", "family": "same"}, {"id": "b", "family": "same"},
                   {"id": "c", "family": "other"})
        split = grouped_split(records)
        locations = [{item["id"] for item in values} for values in split.values()]
        self.assertTrue(any({"a", "b"} <= values for values in locations))


class ResourceAndVisionTests(unittest.TestCase):
    def test_resource_profile_stops_on_success(self):
        class Sampler:
            def start(self): self.started = True
            def stop(self): return {"peak_memory_bytes": 12, "elapsed_seconds": .5}
        result, report = profile(lambda: 42, Sampler(), batch_size=2)
        self.assertEqual(result, 42)
        self.assertEqual((report.peak_memory_bytes, report.batch_size), (12, 2))

    def test_resource_pareto_drops_dominated_and_oom(self):
        good = ResourceResult("good", .9, 10, 1)
        dominated = ResourceResult("bad", .8, 12, 2)
        oom = ResourceResult("oom", 1, 1, 1, oom=True)
        self.assertEqual(pareto_frontier((dominated, oom, good)), (good,))

    def test_embedding_cache_keys_provenance(self):
        calls = []
        cache = EmbeddingCache()
        encoder = lambda data: calls.append(data) or (len(data),)
        first = cache.get_or_compute(b"x", encoder, checkpoint="a",
                                     preprocessing={"size": 2})
        second = cache.get_or_compute(b"x", encoder, checkpoint="a",
                                      preprocessing={"size": 2})
        third = cache.get_or_compute(b"x", encoder, checkpoint="b",
                                     preprocessing={"size": 2})
        self.assertFalse(first[1])
        self.assertTrue(second[1])
        self.assertFalse(third[1])
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(first[2], third[2])

    def test_residual_guard_bounds_and_finiteness(self):
        self.assertEqual(guard_residual((1, 1), (1, -1))[0], (1.1, .9))
        self.assertEqual(guard_residual((0,), (2,))[1], ("residual-bound",))
        self.assertEqual(guard_residual((0,), (2,), policy="clip")[0], (.1,))
        self.assertEqual(guard_residual((0,), (float("nan"),))[1],
                         ("non-finite-residual",))

    def test_model_promotion_requires_quality_memory_and_evidence(self):
        accepted = promotion_gate(
            baseline_quality=.7, candidate_quality=.8, candidate_peak_memory=10,
            memory_ceiling=12, minimum_improvement=.05, evidence_count=20)
        self.assertTrue(accepted.promoted)
        rejected = promotion_gate(
            baseline_quality=.7, candidate_quality=.71, candidate_peak_memory=20,
            memory_ceiling=12, minimum_improvement=.05, evidence_count=0,
            minimum_evidence=2)
        self.assertEqual(set(rejected.reasons), {
            "insufficient-evidence", "quality-threshold", "memory-ceiling"})


if __name__ == "__main__":
    unittest.main()
