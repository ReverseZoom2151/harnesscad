"""Tests for geometry-prompted segmentation, domain randomization, instance
matching, point-cloud ingest/candidates/budgets, sketch-boolean recipes, and
the conservative (non-executing) code-quality metrics.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import math
import unittest

from harnesscad.eval.bench.protocols.appearance_invariance import appearance_invariance
from harnesscad.eval.bench.harness.candidate_scaling import candidate_scaling
from harnesscad.eval.bench.data.segmentation_manifests import GeometrySegmentationCase, audit_cases
from harnesscad.eval.bench.vision.instance_segmentation import instance_metrics
from harnesscad.eval.bench.harness.point_budget import point_budget_report
from harnesscad.eval.bench.data.pointcloud_corruption import corrupt_cloud, robustness_curve
from harnesscad.data.dataengine.audit.code_modularity import code_modularity
from harnesscad.data.datagen.domain_randomization import RandomAxis, draw_scene, independence_audit
from harnesscad.data.datagen.reverse_engineering import build_reverse_sample
from harnesscad.data.datagen.sketch_boolean import realize_recipe, sketch_recipe
from harnesscad.io.ingest.fourier_features import fourier_features
from harnesscad.io.ingest.point_cloud import canonicalize_cloud
from harnesscad.eval.quality.graph.abstraction import accept_abstraction, propose_abstraction
from harnesscad.eval.quality.report.parameter_exposure import expose_parameters
from harnesscad.eval.quality.sequence.quantization_risk import quantization_risks
from harnesscad.domain.reconstruction.evaluate.expressivity import expressivity_report
from harnesscad.domain.reconstruction.fitting.pointcloud_candidates import select_pointcloud_candidate
from harnesscad.io.surfaces.canonical_views import canonical_views
from harnesscad.domain.vision.geometry_prompt import GeometryPrompt, PromptView
from harnesscad.domain.vision.instance_matching import mask_iou, mask_nms, one_to_many
from harnesscad.domain.vision.mask_sampling import sample_mask


BOOL_MASK = ((True, False), (False, True))
MASK = ((0, 0), (1, 1))


class GeometryPromptPipelineTest(unittest.TestCase):
    def test_canonical_views_are_deterministic(self):
        self.assertEqual(canonical_views(5), canonical_views(5))

    def test_canonical_view_directions_are_unit_length(self):
        for view in canonical_views(5):
            self.assertTrue(
                math.isclose(sum(x * x for x in view.direction), 1.0))

    def test_prompt_validates_and_has_a_stable_digest(self):
        points = sample_mask(BOOL_MASK, count=5, seed=4)
        prompt = GeometryPrompt("mesh", (
            PromptView("v0", (1.0,), b"rgb", BOOL_MASK, points),
        ))
        self.assertFalse(prompt.validate())
        self.assertEqual(prompt.digest, prompt.digest)


class DomainRandomizationTest(unittest.TestCase):
    AXES = (RandomAxis("light", 1, 2), RandomAxis("texture", 3, 4))

    def test_scene_draw_is_seeded_and_reproducible(self):
        scene = draw_scene(self.AXES, seed=3, identity="part")
        self.assertEqual(scene, draw_scene(self.AXES, seed=3, identity="part"))

    def test_independence_audit_passes_for_distinct_identities(self):
        scenes = (draw_scene(self.AXES, seed=3, identity="part"),
                  draw_scene(self.AXES, seed=4, identity="other"))
        self.assertTrue(independence_audit(scenes))


class InstanceMatchingTest(unittest.TestCase):
    def test_identical_masks_have_unit_iou(self):
        self.assertEqual(mask_iou(MASK, MASK), 1)

    def test_one_to_many_matches_every_prediction(self):
        predictions = ({"mask": MASK, "score": 1.0},
                       {"mask": ((0, 0),), "score": .8})
        matched, selected = one_to_many(predictions, (MASK,), threshold=.4)
        self.assertEqual(len(matched), 2)
        self.assertEqual(selected, (1, 1))

    def test_nms_collapses_duplicate_predictions(self):
        prediction = {"mask": MASK, "score": 1.0}
        self.assertEqual(mask_nms((prediction, prediction)), (0,))

    def test_perfect_prediction_scores_unit_panoptic_quality(self):
        self.assertEqual(instance_metrics((MASK,), (MASK,))["pq"], 1)


class SegmentationManifestTest(unittest.TestCase):
    def test_audit_flags_the_case_without_instances(self):
        cases = (
            GeometrySegmentationCase("a", "i1", "m1", "p", ("x",), "train"),
            GeometrySegmentationCase("b", "i2", "m2", "p", (), "test"),
        )
        self.assertEqual(audit_cases(cases), (("b", "no-instances"),))

    def test_appearance_invariance_suspects_a_shortcut(self):
        result = appearance_invariance(
            ({"id": "a", "geometry": 1, "value": 1},
             {"id": "b", "geometry": 1, "value": 1},
             {"id": "c", "geometry": 2, "value": 9}),
            lambda case: case["value"], lambda a, b: abs(a - b))
        self.assertTrue(result["shortcut_suspected"])


class PointCloudPipelineTest(unittest.TestCase):
    def test_canonicalizing_transform_is_invertible(self):
        cloud, transform = canonicalize_cloud(((0, 0, 0), (2, 2, 2)),
                                              normalize=True)
        self.assertEqual(tuple(transform.invert(point) for point in cloud),
                         ((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)))

    def test_fourier_features_have_the_expected_width(self):
        # 3 coords * 2 bands * 2 (sin, cos) + 3 raw coords = 15
        self.assertEqual(len(fourier_features((.5, .25, 0), (1, 2))), 15)

    def test_candidate_selection_picks_the_first_winner_and_scales(self):
        cloud, _ = canonicalize_cloud(((0, 0, 0), (2, 2, 2)), normalize=True)
        selected = select_pointcloud_candidate(
            cloud, lambda _, seed: seed, lambda x: x,
            lambda shape, count, seed: cloud, count=3)
        self.assertEqual(selected["winner"].index, 0)
        scaling = candidate_scaling(tuple(
            {"valid": item.valid, "distance": item.distance, "cost": 1}
            for item in selected["attempts"]))
        self.assertEqual(scaling["rows"][-1]["invalidity"], 0)


class SketchRecipeTest(unittest.TestCase):
    def test_recipe_realizes_through_a_host_neutral_adapter(self):
        recipe = sketch_recipe(8, minimum=3, maximum=3)

        class Adapter:
            empty = lambda self: []
            primitive = lambda self, kind, params: (kind, params)
            boolean = lambda self, result, shape, operation: result + [(shape, operation)]
            boundary_loops = lambda self, result: (((0, 1),),)
            intersects = lambda self, loops: False
            length = lambda self, edge: 1

        self.assertTrue(realize_recipe(recipe, Adapter())["accepted"])

    def test_reverse_sample_carries_provenance(self):
        recipe = sketch_recipe(8, minimum=3, maximum=3)
        sample = build_reverse_sample(
            "s", recipe, lambda _: ({"op": "box"},), lambda ops: ops,
            repr, lambda shape: shape,
            lambda triangles, count, seed: ((0, 0, 0),),
            provenance={"seed": 8})
        self.assertEqual(sample.provenance["seed"], 8)


class ConservativeQualityMetricsTest(unittest.TestCase):
    OPS = (
        {"op": "add_line", "x1": 0, "y1": 0, "x2": 2, "y2": 0},
        {"op": "add_line", "x1": 2, "y1": 0, "x2": 2, "y2": 1},
        {"op": "add_line", "x1": 2, "y1": 1, "x2": 0, "y2": 1},
        {"op": "add_line", "x1": 0, "y1": 1, "x2": 0, "y2": 0},
        {"op": "extrude", "distance": 3},
    )

    def test_code_modularity_reports_reuse_and_dead_definitions(self):
        source = "w = 2\nbox(w, w, 2)\nunused = 4\n"
        metrics = code_modularity(source)
        self.assertEqual(metrics["reused_variables"], ("w",))
        self.assertEqual(metrics["dead_definitions"], ("unused",))

    def test_closed_rectangle_extrusion_abstracts_to_an_accepted_box(self):
        proposal = propose_abstraction(self.OPS)
        accepted = accept_abstraction(
            self.OPS, proposal, lambda x: x,
            lambda a, b: {"valid": True, "distance": 0}, tolerance=1e-6)
        self.assertEqual(proposal["kind"], "box")
        self.assertTrue(accepted["accepted"])

    def test_parameter_exposure_never_marks_ops_executable(self):
        self.assertIs(expose_parameters(self.OPS)["executable"], False)

    def test_quantization_risks_flags_every_degenerate_dimension(self):
        risks = quantization_risks(step=1, extrusion=.1, radii=(1.1, 1.2),
                                   clearances=(.1,))
        self.assertEqual(set(risks), {"zero-extrusion", "coincident-radii",
                                      "collapsed-clearance"})


class ReportsTest(unittest.TestCase):
    def test_unsupported_primitive_blocks_reconstruction(self):
        report = expressivity_report(("box", "spline"), ("box",))
        self.assertEqual(report.unsupported, ("spline",))
        self.assertFalse(report.reconstructable)

    def test_robustness_curve_records_the_corruption_seed(self):
        cloud, manifest = corrupt_cloud(((0, 0, 0), (1, 1, 1)), seed=4,
                                        dropout=.5, outliers=1)
        curve = robustness_curve(({"cloud": cloud, "manifest": manifest},),
                                 lambda value: {"distance": len(value)})
        self.assertEqual(curve[0]["seed"], 4)

    def test_point_budget_report_averages_distance_per_budget(self):
        budget = point_budget_report((
            {"points": 32, "distance": 2, "latency": 1, "memory": 3,
             "valid": True},
            {"points": 32, "distance": 4, "latency": 3, "memory": 5,
             "valid": False},
        ))
        self.assertEqual(budget[0]["mean_distance"], 2)


if __name__ == "__main__":
    unittest.main()
