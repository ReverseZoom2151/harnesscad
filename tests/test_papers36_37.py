import math

from bench.appearance_invariance import appearance_invariance
from bench.candidate_scaling import candidate_scaling
from bench.geometry_prompted_segmentation import GeometrySegmentationCase, audit_cases
from bench.instance_segmentation import instance_metrics
from bench.point_budget import point_budget_report
from bench.pointcloud_robustness import corrupt_cloud, robustness_curve
from dataengine.code_modularity import code_modularity
from datagen.domain_randomization import RandomAxis, draw_scene, independence_audit
from datagen.reverse_engineering import build_reverse_sample
from datagen.sketch_boolean import realize_recipe, sketch_recipe
from ingest.fourier_features import fourier_features
from ingest.point_cloud import canonicalize_cloud
from quality.cad_abstraction import accept_abstraction, propose_abstraction
from quality.parameter_exposure import expose_parameters
from quality.quantization_risk import quantization_risks
from reconstruction.expressivity import expressivity_report
from reconstruction.pointcloud_candidates import select_pointcloud_candidate
from surfaces.canonical_views import canonical_views
from vision.geometry_prompt import GeometryPrompt, PromptView
from vision.instance_matching import mask_iou, mask_nms, one_to_many
from vision.mask_sampling import sample_mask


BOOL_MASK = ((True, False), (False, True))
MASK = ((0, 0), (1, 1))


def test_geometry_prompt_pipeline_is_seeded_and_auditable():
    views = canonical_views(5)
    assert views == canonical_views(5)
    assert all(math.isclose(sum(x*x for x in view.direction), 1.0) for view in views)
    points = sample_mask(BOOL_MASK, count=5, seed=4)
    prompt = GeometryPrompt("mesh", (
        PromptView("v0", (1.0,), b"rgb", BOOL_MASK, points),
    ))
    assert not prompt.validate()
    assert prompt.digest == prompt.digest


def test_domain_randomization_records_axes_and_detects_identity_leakage():
    axes = (RandomAxis("light", 1, 2), RandomAxis("texture", 3, 4))
    scene = draw_scene(axes, seed=3, identity="part")
    assert scene == draw_scene(axes, seed=3, identity="part")
    assert independence_audit((scene, draw_scene(axes, seed=4, identity="other")))


def test_instance_matching_and_metrics():
    other = ((0, 0),)
    assert mask_iou(MASK, MASK) == 1
    predictions = ({"mask": MASK, "score": 1.0},
                   {"mask": other, "score": .8})
    matched, selected = one_to_many(predictions, (MASK,), threshold=.4)
    assert len(matched) == 2 and selected == (1, 1)
    assert mask_nms((predictions[0], predictions[0])) == (0,)
    assert instance_metrics((MASK,), (MASK,))["pq"] == 1


def test_segmentation_manifest_and_appearance_audits():
    cases = (
        GeometrySegmentationCase("a", "i1", "m1", "p", ("x",), "train"),
        GeometrySegmentationCase("b", "i2", "m2", "p", (), "test"),
    )
    audit = audit_cases(cases)
    assert audit == (("b", "no-instances"),)
    result = appearance_invariance(
        ({"id": "a", "geometry": 1, "value": 1},
         {"id": "b", "geometry": 1, "value": 1},
         {"id": "c", "geometry": 2, "value": 9}),
        lambda case: case["value"], lambda a, b: abs(a-b))
    assert result["shortcut_suspected"]


def test_point_cloud_ingest_features_candidates_and_scaling():
    cloud, transform = canonicalize_cloud(((0, 0, 0), (2, 2, 2)),
                                           normalize=True)
    assert tuple(transform.invert(point) for point in cloud) == (
        (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    assert len(fourier_features((.5, .25, 0), (1, 2))) == 15
    selected = select_pointcloud_candidate(
        cloud, lambda _, seed: seed, lambda x: x,
        lambda shape, count, seed: cloud, count=3)
    assert selected["winner"].index == 0
    scaling = candidate_scaling(tuple(
        {"valid": item.valid, "distance": item.distance, "cost": 1}
        for item in selected["attempts"]))
    assert scaling["rows"][-1]["invalidity"] == 0


def test_sketch_recipe_has_adapter_seam_and_reverse_sample_provenance():
    recipe = sketch_recipe(8, minimum=3, maximum=3)

    class Adapter:
        empty = lambda self: []
        primitive = lambda self, kind, params: (kind, params)
        boolean = lambda self, result, shape, operation: result + [(shape, operation)]
        boundary_loops = lambda self, result: (((0, 1),),)
        intersects = lambda self, loops: False
        length = lambda self, edge: 1

    assert realize_recipe(recipe, Adapter())["accepted"]
    sample = build_reverse_sample(
        "s", recipe, lambda _: ({"op": "box"},), lambda ops: ops,
        repr, lambda shape: shape, lambda triangles, count, seed: ((0, 0, 0),),
        provenance={"seed": 8})
    assert sample.provenance["seed"] == 8


def test_quality_metrics_are_conservative_and_non_executable():
    source = "w = 2\nbox(w, w, 2)\nunused = 4\n"
    metrics = code_modularity(source)
    assert metrics["reused_variables"] == ("w",)
    assert metrics["dead_definitions"] == ("unused",)
    ops = (
        {"op": "add_line", "x1": 0, "y1": 0, "x2": 2, "y2": 0},
        {"op": "add_line", "x1": 2, "y1": 0, "x2": 2, "y2": 1},
        {"op": "add_line", "x1": 2, "y1": 1, "x2": 0, "y2": 1},
        {"op": "add_line", "x1": 0, "y1": 1, "x2": 0, "y2": 0},
        {"op": "extrude", "distance": 3},
    )
    proposal = propose_abstraction(ops)
    accepted = accept_abstraction(ops, proposal, lambda x: x,
                                  lambda a, b: {"valid": True, "distance": 0},
                                  tolerance=1e-6)
    assert proposal["kind"] == "box" and accepted["accepted"]
    assert expose_parameters(ops)["executable"] is False
    assert set(quantization_risks(step=1, extrusion=.1, radii=(1.1, 1.2),
                                  clearances=(.1,))) == {
                                      "zero-extrusion", "coincident-radii",
                                      "collapsed-clearance"}


def test_expressivity_robustness_and_point_budget_reports():
    report = expressivity_report(("box", "spline"), ("box",))
    assert report.unsupported == ("spline",) and not report.reconstructable
    cloud, manifest = corrupt_cloud(((0, 0, 0), (1, 1, 1)), seed=4,
                                    dropout=.5, outliers=1)
    curve = robustness_curve(({"cloud": cloud, "manifest": manifest},),
                             lambda value: {"distance": len(value)})
    assert curve[0]["seed"] == 4
    budget = point_budget_report((
        {"points": 32, "distance": 2, "latency": 1, "memory": 3,
         "valid": True},
        {"points": 32, "distance": 4, "latency": 3, "memory": 5,
         "valid": False},
    ))
    assert budget[0]["mean_distance"] == 2
