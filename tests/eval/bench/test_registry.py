"""Tests for the bench metric registry and suite runner.

The load-bearing claim these tests pin down is the campaign's key finding:
**rival metrics are not interchangeable and must never be averaged together.**
``geometry.chamfer_unit_sphere`` and ``geometry.chamfer_bbox_judged`` are two
papers' Chamfer protocols; run on the SAME pred/gold pair they return different
numbers, and the registry must report them as separate, separately-named entries
rather than blending them into one "Chamfer" figure.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from harnesscad.core import cli
from harnesscad.eval.bench import registry as bench

# ---------------------------------------------------------------------------
# A small, deterministic synthetic sample: a unit cube (gold) against the same
# cube scaled by 1.1 with a corner voxel removed (pred). No randomness, no I/O.
# ---------------------------------------------------------------------------

CUBE_VERTICES = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
CUBE_TRIANGLES = [
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
]


def payload(scale: float, perturbed: bool) -> dict:
    """One side (pred or gold) of a sample: every input kind the metrics need.

    ``perturbed`` (the prediction) is not merely a scaled copy of the gold: it is
    stretched anisotropically (x1.25 in x, x0.9 in z). That matters -- a pure
    uniform scale is cancelled by every normalising Chamfer protocol, so the
    rivals would all collapse to ~0 and the disagreement they are famous for
    would be invisible. A non-similar shape makes each protocol's normalisation
    choice bite, which is exactly what the rival test pins down.
    """
    stretch = (1.25, 1.0, 0.9) if perturbed else (1.0, 1.0, 1.0)
    verts = [tuple(scale * stretch[d] * v[d] for d in range(3))
             for v in CUBE_VERTICES]
    points = list(verts) + [
        tuple(sum(verts[i][d] for i in tri) / 3.0 for d in range(3))
        for tri in CUBE_TRIANGLES
    ]
    voxels = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    if perturbed:
        voxels = [v for v in voxels if v != (0, 0, 0)]
    return {
        "mesh": {"vertices": verts, "faces": [list(t) for t in CUBE_TRIANGLES]},
        "points": points,
        "points2d": [(x * scale, y * scale) for x, y in
                     [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]],
        "voxels": voxels,
        "commands": [
            {"type": "line", "params": [0.0, 0.0, 1.0 * scale, 0.0]},
            {"type": "line", "params": [1.0 * scale, 0.0, 1.0 * scale, 1.0 * scale]},
            {"type": "circle", "params": [0.5, 0.5, 0.25 * scale]},
            {"type": "extrude", "params": [1.0 * scale]},
        ],
        "deepcad_rows": [
            {"type": "SOL"},
            {"type": "Line", "x": 10, "y": 0},
            {"type": "Line", "x": int(10 * scale), "y": int(10 * scale)},
            {"type": "Ext", "s": int(20 * scale), "ox": 0, "oy": 0, "oz": 0},
        ],
        "slot_rows": [
            [0] + [-1] * 16,
            [1] + [int(10 * scale), 0] + [-1] * 14,
            [4] + [int(20 * scale)] + [0] * 15,
        ],
        "op_matrix": [[0, -1, -1, -1],
                      [1, int(10 * scale), 0, 0],
                      [4, int(20 * scale), 0, 0]],
        "op_tokens": (["Workplane", "box", "extrude"] if perturbed
                      else ["Workplane", "box", "faces", "extrude"]),
        "code": ("import cadquery as cq\n"
                 "r = cq.Workplane('XY').box(22.0, 10.0, 5.0).faces('>Z')\n"
                 if perturbed else
                 "import cadquery as cq\n"
                 "r = cq.Workplane('XY').box(20.0, 10.0, 5.0)\n"),
        "cad_sequence": {
            "curves": [{"type": "line", "start": (0.0, 0.0),
                        "end": (1.0 * scale, 0.0)}],
            "extrusion": {"d_plus": 1.0 * scale, "d_minus": 0.0},
        },
        "tokens": ([(7, 8), (9, 11), (11, 12), (3, 4)] if perturbed
                   else [(7, 8), (9, 10), (11, 12), (3, 4)]),
        "params": {"width": 20.0 * scale, "height": 10.0, "depth": 5.0},
        "sketch": {
            "primitives": [
                {"id": "p1", "type": "line", "params": [0.0, 0.0, 1.0 * scale, 0.0]},
                {"id": "p2", "type": "circle", "params": [0.5, 0.5, 0.25]},
            ],
            "constraints": [
                {"id": "c1", "type": "coincident", "primitives": ["p1", "p2"]},
            ],
        },
        "sketch_map": {"sk1": [("line", 0.0, 0.0, 1.0 * scale, 0.0),
                               ("circle", 0.5, 0.5, 0.25)]},
        "entities": [("line", 0.0, 0.0, 1.0 * scale, 0.0),
                     ("circle", 0.5, 0.5, 0.25)],
        "raster": [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1],
                   [0, 0, 1, 0 if perturbed else 1]],
        "mask_pixels": ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (2, 3), (3, 2)]
                        + ([] if perturbed else [(3, 3)])),
        "mask": [[1.0, 1.0, 0.0, 0.0], [1.0, 0.9, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.8]],
        "depth": [1.0, 1.1, 1.2, 1.4, 1.9, 2.5],
        # A curvature sample is (SDF gradient, Hessian): a plane (zero Hessian)
        # is developable; the perturbed side bends one principal direction.
        "curvatures": [
            ((0.0, 0.0, 1.0), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                               (0.0, 0.0, 0.0))),
            ((0.0, 0.0, 1.0), ((0.1 * scale, 0.0, 0.0),
                               (0.0, 0.0 if not perturbed else 0.2, 0.0),
                               (0.0, 0.0, 0.0))),
            ((1.0, 0.0, 0.0), ((0.0, 0.0, 0.0), (0.0, 0.05, 0.0),
                               (0.0, 0.0, 0.05 * scale))),
        ],
        "latents": [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.2, 0.5, 0.1],
                    [0.3, 0.3, 0.4], [0.5, 0.2, 0.2]],
        "ranking": ([1.0, 1.0, 0.0, 0.0, 0.0] if perturbed
                    else [1.0, 0.0, 1.0, 0.0, 0.0]),
        # -- payload kinds the second adapter wave needs ----------------------
        "adjacency": {i: [j for j in (i - 1, i + 1) if 0 <= j < 6]
                      for i in range(6)},
        "labels": ([0, 0, 1, 1, 1, 2] if perturbed else [0, 0, 0, 1, 1, 2]),
        "face_labels": ({0: 1, 1: 1, 2: 2, 3: 0} if perturbed
                        else {0: 1, 1: 2, 2: 2, 3: 0}),
        "cluster_labels": ([0, 0, 1, 1, 1] if perturbed else [0, 0, 0, 1, 1]),
        "instances": ([[0, 1, 2], [3, 4], [5]] if perturbed
                      else [[0, 1, 2, 3], [4], [5]]),
        "symbol_instances": {
            "lengths": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "instances": ([{"class_id": 0, "indices": [0, 1], "score": 0.9},
                           {"class_id": 1, "indices": [2, 3, 4], "score": 0.8}]
                          if perturbed else
                          [{"class_id": 0, "indices": [0, 1, 2], "score": 1.0},
                           {"class_id": 1, "indices": [3, 4], "score": 1.0}]),
        },
        "bbox": [0.0, 20.0 * scale, 0.0, 10.0, 0.0, 5.0],
        "scad": ("cube([22, 10, 5]);" if perturbed else "cube([20, 10, 5]);"),
        "deepcad_commands": [
            {"type": "SOL"},
            {"type": "Line", "x": 0, "y": 0},
            {"type": "Line", "x": int(10 * scale), "y": 0},
            {"type": "Line", "x": int(10 * scale), "y": int(10 * scale)},
        ] + ([] if perturbed else [{"type": "Line", "x": 0, "y": 10}]) + [
            {"type": "Ext", "theta": 0, "phi": 0, "gamma": 0, "px": 0, "py": 0,
             "pz": 0, "s": 1, "e1": int(2 * scale), "e2": 0, "b": 0, "u": 0},
        ],
        "text2cad_model": [{
            "sketch": [[[
                {"type": "line", "start": (0.0, 0.0), "end": (10.0 * scale, 0.0)},
                {"type": "line", "start": (10.0 * scale, 0.0),
                 "end": (10.0 * scale, 10.0)},
                {"type": "line", "start": (10.0 * scale, 10.0), "end": (0.0, 10.0)},
                {"type": "line", "start": (0.0, 10.0), "end": (0.0, 0.0)},
            ]]],
            "extrusion": {"extent_one": 0.75 * scale, "extent_two": 0.0,
                          "origin": (0.0, 0.0, 0.0), "euler": (0.0, 0.0, 0.0),
                          "sketch_size": 0.75, "boolean": 0},
        }],
        # DAVINCI 8-token blocks: t1 type, t2..t7 params in [1..64], t8 flag.
        "primitive_tokens": ([[3, 1, 1, 40, 1, 1, 1, 0],
                              [2, 32, 32, 16, 1, 1, 1, 0]] if perturbed else
                             [[3, 1, 1, 32, 1, 1, 1, 0],
                              [2, 32, 32, 16, 1, 1, 1, 0]]),
        "pose": ({"R": [[1.0, 0.0, 0.0], [0.0, 0.996, -0.087],
                        [0.0, 0.087, 0.996]], "t": [0.02, 0.0, 0.0]}
                 if perturbed else
                 {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                  "t": [0.0, 0.0, 0.0]}),
        "poses": [{"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                   "t": [0.0, 0.0, 0.0]},
                  {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                   "t": [1.0 * scale, 0.0, 0.0]},
                  {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                   "t": [1.0 * scale, 1.0, 0.0]}],
        "part_names": (["bolt.step", "nut.step", "washer.step"] if perturbed
                       else ["bolt.step", "nut.step", "bracket.step"]),
        "scored_candidates": {
            "scores": [0.1, 0.9, 0.4, 0.2] if perturbed else [0.9, 0.1, 0.4, 0.2],
            "labels": [1, 0, 0, 0],
        },
        "similarity": ([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]]
                       if perturbed else
                       [[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]]),
        "design": {"curves": [
            {"kind": "line", "points": [(0.0, 0.0), (10.0 * scale, 0.0)]},
            {"kind": "line", "points": [(10.0 * scale, 0.0), (10.0 * scale, 10.0)]},
        ]},
    }


def sample(sample_id: str = "synth-1") -> dict:
    return {"id": sample_id,
            "pred": payload(1.10, perturbed=True),
            "gold": payload(1.00, perturbed=False)}


def _boom(pred: dict, gold: dict) -> float:
    raise RuntimeError("metric exploded on purpose")


BROKEN = bench.Metric(
    name="geometry.zzz_broken", kind="geometry",
    dotted="harnesscad.eval.bench.geometry.chamfer",
    inputs=("points",), adapter=_boom, summary="a metric that always raises",
)


class DiscoveryTest(unittest.TestCase):
    """The metric list is derived from the AST capability index, not hardcoded."""

    def test_discovers_more_than_thirty_metrics(self):
        found = bench.metrics()
        self.assertGreater(len(found), 30, f"only {len(found)} metrics discovered")

    def test_every_metric_points_at_a_real_indexed_bench_module(self):
        from harnesscad import registry as capability_registry
        indexed = {e.dotted for e in capability_registry.find(package="bench")}
        for m in bench.metrics():
            self.assertIn(m.dotted, indexed, f"{m.name} adapts an unindexed module")
            self.assertIn(m.kind, bench.KINDS)
            for key in m.inputs:
                self.assertIn(key, bench.INPUT_KINDS)

    def test_kind_filter(self):
        geometry = bench.metrics(kind="geometry")
        self.assertTrue(geometry)
        self.assertTrue(all(m.kind == "geometry" for m in geometry))
        self.assertLess(len(geometry), len(bench.metrics()))
        for kind in ("sequence", "sketch", "vision", "retrieval", "generative"):
            self.assertTrue(bench.metrics(kind=kind), f"no {kind} metrics")

    def test_unadapted_modules_are_reported_not_hidden(self):
        adapted = {m.dotted for m in bench.metrics()}
        self.assertTrue(bench.unadapted())
        self.assertFalse(adapted.intersection(bench.unadapted()))

    def test_metric_ordering_is_deterministic(self):
        self.assertEqual([m.name for m in bench.metrics()],
                         sorted(m.name for m in bench.metrics()))


class SuiteRunTest(unittest.TestCase):
    """A named suite runs end to end on the synthetic pred/gold pair."""

    def test_suites_are_named_and_nonempty(self):
        self.assertIn("deepcad", bench.suites())
        self.assertIn("cadrille", bench.suites())
        for name in bench.suites():
            self.assertTrue(bench.suite(name).metric_names)

    def test_unknown_suite_raises(self):
        with self.assertRaises(KeyError):
            bench.suite("does-not-exist")

    def test_run_suite_end_to_end(self):
        report = bench.run_suite("deepcad", [sample()])
        self.assertEqual(report.suite, "deepcad")
        self.assertEqual(report.n_samples, 1)
        self.assertEqual(len(report.results), len(bench.suite("deepcad").metric_names))
        self.assertFalse(report.errors(), [r.error for r in report.errors()])
        self.assertTrue(report.ok())

        aggregates = report.aggregates()
        self.assertIn("geometry.chamfer_unit_sphere", aggregates)
        self.assertIn("sequence.reconstruction_accuracy", aggregates)
        # Every number is stamped with the metric AND the module that produced it.
        for result in report.ok():
            self.assertTrue(result.dotted.startswith("harnesscad.eval.bench."))
        payload_json = json.dumps(report.to_dict(), sort_keys=True)
        self.assertIn("geometry.chamfer_unit_sphere", payload_json)

    def test_run_suite_is_deterministic(self):
        a = bench.run_suite("cadrille", [sample()]).to_dict()
        b = bench.run_suite("cadrille", [sample()]).to_dict()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_missing_inputs_are_skipped_not_guessed(self):
        thin = {"id": "thin", "pred": {"points": payload(1.1, True)["points"]},
                "gold": {"points": payload(1.0, False)["points"]}}
        report = bench.run_suite("deepcad", [thin])
        self.assertTrue(report.skipped())
        self.assertFalse(report.errors())
        chamfer = report.by_metric("geometry.chamfer_unit_sphere")
        self.assertEqual(chamfer[0].status, "ok")


class RivalMetricTest(unittest.TestCase):
    """The campaign's key finding, pinned: rivals disagree and are never blended."""

    def test_chamfer_rivals_produce_different_numbers_on_the_same_input(self):
        s = sample()
        sphere = bench.metric("geometry.chamfer_unit_sphere")
        judged = bench.metric("geometry.chamfer_bbox_judged")
        cube = bench.metric("geometry.chamfer_unit_cube")

        r_sphere = bench.run_metric(sphere, s)
        r_judged = bench.run_metric(judged, s)
        r_cube = bench.run_metric(cube, s)
        for r in (r_sphere, r_judged, r_cube):
            self.assertEqual(r.status, "ok", r.error)

        # Same pred, same gold -- three protocols, three different numbers.
        sphere_cd = r_sphere.value
        judged_cd = r_judged.value["cd"]
        cube_cd = r_cube.value["cd"] if isinstance(r_cube.value, dict) else r_cube.value
        self.assertIsInstance(sphere_cd, float)
        self.assertNotEqual(sphere_cd, judged_cd)
        self.assertNotEqual(sphere_cd, cube_cd)
        self.assertNotEqual(judged_cd, cube_cd)

        # Not float dust: the protocols disagree by orders of magnitude. Averaging
        # them would be meaningless, which is precisely why suites select one.
        values = sorted((sphere_cd, judged_cd, cube_cd))
        self.assertGreater(values[-1] / values[0], 100.0)
        for a, b in ((sphere_cd, judged_cd), (sphere_cd, cube_cd),
                     (judged_cd, cube_cd)):
            self.assertGreater(abs(a - b) / max(abs(a), abs(b)), 0.01)

        # They are separately named entries pointing at DIFFERENT modules.
        self.assertNotEqual(r_sphere.metric, r_judged.metric)
        self.assertNotEqual(r_sphere.dotted, r_judged.dotted)
        self.assertEqual(r_sphere.dotted,
                         "harnesscad.eval.bench.geometry.chamfer_unit_sphere")
        self.assertEqual(r_judged.dotted,
                         "harnesscad.eval.bench.protocols.chamfer_bbox_judged")

    def test_betti_rivals_disagree_by_design(self):
        s = sample()
        graded = bench.run_metric(bench.metric("geometry.betti_graded"), s)
        exact = bench.run_metric(bench.metric("geometry.betti_exact"), s)
        self.assertEqual(graded.status, "ok", graded.error)
        self.assertEqual(exact.status, "ok", exact.error)
        self.assertNotEqual(graded.dotted, exact.dotted)
        # They live in the same rival family, so no suite may select both.
        family = dict(bench.RIVAL_FAMILIES)["betti_topology"]
        self.assertIn("geometry.betti_graded", family)
        self.assertIn("geometry.betti_exact", family)

    def test_no_suite_blends_rival_metrics(self):
        families = bench.rivals()
        for name in bench.suites():
            chosen = set(bench.suite(name).metric_names)
            for family, members in families.items():
                overlap = chosen.intersection(members)
                self.assertLessEqual(
                    len(overlap), 1,
                    f"suite {name!r} blends rivals from {family!r}: {sorted(overlap)}")

    def test_rival_metrics_land_in_separate_report_entries(self):
        s = sample()
        sphere_report = bench.run_suite("deepcad", [s])          # unit-sphere CD
        judged_report = bench.run_suite("text_to_cadquery", [s])  # bbox-judged CD
        sphere_aggr = sphere_report.aggregates()
        judged_aggr = judged_report.aggregates()
        self.assertIn("geometry.chamfer_unit_sphere", sphere_aggr)
        self.assertNotIn("geometry.chamfer_bbox_judged", sphere_aggr)
        self.assertIn("geometry.chamfer_bbox_judged", judged_aggr)
        self.assertNotIn("geometry.chamfer_unit_sphere", judged_aggr)
        # No pooled "chamfer" key exists anywhere: nothing averages the rivals.
        self.assertNotIn("chamfer", sphere_aggr)
        self.assertNotIn("chamfer", judged_aggr)

    def test_running_rivals_together_is_refused(self):
        rival = bench.Metric(
            name="geometry.chamfer_bbox_judged", kind="geometry",
            dotted="harnesscad.eval.bench.protocols.chamfer_bbox_judged",
            inputs=("points",),
            adapter=bench.metric("geometry.chamfer_bbox_judged").adapter,
        )
        with self.assertRaises(bench.RivalBlendError):
            bench.run_suite("deepcad", [sample()], extra_metrics=[rival])


class ErrorHandlingTest(unittest.TestCase):
    """A metric that raises is recorded, not fatal."""

    def test_raising_metric_becomes_an_error_entry_without_aborting(self):
        report = bench.run_suite("geometry_smoke", [sample()],
                                 extra_metrics=[BROKEN])
        errors = report.errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].metric, "geometry.zzz_broken")
        self.assertEqual(errors[0].status, "error")
        self.assertIn("RuntimeError", errors[0].error)
        self.assertIn("exploded on purpose", errors[0].error)
        # The rest of the suite still produced numbers.
        self.assertTrue(report.ok())
        self.assertIn("geometry.chamfer_unit_sphere", report.aggregates())
        # ... and the broken metric contributes nothing to any aggregate.
        self.assertNotIn("geometry.zzz_broken", report.aggregates())

    def test_run_metric_never_raises(self):
        result = bench.run_metric(BROKEN, sample())
        self.assertEqual(result.status, "error")
        self.assertIsNone(result.value)


#: The metrics added by the second adapter wave. Each name must resolve to a
#: Metric whose ``dotted`` is a really-indexed bench module (asserted below).
WAVE_TWO = (
    "geometry.boundary_fscore",
    "geometry.chamfer_refinement_2d",
    "geometry.dimension_accuracy",
    "geometry.program_shape_match",
    "sequence.sequence_f1",
    "sequence.code_validity",
    "sequence.primitive_f1_null_class",
    "sketch.set_prediction_f1",
    "vision.face_segmentation",
    "vision.pointwise_semantic",
    "vision.instance_segmentation",
    "vision.length_weighted_panoptic",
    "vision.point_weighted_panoptic",
    "vision.object_pose_add",
    "vision.camera_pose_trajectory",
    "retrieval.clustering_external",
    "retrieval.clustering_internal",
    "retrieval.graded_retrieval",
    "retrieval.gallery_retrieval",
    "retrieval.image_retrieval_accuracy",
    "retrieval.latent_alignment",
    "retrieval.part_retrieval",
    "retrieval.joint_prediction_ranking",
    "generative.diversity_similarity_matrix",
)

#: The rival families the second wave introduced (or joined). Every one of these
#: must be enforced: two members in one suite is a RivalBlendError.
WAVE_TWO_RIVALS = {
    "chamfer_distance_2d": ("sketch.chamfer_2d", "geometry.chamfer_refinement_2d",
                            "sketch.set_prediction_f1"),
    "primitive_f1": ("sequence.command_f1", "sequence.sequence_f1",
                     "sequence.primitive_f1_null_class"),
    "labelwise_agreement": ("vision.face_segmentation", "vision.pointwise_semantic"),
    "panoptic_quality": ("vision.instance_segmentation",
                         "vision.length_weighted_panoptic",
                         "vision.point_weighted_panoptic"),
    "validity_rate": ("sequence.invalidity_ratio", "sequence.code_validity"),
    "latent_retrieval_accuracy": ("retrieval.graded_retrieval",
                                  "retrieval.gallery_retrieval",
                                  "retrieval.image_retrieval_accuracy",
                                  "retrieval.latent_alignment"),
    "set_diversity": ("generative.diversity",
                      "generative.diversity_similarity_matrix"),
    "volumetric_iou": ("geometry.voxel_iou_points", "geometry.program_shape_match"),
}


class SecondWaveDiscoveryTest(unittest.TestCase):
    """The newly adapted metrics are discovered and bound to real modules."""

    def test_every_new_metric_is_discovered(self):
        known = {m.name: m for m in bench.metrics()}
        for name in WAVE_TWO:
            self.assertIn(name, known, f"{name} was not discovered")

    def test_every_new_metric_maps_to_a_real_indexed_module(self):
        from harnesscad import registry as capability_registry
        indexed = {e.dotted for e in capability_registry.find(package="bench")}
        for name in WAVE_TWO:
            m = bench.metric(name)
            self.assertIn(m.dotted, indexed, f"{name} adapts an unindexed module")
            self.assertIn(m.kind, bench.KINDS)
            for key in m.inputs:
                self.assertIn(key, bench.INPUT_KINDS, f"{name} needs unknown {key}")

    def test_new_metrics_do_not_reuse_an_already_adapted_module(self):
        # Except by design: two metrics may share a module only when they are
        # DIFFERENT protocols over it (as voxel_iou_points / voxel_iou_grid are).
        counts = {}
        for m in bench.metrics():
            counts.setdefault(m.dotted, []).append(m.name)
        for dotted, names in counts.items():
            if len(names) > 1:
                self.assertEqual(sorted(names),
                                 sorted(["geometry.voxel_iou_grid",
                                         "geometry.voxel_iou_points"]),
                                 f"{dotted} is adapted twice: {names}")

    def test_registry_grew_and_the_unadapted_list_shrank(self):
        # 51 metrics over 48 modules before the second wave; 132 bench orphans.
        self.assertGreaterEqual(len(bench.metrics()), 51 + len(WAVE_TWO))
        adapted_modules = {m.dotted for m in bench.metrics()}
        self.assertGreaterEqual(len(adapted_modules), 48 + len(WAVE_TWO))
        self.assertFalse(adapted_modules.intersection(bench.unadapted()))
        # Every module we adapted really left the unadapted list.
        for name in WAVE_TWO:
            self.assertNotIn(bench.metric(name).dotted, bench.unadapted())

    def test_every_stated_reason_names_a_still_unadapted_module(self):
        unadapted = set(bench.unadapted())
        self.assertTrue(bench.reasons())
        for dotted, reason in bench.UNADAPTED_REASONS:
            self.assertIn(dotted, unadapted,
                          f"{dotted} has a reason but IS adapted")
            self.assertTrue(reason.strip(), f"{dotted} has an empty reason")

    def test_unadapted_report_is_listed_with_reasons_in_the_cli(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["bench", "--unadapted"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("without an adapter", text)
        self.assertIn("reason:", text)


class SecondWaveScoringTest(unittest.TestCase):
    """A sample of the new metrics really score on the synthetic sample."""

    def test_every_new_metric_scores_without_error(self):
        s = sample()
        for name in WAVE_TWO:
            result = bench.run_metric(bench.metric(name), s)
            self.assertEqual(result.status, "ok", f"{name}: {result.error}")
            self.assertTrue(bench._numeric_fields(result.value),
                            f"{name} produced no numbers")

    def test_selected_new_metrics_produce_the_expected_numbers(self):
        s = sample()
        value = lambda name: bench.run_metric(bench.metric(name), s).value

        # A 3-part boundary shifted by one node: the boundary sets still touch.
        self.assertIn("boundary_f1", value("geometry.boundary_fscore"))

        # The pred bbox is 10% wider than the gold one -> width accuracy 0.9.
        self.assertAlmostEqual(
            value("geometry.dimension_accuracy")["accuracy_width"], 0.9, places=6)

        # The CadQuery snippet has no `solid = ...` assignment: contract fails.
        self.assertEqual(value("sequence.code_validity")["valid"], 0.0)

        # The prediction drops one line of the rectangle -> line F1 below 1.
        self.assertLess(value("sequence.sequence_f1")["line"], 1.0)

        # Paired latents: pred[i] IS gold[i], so top-1 retrieval is perfect.
        self.assertEqual(value("retrieval.latent_alignment")["top1_accuracy"], 1.0)

        # The gold part set differs by one filename -> not an exact match.
        self.assertEqual(value("retrieval.part_retrieval")["exact_match"], 0.0)

    def test_new_metrics_are_deterministic(self):
        s = sample()
        for name in WAVE_TWO:
            a = bench.run_metric(bench.metric(name), s).to_dict()
            b = bench.run_metric(bench.metric(name), s).to_dict()
            self.assertEqual(json.dumps(a, sort_keys=True),
                             json.dumps(b, sort_keys=True), name)

    def test_new_suites_run_clean(self):
        s = sample()
        for name in ("text2cad", "sympoint", "cluster3d"):
            report = bench.run_suite(name, [s])
            self.assertFalse(report.errors(), [r.error for r in report.errors()])
            self.assertTrue(report.aggregates(), name)

    def test_a_new_metric_that_raises_is_captured_not_fatal(self):
        broken = bench.Metric(
            name="vision.zzz_broken", kind="vision",
            dotted="harnesscad.eval.bench.vision.face_segmentation",
            inputs=("face_labels",), adapter=_boom)
        report = bench.run_suite("sympoint", [sample()], extra_metrics=[broken])
        self.assertEqual(len(report.errors()), 1)
        self.assertTrue(report.ok())


class SecondWaveRivalTest(unittest.TestCase):
    """Every rival the second wave introduced is registered AND enforced."""

    def test_each_new_rival_is_in_its_family(self):
        families = bench.rivals()
        for family, members in WAVE_TWO_RIVALS.items():
            self.assertIn(family, families, f"{family} is not a rival family")
            for name in members:
                self.assertIn(name, families[family],
                              f"{name} is not registered in {family}")

    def test_two_members_of_a_family_cannot_be_run_together(self):
        s = sample()
        for family, members in WAVE_TWO_RIVALS.items():
            first, second = bench.metric(members[0]), bench.metric(members[1])
            # An ad-hoc suite of two rivals is refused at run time...
            with self.assertRaises(bench.RivalBlendError, msg=family):
                bench.run_suite("geometry_smoke", [s],
                                extra_metrics=[first, second])
            # ... and the conflict is attributed to the right family.
            conflicts = dict(bench._rival_conflicts([first.name, second.name]))
            self.assertIn(family, conflicts)

    def test_a_suite_definition_blending_new_rivals_is_refused(self):
        for family, members in WAVE_TWO_RIVALS.items():
            conflicts = bench._rival_conflicts(list(members))
            self.assertTrue(conflicts, f"{family} is not enforced")

    def test_new_rivals_disagree_on_the_same_input(self):
        s = sample()
        # Three panoptic protocols, three different PQ numbers on one sample.
        pq = []
        for name in WAVE_TWO_RIVALS["panoptic_quality"]:
            result = bench.run_metric(bench.metric(name), s)
            self.assertEqual(result.status, "ok", result.error)
            pq.append(result.value["pq"])
        self.assertEqual(len(set(pq)), len(pq), f"panoptic rivals agree: {pq}")

    def test_no_suite_selects_two_members_of_any_new_family(self):
        for name in bench.suites():
            chosen = set(bench.suite(name).metric_names)
            for family, members in WAVE_TWO_RIVALS.items():
                self.assertLessEqual(len(chosen.intersection(members)), 1,
                                     f"suite {name!r} blends {family!r}")


class CliTest(unittest.TestCase):
    """`harnesscad bench` is wired in and the existing subcommands still work."""

    def _run(self, argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(argv)
        return code, out.getvalue()

    def test_bench_list(self):
        code, out = self._run(["bench", "--list"])
        self.assertEqual(code, 0)
        self.assertIn("geometry.chamfer_unit_sphere", out)
        self.assertIn("metrics", out)

    def test_bench_suites_and_rivals(self):
        code, out = self._run(["bench", "--suites"])
        self.assertEqual(code, 0)
        self.assertIn("deepcad", out)
        code, out = self._run(["bench", "--rivals"])
        self.assertEqual(code, 0)
        self.assertIn("never averaged together", out)

    def test_bench_run_suite_from_json(self):
        handle, path = tempfile.mkstemp(suffix=".json")
        os.close(handle)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"samples": [sample()]}, fh)
            code, out = self._run(["bench", "--suite", "deepcad", "--input", path,
                                   "--json"])
            self.assertEqual(code, 0, out)
            report = json.loads(out)
            self.assertEqual(report["suite"], "deepcad")
            self.assertEqual(report["n_error"], 0)
            self.assertIn("geometry.chamfer_unit_sphere", report["aggregates"])
        finally:
            os.unlink(path)

    def test_bench_suite_without_input_errors(self):
        code, _ = self._run(["bench", "--suite", "deepcad"])
        self.assertEqual(code, 2)

    def test_existing_subcommands_still_work(self):
        parser = cli.build_parser()
        for argv in (["apply", "ops.json"], ["demo"], ["build", "a plate"],
                     ["capabilities", "--stats"], ["bench", "--list"]):
            args = parser.parse_args(argv)
            self.assertTrue(callable(args.func), argv)

        code, out = self._run(["demo"])
        self.assertEqual(code, 0)
        self.assertIn("digest:", out)

        code, out = self._run(["capabilities", "--stats"])
        self.assertEqual(code, 0)
        self.assertIn("total modules:", out)


if __name__ == "__main__":
    unittest.main()
