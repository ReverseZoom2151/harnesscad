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
        "curvatures": [(0.0, 0.0), (0.01 * scale, 0.2), (-0.02, 0.1),
                       (0.3, 0.4), (0.0, 0.0), (0.05, 0.05)],
        "latents": [[0.1, 0.2, 0.3], [0.4, 0.1, 0.2], [0.2, 0.5, 0.1],
                    [0.3, 0.3, 0.4], [0.5, 0.2, 0.2]],
        "ranking": ([1.0, 1.0, 0.0, 0.0, 0.0] if perturbed
                    else [1.0, 0.0, 1.0, 0.0, 0.0]),
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
