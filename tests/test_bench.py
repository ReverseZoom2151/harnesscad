"""Tests for the CADBench-Verified evaluation harness (bench/).

Runs the shipped sample tasks against the dependency-free StubBackend, so the
whole harness spine + metrics + report aggregation are exercised with no
geometry kernel installed.
"""

import os
import unittest

from backends.stub import StubBackend
from bench import (
    DIFFICULTIES, Task, load_tasks, run_suite, run_task,
    assembly_mate_accuracy, cad_sequence_f1, collision_rate,
    dimension_match, program_execution_rate, sketch_editability,
    trajectory_efficiency,
)

_TASK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "bench_tasks",
)


class TestLoadTasks(unittest.TestCase):
    def test_loads_sample_tasks(self):
        tasks = load_tasks(_TASK_DIR)
        self.assertEqual(len(tasks), 3)
        for t in tasks:
            self.assertIsInstance(t, Task)
            self.assertIn(t.difficulty, DIFFICULTIES)
            self.assertTrue(t.ops)
            self.assertIn("summary", t.acceptance)
        ids = {t.id for t in tasks}
        self.assertIn("easy_plate_20x10x5", ids)

    def test_rejects_bad_difficulty(self):
        with self.assertRaises(ValueError):
            Task(id="x", difficulty="trivial", brief="", ops=[], acceptance={})


class TestSuiteOnStub(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks(_TASK_DIR)
        self.report = run_suite(self.tasks, backend_factory=StubBackend)

    def test_report_task_count(self):
        self.assertEqual(self.report.n_tasks, len(self.tasks))
        self.assertEqual(len(self.report.results), len(self.tasks))

    def test_success_rate_in_unit_interval(self):
        self.assertGreaterEqual(self.report.task_success_rate, 0.0)
        self.assertLessEqual(self.report.task_success_rate, 1.0)

    def test_sample_tasks_all_pass_on_stub(self):
        # Sample tasks assert only stub-observable acceptance (summary +
        # validity-via-fallback); the measure family is skipped on the stub.
        self.assertEqual(self.report.task_success_rate, 1.0)
        for r in self.report.results:
            self.assertTrue(r.success, f"{r.task_id} unexpectedly failed")

    def test_per_difficulty_buckets(self):
        pd = self.report.per_difficulty
        self.assertEqual(set(pd), {"easy", "medium", "hard"})
        for difficulty, bucket in pd.items():
            self.assertEqual(bucket.n_tasks, 1)
            self.assertGreaterEqual(bucket.success_rate, 0.0)
            self.assertLessEqual(bucket.success_rate, 1.0)
            self.assertGreaterEqual(bucket.mean_trajectory_efficiency, 0.0)
            self.assertLessEqual(bucket.mean_trajectory_efficiency, 1.0)

    def test_reference_solver_is_optimal(self):
        # Replaying reference ops == the optimal trajectory -> efficiency 1.0.
        for r in self.report.results:
            self.assertEqual(r.trajectory_efficiency, 1.0)

    def test_report_serialises(self):
        d = self.report.to_dict()
        self.assertEqual(d["n_tasks"], len(self.tasks))
        self.assertIn("per_difficulty", d)

    def test_program_execution_rate_all_pass(self):
        # Every sample task rebuilds cleanly on the stub -> rate 1.0.
        self.assertEqual(self.report.program_execution_rate, 1.0)

    def test_assembly_and_collision_none_on_single_part_stub(self):
        # The stub exposes no assembly query, and the sample tasks carry no
        # reference_ops/reference_assembly, so every optional metric is skipped.
        self.assertIsNone(self.report.collision_rate)
        self.assertIsNone(self.report.mean_assembly_mate_accuracy)
        self.assertIsNone(self.report.mean_cad_sequence_f1)
        for r in self.report.results:
            self.assertIsNone(r.collision_rate)
            self.assertIsNone(r.assembly_mate_accuracy)
            self.assertIsNone(r.cad_sequence_f1)
        # ...and they serialise as null-valued fields without breaking to_dict.
        d = self.report.to_dict()
        self.assertIsNone(d["collision_rate"])
        self.assertIsNone(d["results"][0]["cad_sequence_f1"])


class TestDeliberateFailure(unittest.TestCase):
    def _plate_task(self, acceptance) -> Task:
        return Task(
            id="wrong_spec_plate",
            difficulty="easy",
            brief="20x10x5 plate with a deliberately wrong acceptance spec.",
            ops=[
                {"op": "new_sketch", "plane": "XY"},
                {"op": "add_rectangle", "sketch": "sk1",
                 "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
                {"op": "extrude", "sketch": "sk1", "distance": 5.0},
            ],
            acceptance=acceptance,
        )

    def test_wrong_bbox_and_feature_count_is_a_fail(self):
        # Deliberately-failing acceptance: a wrong bbox (measure) AND a wrong
        # feature_count (summary). On the StubBackend the measure/bbox family is
        # skipped (no geometry kernel), so the summary mismatch drives the fail;
        # on a geometry backend the wrong bbox fails the dimension check directly.
        task = self._plate_task({
            "summary": {"feature_count": 5, "solid_present": True},
            "measure": {"bbox": [99.0, 99.0, 99.0]},
            "tolerance": 0.02,
        })
        result = run_task(task, backend_factory=StubBackend)
        self.assertFalse(result.dimension_match)
        self.assertFalse(result.success)
        # The op stream itself rebuilt fine — only acceptance failed.
        self.assertTrue(result.program_execution)

    def test_correct_spec_passes(self):
        task = self._plate_task({
            "summary": {"feature_count": 1, "solid_present": True},
            "validity": {"is_valid": True},
            "tolerance": 0.02,
        })
        result = run_task(task, backend_factory=StubBackend)
        self.assertTrue(result.dimension_match)
        self.assertTrue(result.success)


class TestMetricsUnits(unittest.TestCase):
    def test_trajectory_efficiency(self):
        self.assertEqual(trajectory_efficiency(5, 5), 1.0)
        self.assertEqual(trajectory_efficiency(5, 10), 0.5)
        self.assertEqual(trajectory_efficiency(5, 0), 0.0)
        self.assertEqual(trajectory_efficiency(10, 5), 1.0)  # capped at 1.0

    def test_sketch_editability_fully_constrained(self):
        # Build a fully-constrained plate sketch on the stub (dof 4 - 4 == 0).
        from cisp.ops import NewSketch, AddRectangle, Constrain
        b = StubBackend()
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk1"))
        for _ in range(4):
            b.apply(Constrain(kind="distance", a="e1", value=1.0))
        self.assertEqual(sketch_editability(b), 1.0)

    def test_sketch_editability_no_sketches(self):
        self.assertEqual(sketch_editability(StubBackend()), 1.0)

    def test_dimension_match_skips_unmeasurable(self):
        # measure is unsupported on the stub -> those fields are skipped, not failed.
        b = StubBackend()
        ok, details = dimension_match(b, {"measure": {"volume": 123.0}})
        self.assertTrue(ok)
        self.assertEqual(details["skipped"], 1)
        self.assertEqual(details["failed"], 0)


class _OkStub:
    """Minimal ApplyOpsResult-shaped object for program_execution_rate tests."""

    def __init__(self, ok):
        self.ok = ok


class TestProgramExecutionRate(unittest.TestCase):
    def test_aggregates_mixed_pass_fail(self):
        # 2 of 4 rebuilt -> 0.5, over ApplyOpsResult-shaped items.
        items = [_OkStub(True), _OkStub(True), _OkStub(False), _OkStub(False)]
        self.assertEqual(program_execution_rate(items), 0.5)

    def test_all_pass_and_all_fail(self):
        self.assertEqual(program_execution_rate([True, True, True]), 1.0)
        self.assertEqual(program_execution_rate([False, False]), 0.0)

    def test_empty_is_none(self):
        self.assertIsNone(program_execution_rate([]))


class TestCadSequenceF1(unittest.TestCase):
    OPS = [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1",
         "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
        {"op": "extrude", "sketch": "sk1", "distance": 5.0},
    ]

    def test_identical_is_one(self):
        score = cad_sequence_f1(self.OPS, self.OPS)
        self.assertEqual(score["f1"], 1.0)
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["matched"], 3)

    def test_partial_match_below_one(self):
        # Same op tags, but the extrude distance differs -> that op is unmatched.
        built = [dict(o) for o in self.OPS]
        built[-1] = {"op": "extrude", "sketch": "sk1", "distance": 7.0}
        score = cad_sequence_f1(built, self.OPS)
        self.assertEqual(score["matched"], 2)
        self.assertLess(score["f1"], 1.0)
        self.assertGreater(score["f1"], 0.0)

    def test_ignores_reference_ids(self):
        # Differing sketch handles must NOT count against the match (id-agnostic).
        built = [dict(o) for o in self.OPS]
        built[1] = dict(built[1], sketch="sketch_a")
        built[2] = dict(built[2], sketch="sketch_a")
        self.assertEqual(cad_sequence_f1(built, self.OPS)["f1"], 1.0)

    def test_none_reference_skips(self):
        self.assertIsNone(cad_sequence_f1(self.OPS, None))


class TestAssemblyMetrics(unittest.TestCase):
    def test_mate_accuracy_none_without_assembly(self):
        self.assertIsNone(assembly_mate_accuracy({}, {}))
        self.assertIsNone(assembly_mate_accuracy(None, {"mates": []}))

    def test_mate_accuracy_scores_types_and_dof(self):
        ref = {"mates": [{"type": "coincident"}, {"type": "concentric"}],
               "residual_dof": 0}
        built = {"mates": [{"type": "coincident"}, {"type": "parallel"}],
                 "residual_dof": 1}
        score = assembly_mate_accuracy(built, ref)
        self.assertEqual(score["mate_type_accuracy"], 0.5)
        self.assertEqual(score["residual_dof_error"], 1.0)

    def test_collision_rate_none_when_single_part(self):
        # No assembly at all (single-part stub) -> not applicable.
        self.assertIsNone(collision_rate([StubBackend()]))
        self.assertIsNone(collision_rate([]))

    def test_collision_rate_fraction_over_assemblies(self):
        clean = {"part_count": 2, "mates": [{"type": "coincident"}],
                 "interferences": []}
        clashing = {"part_count": 2, "interferences": [{"a": "p1", "b": "p2"}]}
        self.assertEqual(collision_rate([clean, clashing]), 0.5)
        self.assertEqual(collision_rate([clashing]), 1.0)
        self.assertEqual(collision_rate([clean]), 0.0)


if __name__ == "__main__":
    unittest.main()
