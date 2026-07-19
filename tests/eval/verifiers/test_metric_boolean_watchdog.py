"""Tests for the metric-boolean watchdog -- the time budget around a single
manifold3d boolean, its subprocess isolation, and the kill+reap on overrun.

The hazard: metric booleans moved onto manifold3d because OCCT booleans HANG on
interface-overlay geometry, but that only LOWERED the hang probability. Upstream
manifold's own fuzzers watchdog every boolean at 10s and its ``ExecutionContext``
cannot interrupt a single large boolean, so a pathological ``a ^ b`` can still
wedge the interpreter. Python cannot interrupt an in-thread C call, so the only
real kill is a separate PROCESS -- exactly what :func:`intersection_volume_isolated`
does, mirroring ``io.ingest.step_check``.

Both arms are asserted: a normal boolean completes and returns the right volume;
an injected hanging boolean is killed by the budget and returns a typed refusal
(``volume=None`` -> UNKNOWN), never a hang and never a fabricated number.
"""

import sys
import time
import unittest

from harnesscad.eval.verifiers.metric_booleans import (
    BOOLEAN_STATUSES,
    BOOLEAN_WORKER_MODULE,
    DEFAULT_BOOLEAN_BUDGET_S,
    BooleanBudgetResult,
    classify_overlap,
    intersection_volume_isolated,
    manifold_available,
    manifold_to_mesh_arrays,
)

_HAVE_MANIFOLD = manifold_available()


def _cubes():
    """Two 10mm cubes offset 5mm on each axis: overlap is a 5mm cube = 125 mm^3."""
    import manifold3d as m3d
    a = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False)
    b = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False).translate([5.0, 5.0, 5.0])
    return a, b


@unittest.skipUnless(_HAVE_MANIFOLD, "manifold3d not installed")
class NormalBooleanArmTest(unittest.TestCase):
    """Arm 1: a normal boolean completes untouched and returns the right volume."""

    def test_isolated_boolean_returns_the_right_volume(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=DEFAULT_BOOLEAN_BUDGET_S)
        self.assertEqual(r.status, "ok")
        self.assertIsNotNone(r.volume)
        self.assertAlmostEqual(r.volume, 125.0, places=3)
        self.assertTrue(r.ok)

    def test_normal_boolean_worker_is_reaped(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=DEFAULT_BOOLEAN_BUDGET_S)
        self.assertEqual(r.returncode, 0)   # ran to completion, reaped
        self.assertFalse(r.timed_out)
        self.assertFalse(r.killed)

    def test_isolated_volume_classifies_as_a_clash(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b)
        self.assertEqual(classify_overlap(r.volume), "clash")

    def test_disjoint_solids_intersect_in_nothing(self):
        import manifold3d as m3d
        a = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False)
        far = m3d.Manifold.cube([10.0, 10.0, 10.0], center=False).translate([100, 0, 0])
        r = intersection_volume_isolated(a, far)
        self.assertIn(r.status, ("ok", "empty"))
        self.assertEqual(r.volume, 0.0)

    def test_mesh_arrays_round_trip(self):
        a, _ = _cubes()
        arrays = manifold_to_mesh_arrays(a)
        self.assertIsNotNone(arrays)
        vp, tv = arrays
        self.assertEqual(vp.shape[1], 3)
        self.assertEqual(tv.shape[1], 3)


@unittest.skipUnless(_HAVE_MANIFOLD, "manifold3d not installed")
class TimeoutArmTest(unittest.TestCase):
    """Arm 2: a hanging boolean is killed by the budget, not waited on."""

    def _stall_worker(self, seconds=60):
        # The REAL worker, told to stall before the boolean -- so the kill+reap
        # path is exercised on the actual boolean worker, not a stand-in.
        return [sys.executable, "-m", BOOLEAN_WORKER_MODULE,
                "--stall", str(seconds)]

    def test_hanging_boolean_times_out_not_hangs(self):
        a, b = _cubes()
        start = time.monotonic()
        r = intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=self._stall_worker())
        elapsed = time.monotonic() - start
        self.assertEqual(r.status, "timeout")
        self.assertTrue(r.timed_out)
        self.assertTrue(r.killed)
        self.assertLess(elapsed, 30.0)   # not the worker's 60s

    def test_timed_out_boolean_returns_unknown_not_a_number(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=self._stall_worker())
        self.assertIsNone(r.volume)
        self.assertEqual(classify_overlap(r.volume), "unknown")

    def test_killed_worker_is_reaped_not_leaked(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=self._stall_worker())
        self.assertIsNotNone(r.returncode)   # only set if the corpse was waited on

    def test_timeout_note_states_the_budget(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=self._stall_worker())
        self.assertIn("1", r.note)

    def test_timed_out_result_is_not_ok(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=self._stall_worker())
        self.assertFalse(r.ok)


@unittest.skipUnless(_HAVE_MANIFOLD, "manifold3d not installed")
class GuardTest(unittest.TestCase):
    """Malformed operands and misbehaving workers degrade, never raise."""

    def test_none_operand_is_unavailable_not_a_crash(self):
        a, _ = _cubes()
        r = intersection_volume_isolated(a, None)
        self.assertEqual(r.status, "unavailable")
        self.assertIsNone(r.volume)

    def test_worker_emitting_garbage_is_an_error(self):
        a, b = _cubes()
        liar = [sys.executable, "-c", "print('not json at all')"]
        r = intersection_volume_isolated(a, b, worker_cmd=liar)
        self.assertEqual(r.status, "error")
        self.assertIsNone(r.volume)

    def test_worker_emitting_unknown_status_is_an_error(self):
        a, b = _cubes()
        liar = [sys.executable, "-c",
                "import json; print(json.dumps({'status': 'brilliant'}))"]
        r = intersection_volume_isolated(a, b, worker_cmd=liar)
        self.assertEqual(r.status, "error")

    def test_unstartable_worker_is_an_error(self):
        a, b = _cubes()
        r = intersection_volume_isolated(a, b,
                                         worker_cmd=["definitely-not-a-binary-xyz"])
        self.assertEqual(r.status, "error")
        self.assertIn("cannot start worker", r.note)

    def test_every_reported_status_is_declared(self):
        a, b = _cubes()
        results = [
            intersection_volume_isolated(a, b),
            intersection_volume_isolated(a, None),
            intersection_volume_isolated(a, b, budget_s=1.0,
                                         worker_cmd=[sys.executable, "-m",
                                                     BOOLEAN_WORKER_MODULE,
                                                     "--stall", "60"]),
        ]
        for r in results:
            with self.subTest(status=r.status):
                self.assertIn(r.status, BOOLEAN_STATUSES)


class NoKernelContractTest(unittest.TestCase):
    """Contract that holds with or without a kernel installed."""

    def test_budget_default_is_generous_and_finite(self):
        self.assertGreater(DEFAULT_BOOLEAN_BUDGET_S, 0)
        self.assertLess(DEFAULT_BOOLEAN_BUDGET_S, 3600)

    def test_result_ok_semantics(self):
        self.assertTrue(BooleanBudgetResult("ok", 1.0).ok)
        self.assertTrue(BooleanBudgetResult("empty", 0.0).ok)
        self.assertFalse(BooleanBudgetResult("timeout", None).ok)
        self.assertFalse(BooleanBudgetResult("unavailable", None).ok)

    def test_none_operand_never_spawns_a_worker(self):
        # No kernel needed: both operands unmarshallable -> unavailable, no child.
        r = intersection_volume_isolated(None, None)
        self.assertEqual(r.status, "unavailable")


class SelfcheckTest(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.eval.verifiers.metric_booleans import main
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
