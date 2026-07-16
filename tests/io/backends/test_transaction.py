"""Tests for io.backends.transaction -- the byte-equivalence invariant.

The load-bearing assertion throughout: after a failed op, a digest of the
backend's FULL state equals the digest taken before the call. "Looks the same"
is not the claim; byte-identical is.
"""

import unittest

from harnesscad.core.cisp.ops import AddCircle, Extrude, NewSketch
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.io.backends.base import ApplyResult
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.backends.transaction import (
    RollbackFailed,
    TransactionalBackend,
    preflight,
    restore_state,
    snapshot_state,
    state_fingerprint,
    with_rollback,
)


class _CorruptingStub(StubBackend):
    """A real StubBackend whose Extrude wrecks state and then fails.

    The half-done fillet, reproduced against the harness's actual backend rather
    than a toy: it mutates nested sketch state in place, invents a feature, and
    only then gives up.
    """

    def __init__(self, mode="raise"):
        self.mode = mode
        super().__init__()

    def _dispatch(self, op):
        if isinstance(op, Extrude):
            # Wreckage first, failure second -- the whole problem in two lines.
            self.features.append({"type": "half-done-extrude"})
            self.solid_present = True
            for sketch in self.sketches.values():
                sketch["entities"].append("orphan")
                sketch["dof"] += 99
            if self.mode == "raise":
                raise RuntimeError("kernel: boolean degeneracy")
            return ApplyResult(False, [], [
                Diagnostic(Severity.ERROR, "kernel-fail", "gave up", None)])
        return super()._dispatch(op)


def _seeded(backend):
    """A backend with real, non-trivial state to be preserved."""
    backend.apply(NewSketch(plane="XY"))
    backend.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=5.0))
    return backend


class ByteEquivalenceTest(unittest.TestCase):
    """The invariant: a failed op leaves state byte-identical."""

    def test_control_unwrapped_backend_is_corrupted(self):
        # Without the guard the corruption is real. If this ever fails, every
        # other test here is proving nothing.
        raw = _seeded(_CorruptingStub())
        before = state_fingerprint(raw)
        with self.assertRaises(RuntimeError):
            raw.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertNotEqual(state_fingerprint(raw), before)
        self.assertIn("orphan", raw.sketches["sk1"]["entities"])

    def test_raising_op_leaves_state_byte_identical(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("raise")))
        before = be.state_fingerprint()
        with self.assertRaises(RuntimeError):
            be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(be.state_fingerprint(), before)

    def test_raising_op_leaves_model_digest_identical(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("raise")))
        before = be.state_digest()
        with self.assertRaises(RuntimeError):
            be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(be.state_digest(), before)

    def test_raising_op_undoes_every_specific_mutation(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("raise")))
        with self.assertRaises(RuntimeError):
            be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(be.inner.features, [])
        self.assertFalse(be.inner.solid_present)
        self.assertNotIn("orphan", be.inner.sketches["sk1"]["entities"])
        self.assertEqual(be.inner.sketches["sk1"]["dof"], 3)

    def test_original_exception_still_propagates(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("raise")))
        with self.assertRaises(RuntimeError) as ctx:
            be.apply(Extrude(sketch="sk1", distance=10.0))
        # The caller must see the kernel's error, not a rollback artefact.
        self.assertIn("boolean degeneracy", str(ctx.exception))

    def test_ok_false_op_is_rolled_back_too(self):
        # block-and-correct becomes ENFORCED rather than trusted.
        be = TransactionalBackend(_seeded(_CorruptingStub("soft")))
        before = be.state_fingerprint()
        result = be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertFalse(result.ok)
        self.assertEqual(be.state_fingerprint(), before)
        self.assertEqual(be.inner.features, [])

    def test_diagnostics_survive_the_rollback(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("soft")))
        result = be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(result.diagnostics[0].code, "kernel-fail")

    def test_rollback_on_failure_false_keeps_mutations(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("soft")),
                                  rollback_on_failure=False)
        before = be.state_fingerprint()
        be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertNotEqual(be.state_fingerprint(), before)

    def test_repeated_failures_do_not_drift(self):
        # Ten failures must land on the same state as zero failures.
        be = TransactionalBackend(_seeded(_CorruptingStub("raise")))
        before = be.state_fingerprint()
        for _ in range(10):
            with self.assertRaises(RuntimeError):
                be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(be.state_fingerprint(), before)

    def test_failure_count_is_tracked(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("soft")))
        be.apply(Extrude(sketch="sk1", distance=10.0))
        be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertEqual(be.failures, 2)


class DefaultSafetyTest(unittest.TestCase):
    """Wrapping an existing backend must change nothing about success."""

    def test_successful_ops_commit_normally(self):
        be = TransactionalBackend(StubBackend())
        self.assertTrue(be.apply(NewSketch(plane="XY")).ok)
        self.assertTrue(be.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=5)).ok)
        self.assertTrue(be.apply(Extrude(sketch="sk1", distance=10.0)).ok)
        self.assertTrue(be.inner.solid_present)
        self.assertEqual(be.failures, 0)

    def test_wrapped_replay_matches_unwrapped_replay(self):
        ops = [NewSketch(plane="XY"),
               AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=5.0),
               Extrude(sketch="sk1", distance=10.0)]
        plain, wrapped = StubBackend(), TransactionalBackend(StubBackend())
        for op in ops:
            plain.apply(op)
            wrapped.apply(op)
        # Same digest AND same full state: the wrapper is transparent.
        self.assertEqual(wrapped.state_digest(), plain.state_digest())
        self.assertEqual(state_fingerprint(wrapped.inner),
                         state_fingerprint(plain))

    def test_stub_native_failure_still_reports_normally(self):
        be = TransactionalBackend(StubBackend())
        be.apply(NewSketch(plane="XY"))
        result = be.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=-1.0))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-value")

    def test_delegates_protocol_methods(self):
        be = TransactionalBackend(StubBackend())
        be.apply(NewSketch(plane="XY"))
        self.assertIsInstance(be.query("summary"), dict)
        self.assertEqual(be.regenerate(), [])
        self.assertIsInstance(be.state_digest(), str)

    def test_reset_clears_inner_and_counters(self):
        be = TransactionalBackend(_seeded(_CorruptingStub("soft")))
        be.apply(Extrude(sketch="sk1", distance=10.0))
        be.reset()
        self.assertEqual(be.failures, 0)
        self.assertEqual(be.inner.sketches, {})

    def test_getattr_passes_through_backend_specifics(self):
        be = TransactionalBackend(StubBackend())
        self.assertFalse(be.solid_present)

    def test_backend_without_preflight_is_unaffected(self):
        # StubBackend has no validate_can_apply; it must not be required.
        self.assertFalse(hasattr(StubBackend(), "validate_can_apply"))
        self.assertIsNone(preflight(StubBackend(), NewSketch(plane="XY")))


class PreflightTest(unittest.TestCase):
    def setUp(self):
        class _Gated(StubBackend):
            def validate_can_apply(self, op):
                if isinstance(op, Extrude) and op.distance > 100:
                    return Diagnostic(Severity.ERROR, "preflight",
                                      "depth exceeds feasibility", None)
                return None

        self.be = TransactionalBackend(_seeded(_Gated()))

    def test_refused_op_returns_the_diagnostic(self):
        result = self.be.apply(Extrude(sketch="sk1", distance=1000.0))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "preflight")

    def test_refused_op_does_not_mutate(self):
        before = self.be.state_fingerprint()
        self.be.apply(Extrude(sketch="sk1", distance=1000.0))
        self.assertEqual(self.be.state_fingerprint(), before)

    def test_permitted_op_proceeds(self):
        self.assertTrue(self.be.apply(Extrude(sketch="sk1", distance=10.0)).ok)

    def test_truthy_and_string_verdicts_are_normalised(self):
        class _Str:
            def validate_can_apply(self, op):
                return "not allowed"

        diag = preflight(_Str(), "op")
        self.assertIsInstance(diag, Diagnostic)
        self.assertEqual(diag.code, "preflight")

        class _True:
            def validate_can_apply(self, op):
                return True  # "yes, you can apply"

        self.assertIsNone(preflight(_True(), "op"))


class SnapshotPrimitivesTest(unittest.TestCase):
    def test_snapshot_is_deep(self):
        be = _seeded(StubBackend())
        snap = snapshot_state(be)
        be.sketches["sk1"]["entities"].append("mutated")
        # A shallow copy would have shared the nested list.
        self.assertNotIn("mutated", snap["sketches"]["sk1"]["entities"])

    def test_restore_removes_attributes_created_after_the_snapshot(self):
        be = _seeded(StubBackend())
        snap = snapshot_state(be)
        be.leaked = "junk"
        restore_state(be, snap)
        self.assertFalse(hasattr(be, "leaked"))

    def test_excluded_attributes_survive_rollback(self):
        class _Recorded(StubBackend):
            _transaction_exclude = ("audit",)

            def __init__(self):
                super().__init__()
                self.audit = []

        be = _Recorded()
        snap = snapshot_state(be)
        be.audit.append("attempted")
        restore_state(be, snap)
        # The record of the attempt is the one thing that must NOT be undone.
        self.assertEqual(be.audit, ["attempted"])

    def test_excluded_attributes_are_outside_the_fingerprint(self):
        class _Recorded(StubBackend):
            _transaction_exclude = ("audit",)

            def __init__(self):
                super().__init__()
                self.audit = []

        be = _Recorded()
        before = state_fingerprint(be)
        be.audit.append("noise")
        self.assertEqual(state_fingerprint(be), before)

    def test_with_rollback_restores_on_exception(self):
        be = _seeded(StubBackend())
        before = state_fingerprint(be)
        with self.assertRaises(ValueError):
            with with_rollback(be):
                be.sketches["sk1"]["dof"] = 999
                raise ValueError("boom")
        self.assertEqual(state_fingerprint(be), before)

    def test_with_rollback_keeps_mutations_on_clean_exit(self):
        be = _seeded(StubBackend())
        with with_rollback(be):
            be.sketches["sk1"]["dof"] = 42
        self.assertEqual(be.sketches["sk1"]["dof"], 42)

    def test_fingerprint_is_stable_and_sensitive(self):
        be = _seeded(StubBackend())
        self.assertEqual(state_fingerprint(be), state_fingerprint(be))
        be.apply(Extrude(sketch="sk1", distance=10.0))
        self.assertNotEqual(state_fingerprint(be), state_fingerprint(StubBackend()))

    def test_fingerprint_survives_unpicklable_state(self):
        # A live kernel handle / lock must degrade to the structural fallback,
        # not explode.
        import threading

        class _Live:
            def __init__(self):
                self.lock = threading.Lock()
                self.n = 1

        be = _Live()
        first = state_fingerprint(be)
        self.assertEqual(first, state_fingerprint(be))
        be.n = 2
        self.assertNotEqual(state_fingerprint(be), first)

    def test_unsnapshotable_backend_raises_rollback_failed(self):
        class _Hostile:
            def __init__(self):
                self.thing = self

            def __deepcopy__(self, memo):
                raise TypeError("cannot copy")

        with self.assertRaises(RollbackFailed):
            snapshot_state(_Hostile())


class SelfcheckTest(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.io.backends.transaction import main
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
