"""Tests for the GuardrailGate hard validation gate + ErrorRecovery ladder."""

import unittest

from cisp.ops import AddCircle, AddRectangle, Boolean, Extrude, Fillet, NewSketch
from reliability.guardrails import ErrorRecovery, GuardrailGate, GuardrailLimits
from verifiers.verify import Severity


def _codes(diags):
    return {d.code for d in diags}


class _EdgeBackend:
    """Minimal backend exposing edge-length measurements for the fillet check."""

    def __init__(self, table):
        self._table = table

    def query(self, q):
        if q == "edge_length":
            return self._table
        return {}


class _BooleanBackend:
    def __init__(self, preview):
        self._preview = preview

    def query(self, q):
        if q == "boolean_preview":
            return self._preview
        return {}


class TestValidOps(unittest.TestCase):
    def test_valid_ops_pass(self):
        gate = GuardrailGate()
        self.assertEqual(gate.check(Extrude(sketch="sk1", distance=5.0)), [])
        self.assertEqual(gate.check(Fillet(edges=(1,), radius=2.0)), [])
        self.assertEqual(gate.check(AddCircle(sketch="sk1", r=3.0)), [])
        self.assertEqual(gate.check(AddRectangle(sketch="sk1", w=4.0, h=2.0)), [])
        # An unhandled op type is simply allowed (no rule -> no block).
        self.assertEqual(gate.check(NewSketch()), [])


class TestExtrude(unittest.TestCase):
    def test_zero_depth_blocked(self):
        diags = GuardrailGate().check(Extrude(sketch="sk1", distance=0.0))
        self.assertIn("extrude-nonpositive", _codes(diags))
        self.assertTrue(all(d.severity is Severity.ERROR for d in diags))

    def test_negative_depth_blocked(self):
        diags = GuardrailGate().check(Extrude(sketch="sk1", distance=-2.0))
        self.assertIn("extrude-nonpositive", _codes(diags))

    def test_out_of_range_depth_blocked(self):
        gate = GuardrailGate(GuardrailLimits(min_dim=1.0, max_dim=100.0))
        self.assertIn("dim-out-of-range",
                      _codes(gate.check(Extrude(sketch="sk1", distance=0.1))))
        self.assertIn("dim-out-of-range",
                      _codes(gate.check(Extrude(sketch="sk1", distance=1000.0))))


class TestFillet(unittest.TestCase):
    def test_nonpositive_radius_blocked(self):
        self.assertIn("fillet-nonpositive",
                      _codes(GuardrailGate().check(Fillet(edges=(1,), radius=0.0))))

    def test_oversize_vs_edge_length_blocked(self):
        gate = GuardrailGate()
        backend = _EdgeBackend({"e1": 3.0})
        diags = gate.check(Fillet(edges=("e1",), radius=5.0), backend=backend)
        self.assertIn("fillet-too-large", _codes(diags))

    def test_radius_within_edge_length_passes(self):
        gate = GuardrailGate()
        backend = _EdgeBackend({"e1": 10.0})
        self.assertEqual(gate.check(Fillet(edges=("e1",), radius=2.0), backend), [])

    def test_measurement_skipped_without_backend(self):
        # No backend -> the length-dependent check is skipped cleanly; a radius
        # that would be "too large" cannot be judged, so only value/range rules run.
        gate = GuardrailGate()
        self.assertEqual(gate.check(Fillet(edges=("e1",), radius=5.0)), [])

    def test_measurement_skipped_when_backend_lacks_data(self):
        gate = GuardrailGate()
        backend = _EdgeBackend({})  # measurement support present but empty
        self.assertEqual(gate.check(Fillet(edges=("e1",), radius=5.0), backend), [])


class TestCircleRectangle(unittest.TestCase):
    def test_circle_nonpositive_blocked(self):
        self.assertIn("circle-nonpositive",
                      _codes(GuardrailGate().check(AddCircle(sketch="s", r=0.0))))

    def test_rectangle_nonpositive_blocked(self):
        self.assertIn("rect-nonpositive",
                      _codes(GuardrailGate().check(AddRectangle(sketch="s", w=0.0, h=2.0))))

    def test_rectangle_out_of_range_blocked(self):
        gate = GuardrailGate(GuardrailLimits(min_dim=1.0, max_dim=50.0))
        self.assertIn("dim-out-of-range",
                      _codes(gate.check(AddRectangle(sketch="s", w=0.1, h=2.0))))


class TestBoolean(unittest.TestCase):
    def test_bad_kind_blocked(self):
        self.assertIn("boolean-bad-kind",
                      _codes(GuardrailGate().check(Boolean(kind="merge", target="a", tool="b"))))

    def test_null_body_blocked_with_measurement(self):
        gate = GuardrailGate()
        backend = _BooleanBackend({"a": 0.0})
        diags = gate.check(Boolean(kind="cut", target="a", tool="b"), backend)
        self.assertIn("boolean-nulls-body", _codes(diags))

    def test_null_body_skipped_without_backend(self):
        gate = GuardrailGate()
        self.assertEqual(gate.check(Boolean(kind="cut", target="a", tool="b")), [])

    def test_nonnull_boolean_passes(self):
        gate = GuardrailGate()
        backend = _BooleanBackend({"a": 12.0})
        self.assertEqual(gate.check(Boolean(kind="cut", target="a", tool="b"), backend), [])


class TestNoMutation(unittest.TestCase):
    def test_gate_does_not_mutate_backend(self):
        gate = GuardrailGate()
        backend = _EdgeBackend({"e1": 1.0})
        before = dict(backend._table)
        gate.check(Fillet(edges=("e1",), radius=5.0), backend)
        self.assertEqual(backend._table, before)


class TestErrorRecovery(unittest.TestCase):
    def test_stage_order(self):
        self.assertEqual(ErrorRecovery.stages(), ["detect", "handle", "recover"])

    def test_strategies_present(self):
        self.assertIn("retry-adjusted-params", ErrorRecovery.strategies("handle"))
        self.assertIn("rollback-feature-tree", ErrorRecovery.strategies("recover"))
        self.assertIn("over-constrained", ErrorRecovery.strategies("detect"))

    def test_unknown_stage_raises(self):
        with self.assertRaises(KeyError):
            ErrorRecovery.strategies("nope")

    def test_next_stage(self):
        self.assertEqual(ErrorRecovery.next_stage("detect"), "handle")
        self.assertEqual(ErrorRecovery.next_stage("handle"), "recover")
        self.assertIsNone(ErrorRecovery.next_stage("recover"))


if __name__ == "__main__":
    unittest.main()
