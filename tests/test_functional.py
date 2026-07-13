"""Tests for the functional-behaviour acceptance oracle (verifiers.functional).

Covers the spec-as-oracle: a declared FunctionalSpec is matched against the built
mechanism's kinematics.

  * a 1-DOF spec passes on a mechanism whose built mobility is 1;
  * a locked weld against a 1-DOF spec is a ``function-mismatch`` ERROR;
  * a wrong-DOF built mechanism is a mismatch; a forbidden motion (a ratchet
    built as a free revolute) is a mismatch;
  * graceful INFO-skip with no spec / no assembly / on the StubBackend;
  * FunctionalSpec round-trips and parses an "N-DOF" behavior phrase.

Deterministic; stdlib + the harness only.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.verifiers.assembly import AssemblyModel, Mate
from harnesscad.eval.quality.kinematics import JointIntent, ROT_NEG, TRANS_POS
from harnesscad.eval.verifiers.functional import (
    FunctionalSpec, FunctionalCheck, functional_diagnostics, with_functional,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _errors(report):
    return [d for d in report.diagnostics if d.severity is Severity.ERROR]


def _one_dof_spatial():
    # Two links, one revolute, spatial: M = 6(2-1) - (6-1) = 1.
    return AssemblyModel(
        parts=["a", "b"], mates=[Mate(kind="revolute", a="a", b="b")])


def _locked_weld():
    # Two links welded rigid, planar: M = 3(2-1) - 3 = 0.
    return AssemblyModel(
        parts=["a", "b"], mates=[Mate(kind="rigid", a="a", b="b")])


class TestMobilityAcceptance(unittest.TestCase):
    def test_matches_one_dof_spec(self):
        spec = FunctionalSpec(name="rotary hinge", required_mobility=1)
        report = FunctionalCheck(spec).check_mechanism(_one_dof_spatial())
        self.assertIn("built-mobility", _codes(report))
        self.assertIn("function-verified", _codes(report))
        self.assertNotIn("function-mismatch", _codes(report))
        self.assertTrue(report.ok)

    def test_locked_mechanism_that_should_move_mismatches(self):
        spec = FunctionalSpec(name="rotary hinge", required_mobility=1,
                              planar=True)
        report = FunctionalCheck(spec).check_mechanism(_locked_weld())
        self.assertIn("function-mismatch", _codes(report))
        self.assertFalse(report.ok)

    def test_wrong_dof_mismatches(self):
        # A free revolute (M=1) where the intent wants a fully-fixed bracket (0).
        spec = FunctionalSpec(name="bracket", required_mobility=0)
        report = FunctionalCheck(spec).check_mechanism(_one_dof_spatial())
        self.assertIn("function-mismatch", _codes(report))
        self.assertFalse(report.ok)


class TestForbiddenMotion(unittest.TestCase):
    def test_ratchet_built_as_free_revolute_mismatches(self):
        model = AssemblyModel(
            parts=["wheel", "pawl_arm"],
            mates=[Mate(kind="revolute", a="wheel", b="pawl_arm", name="pawl")])
        spec = FunctionalSpec(
            name="one-way ratchet",
            motions={"pawl": JointIntent(forbidden={ROT_NEG})})
        report = FunctionalCheck(spec).check_mechanism(model)
        self.assertIn("function-mismatch", _codes(report))
        self.assertFalse(report.ok)

    def test_one_way_joint_satisfies_intent(self):
        model = AssemblyModel(
            parts=["wheel", "pawl_arm"],
            mates=[Mate(kind="ratchet", a="wheel", b="pawl_arm", name="pawl")])
        spec = FunctionalSpec(
            name="one-way ratchet",
            motions={"pawl": JointIntent(forbidden={ROT_NEG})})
        report = FunctionalCheck(spec).check_mechanism(model)
        self.assertNotIn("function-mismatch", _codes(report))
        self.assertIn("function-verified", _codes(report))
        self.assertTrue(report.ok)

    def test_unknown_joint_ref_warns_not_errors(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="revolute", a="a", b="b", name="real")])
        spec = FunctionalSpec(motions={"ghost": JointIntent(forbidden={ROT_NEG})})
        report = FunctionalCheck(spec).check_mechanism(model)
        self.assertIn("unknown-joint-ref", _codes(report))
        self.assertEqual(_errors(report), [])


class TestBackendIntegration(unittest.TestCase):
    def test_no_spec_info_skips(self):
        report = FunctionalCheck(None).check(StubBackend(), None)
        self.assertIn("functional-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_stub_without_assembly_info_skips(self):
        spec = FunctionalSpec(required_mobility=1)
        report = FunctionalCheck(spec).check(StubBackend(), None)
        self.assertIn("functional-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_backend_payload_matches_spec(self):
        payload = {
            "parts": ["a", "b"],
            "mates": [{"kind": "revolute", "a": "a", "b": "b"}],
        }

        class _AssemblyBackend:
            def query(self, q):
                return payload if q == "assembly" else {}

        spec = FunctionalSpec(required_mobility=1)  # spatial revolute -> M=1
        report = FunctionalCheck(spec).check(_AssemblyBackend(), None)
        self.assertIn("function-verified", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_on_broken_backend(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("no query support")

        spec = FunctionalSpec(required_mobility=1)
        report = FunctionalCheck(spec).check(_Boom(), None)
        self.assertIn("functional-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestSpecRoundTrip(unittest.TestCase):
    def test_spec_round_trips(self):
        spec = FunctionalSpec(
            name="ratchet", required_mobility=1, planar=True,
            behavior="1-DOF one-way ratchet",
            motions={"pawl": JointIntent(forbidden={ROT_NEG}),
                     "slide": JointIntent(permitted={TRANS_POS})})
        back = FunctionalSpec.from_dict(spec.to_dict())
        self.assertEqual(back.name, "ratchet")
        self.assertEqual(back.required_mobility, 1)
        self.assertTrue(back.planar)
        self.assertEqual(back.motions["pawl"].forbidden, {ROT_NEG})
        self.assertEqual(back.motions["slide"].permitted, {TRANS_POS})
        self.assertEqual(back.to_dict(), spec.to_dict())

    def test_behavior_phrase_seeds_mobility(self):
        spec = FunctionalSpec.from_dict({"behavior": "1-DOF rotary"})
        self.assertEqual(spec.required_mobility, 1)

    def test_describe(self):
        spec = FunctionalSpec(required_mobility=1, planar=True)
        self.assertIn("1-DOF", spec.describe())


class TestWiring(unittest.TestCase):
    def test_with_functional_appends(self):
        base = ["x", "y"]
        result = with_functional(base, FunctionalSpec(name="m"))
        self.assertEqual(len(result), 3)
        self.assertEqual(result[-1].name, "functional")
        self.assertIsNotNone(result[-1].spec)
        self.assertEqual(base, ["x", "y"])  # original untouched

    def test_functional_diagnostics_backend_free(self):
        diags = functional_diagnostics(
            _one_dof_spatial(), FunctionalSpec(required_mobility=1))
        self.assertTrue(any(d.code == "built-mobility" for d in diags))


if __name__ == "__main__":
    unittest.main()
