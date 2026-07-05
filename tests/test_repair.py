"""Tests for the B-rep repair + repair-advisor layer (repair.py).

Two halves, matching repair.py:
  * RepairAdvisor.suggest — deterministic diagnostic-code -> concrete-op mapping;
    exercised without any geometry kernel (pure, stdlib-only).
  * repair_solid — geometric heal; a clean no-op on a StubBackend / when cadquery
    is absent, and (guarded by cadquery) a real ShapeFix pass over a valid solid
    that reports healed=False with no crash.
"""

import unittest

from backends.stub import StubBackend
from cisp.ops import (
    Constrain, NewSketch, AddRectangle, AddCircle, Extrude,
)
from state.opdag import OpDAG
from verify import Diagnostic, Severity
from repair import RepairAdvisor, RepairResult, RepairSuggestion, repair_solid
from guardrails import ErrorRecovery


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


def _diag(code, message="msg", severity=Severity.ERROR, where=None):
    return Diagnostic(severity, code, message, where)


# ======================================================================
# RepairAdvisor
# ======================================================================
class TestRepairAdvisorMapping(unittest.TestCase):
    def setUp(self):
        self.advisor = RepairAdvisor()

    def test_over_constrained_suggests_dropping_a_constraint(self):
        suggestions = self.advisor.suggest([_diag("over-constrained", where="sk1")])
        self.assertEqual(len(suggestions), 1)
        s = suggestions[0]
        self.assertIsInstance(s, RepairSuggestion)
        self.assertEqual(s.code, "over-constrained")
        self.assertTrue(s.candidate_ops)
        # At least one candidate must be a drop of a constrain op.
        self.assertTrue(any(
            c.get("action") == "drop_op" and c.get("op") == "constrain"
            for c in s.candidate_ops))

    def test_over_constrained_names_concrete_constraint_from_opdag(self):
        dag = OpDAG()
        dag.append(NewSketch(plane="XY"))
        dag.append(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        dag.append(Constrain(kind="horizontal", a="e1"))
        dag.append(Constrain(kind="distance", a="e1", value=10.0))
        s = self.advisor.suggest([_diag("over-constrained", where="sk1")], opdag=dag)[0]
        drop = next(c for c in s.candidate_ops if c.get("action") == "drop_op")
        # The most-recent constraint (the distance) is the primary drop candidate.
        self.assertEqual(drop["kind"], "distance")
        self.assertEqual(drop["a"], "e1")

    def test_under_constrained_suggests_adding_a_constraint(self):
        s = self.advisor.suggest([_diag("under-constrained", "dof>0", Severity.WARNING,
                                        "sk1")])[0]
        self.assertTrue(any(
            c.get("action") == "add_op" and c.get("op") == "constrain"
            for c in s.candidate_ops))

    def test_invalid_brep_suggests_heal_or_fillet(self):
        s = self.advisor.suggest([_diag("invalid-brep")])[0]
        actions = {c.get("action") for c in s.candidate_ops}
        self.assertIn("repair_solid", actions)
        # a small fillet for continuity is offered too
        self.assertTrue(any(c.get("op") == "fillet" for c in s.candidate_ops))

    def test_self_intersection_suggests_heal_or_offset(self):
        s = self.advisor.suggest([_diag("self-intersection")])[0]
        self.assertTrue(s.candidate_ops)
        self.assertIn("repair_solid", {c.get("action") for c in s.candidate_ops})

    def test_empty_solid_suggests_checking_closed_profile(self):
        s = self.advisor.suggest([_diag("empty-solid")])[0]
        blob = " ".join(c.get("detail", "") for c in s.candidate_ops).lower()
        self.assertIn("closed", blob)

    def test_dim_out_of_range_suggests_loosening(self):
        s = self.advisor.suggest([_diag("dim-out-of-range")])[0]
        self.assertTrue(s.candidate_ops)

    def test_unknown_code_gets_generic_but_nonempty_suggestion(self):
        s = self.advisor.suggest([_diag("totally-made-up-code")])[0]
        self.assertEqual(s.code, "totally-made-up-code")
        self.assertTrue(s.candidate_ops)  # never empty

    def test_every_suggestion_carries_a_valid_recovery_ladder_rung(self):
        codes = ["over-constrained", "invalid-brep", "empty-solid",
                 "self-intersection", "boolean-fail", "some-unknown"]
        diags = [_diag(c) for c in codes]
        for s in self.advisor.suggest(diags):
            self.assertEqual(set(s.recovery), {"detect", "handle", "recover"})
            # Each strategy name must exist in the composed ErrorRecovery ladder.
            self.assertIn(s.recovery["detect"], ErrorRecovery.strategies("detect"))
            self.assertIn(s.recovery["handle"], ErrorRecovery.strategies("handle"))
            self.assertIn(s.recovery["recover"], ErrorRecovery.strategies("recover"))

    def test_suggest_is_deterministic_and_order_preserving(self):
        diags = [_diag("over-constrained", where="sk1"), _diag("empty-solid")]
        a = [s.to_dict() for s in self.advisor.suggest(diags)]
        b = [s.to_dict() for s in self.advisor.suggest(diags)]
        self.assertEqual(a, b)
        self.assertEqual([s["code"] for s in a], ["over-constrained", "empty-solid"])

    def test_empty_diagnostics_yields_empty_suggestions(self):
        self.assertEqual(self.advisor.suggest([]), [])
        self.assertEqual(self.advisor.suggest(None), [])


# ======================================================================
# repair_solid — no-op paths (no kernel required)
# ======================================================================
class TestRepairSolidNoOp(unittest.TestCase):
    def test_stub_backend_is_a_clean_noop(self):
        # StubBackend exposes no 'validity' query and no _solids -> no-op, no crash.
        b = StubBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10))
        b.apply(Extrude(sketch="sk1", distance=5))
        result = repair_solid(b)
        self.assertIsInstance(result, RepairResult)
        self.assertFalse(result.healed)
        self.assertEqual(result.actions, [])
        self.assertTrue(result.note)  # a clear note explaining the no-op

    def test_no_solid_present_is_a_clean_noop(self):
        b = StubBackend()
        result = repair_solid(b)
        self.assertFalse(result.healed)
        self.assertIn("nothing to repair", result.note)

    def test_result_to_dict_is_serialisable(self):
        result = repair_solid(StubBackend())
        d = result.to_dict()
        self.assertEqual(
            set(d),
            {"healed", "actions", "before_validity", "after_validity", "diff", "note"})


class _FakeValidBackend:
    """A backend that reports a valid solid but exposes no OCCT solids.

    Exercises the 'validity present but no _solids to heal' graceful-skip branch
    without needing cadquery.
    """

    def query(self, q):
        if q == "validity":
            return {"manifold": True, "watertight": True,
                    "is_valid": True, "solid_present": True}
        return {}


class TestRepairSolidDegrades(unittest.TestCase):
    def test_validity_present_but_no_solids_skips_cleanly(self):
        result = repair_solid(_FakeValidBackend())
        self.assertFalse(result.healed)
        self.assertTrue(result.note)
        self.assertTrue(result.before_validity.get("solid_present"))


# ======================================================================
# repair_solid — real OCCT path (guarded by cadquery)
# ======================================================================
@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRepairSolidWithCadquery(unittest.TestCase):
    def _plate(self):
        from backends.cadquery_backend import CadQueryBackend
        b = CadQueryBackend()
        self.assertTrue(b.apply(NewSketch(plane="XY")).ok)
        self.assertTrue(b.apply(AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10)).ok)
        self.assertTrue(b.apply(Extrude(sketch="sk1", distance=5)).ok)
        return b

    def test_valid_solid_reports_healed_false_and_runs_shapefix(self):
        b = self._plate()
        # sanity: the plate is a valid solid to start with
        self.assertTrue(b.query("validity")["is_valid"])
        result = repair_solid(b)
        self.assertIsInstance(result, RepairResult)
        # A valid solid should not be reported as healed...
        self.assertFalse(result.healed)
        # ...but the ShapeFix path DID run and produced an after-validity report.
        self.assertTrue(result.after_validity.get("solid_present"))
        self.assertTrue(result.after_validity.get("is_valid"))
        # and the backend is still valid / undamaged afterwards
        self.assertTrue(b.query("validity")["is_valid"])

    def test_repair_solid_never_raises_on_multi_solid_model(self):
        # Two disjoint solids in the backend -> _combined() is a Compound; the
        # heal path must handle it without crashing.
        from backends.cadquery_backend import CadQueryBackend
        from cisp.ops import Boolean
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=5))
        b.apply(Extrude(sketch="sk1", distance=4))
        result = repair_solid(b)
        self.assertIsInstance(result, RepairResult)
        self.assertTrue(b.query("validity")["is_valid"])


if __name__ == "__main__":
    unittest.main()
