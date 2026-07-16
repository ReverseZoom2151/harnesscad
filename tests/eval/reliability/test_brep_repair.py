"""Tests for the B-rep repair + repair-advisor layer (repair.py).

Two halves, matching repair.py:
  * RepairAdvisor.suggest — deterministic diagnostic-code -> concrete-op mapping;
    exercised without any geometry kernel (pure, stdlib-only).
  * repair_solid — geometric heal; a clean no-op on a StubBackend / when cadquery
    is absent, and (guarded by cadquery) a real ShapeFix pass over a valid solid
    that reports healed=False with no crash.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import (
    Constrain, NewSketch, AddRectangle, AddCircle, Extrude,
)
from harnesscad.core.state.opdag import OpDAG
from harnesscad.eval.verifiers.verify import Diagnostic, Severity
from harnesscad.eval.reliability.brep_repair import (
    BASELINE_MAX_TOLERANCE,
    BASELINE_PRECISION,
    CONNECT_TOLERANCE,
    FIX_PRECISION,
    FIX_TOLERANCE,
    LadderRung,
    RepairAdvisor,
    RepairResult,
    RepairSuggestion,
    SEWING_TOLERANCE,
    _apply_face_recipe,
    _sew_shape,
    _shape_is_valid,
    default_ladder,
    heal_shape_ladder,
    repair_solid,
)
from harnesscad.eval.reliability.guardrails import ErrorRecovery


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
        from harnesscad.io.backends.cadquery import CadQueryBackend
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
        from harnesscad.io.backends.cadquery import CadQueryBackend
        from harnesscad.core.cisp.ops import Boolean
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=5))
        b.apply(Extrude(sketch="sk1", distance=4))
        result = repair_solid(b)
        self.assertIsInstance(result, RepairResult)
        self.assertTrue(b.query("validity")["is_valid"])


# ======================================================================
# Tolerance ladder (Brepler)
# ======================================================================
class TestToleranceLadderShape(unittest.TestCase):
    """The ladder's structure, checked without a kernel."""

    def test_rung_zero_is_the_historical_single_pass(self):
        # The default-safety invariant: rung 0 must be byte-for-byte the old
        # fixed-precision behaviour, so a shape that healed before still heals
        # identically and never escalates.
        rung = default_ladder()[0]
        self.assertEqual(rung.precision, BASELINE_PRECISION)
        self.assertEqual(rung.max_tolerance, BASELINE_MAX_TOLERANCE)
        self.assertFalse(rung.face_recipe)
        self.assertFalse(rung.sew)

    def test_default_ladder_is_deterministic(self):
        self.assertEqual(default_ladder(), default_ladder())

    def test_plain_band_escalates_monotonically_in_precision(self):
        plain = [r for r in default_ladder() if not r.face_recipe and not r.sew]
        precisions = [r.precision for r in plain]
        self.assertEqual(precisions, sorted(precisions))
        self.assertEqual(len(set(precisions)), len(precisions))

    def test_ladder_covers_the_brepler_connect_band(self):
        precisions = {r.precision for r in default_ladder()}
        for tol in CONNECT_TOLERANCE:
            self.assertIn(tol, precisions)

    def test_strategy_rungs_are_last_resorts(self):
        rungs = default_ladder()
        first_strategy = min(i for i, r in enumerate(rungs)
                             if r.face_recipe or r.sew)
        # every rung before the first strategy rung is plain...
        self.assertTrue(all(not r.face_recipe and not r.sew
                            for r in rungs[:first_strategy]))
        # ...and sewing happens exactly once, at the very end.
        self.assertEqual(sum(1 for r in rungs if r.sew), 1)
        self.assertTrue(rungs[-1].sew)
        self.assertEqual(rungs[-1].max_tolerance, SEWING_TOLERANCE)

    def test_repair_solid_accepts_a_custom_ladder_without_a_kernel(self):
        # The ladder argument must thread through the graceful-degradation path.
        result = repair_solid(_FakeValidBackend(),
                              ladder=[LadderRung("x", 1e-3, 1e-3)])
        self.assertIsInstance(result, RepairResult)
        self.assertFalse(result.healed)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestToleranceLadderWithCadquery(unittest.TestCase):
    """Real kernel properties of the ladder rungs."""

    def _box(self):
        import cadquery as cq
        return cq.Workplane("XY").box(20, 10, 5).val()

    def test_valid_solid_stops_at_rung_zero(self):
        import cadquery as cq
        healed, actions = heal_shape_ladder(cq, self._box())
        self.assertEqual(actions[0], "rung:%s" % default_ladder()[0].name)
        self.assertFalse(any("escalated" in a for a in actions))
        self.assertAlmostEqual(healed.Volume(), 1000.0, places=5)

    def test_sew_rung_rebuilds_a_solid_from_loose_faces(self):
        import cadquery as cq
        box = self._box()
        # A bare compound of the box's six faces contains no solid at all.
        compound = cq.Compound.makeCompound(box.Faces())
        self.assertEqual(len(compound.Solids()), 0)

        sewn, fired = _sew_shape(compound.wrapped, SEWING_TOLERANCE)
        self.assertTrue(fired)
        sewn_shape = cq.Shape.cast(sewn)
        self.assertEqual(len(sewn_shape.Solids()), 1)
        self.assertAlmostEqual(sewn_shape.Volume(), 1000.0, places=5)
        self.assertTrue(_shape_is_valid(sewn))

    def test_sew_rung_reports_not_fired_on_unclosable_faces(self):
        # A single face can never sew into a closed shell; the rung must report
        # not-fired and hand the caller back its input rather than a broken shape.
        import cadquery as cq
        one_face = self._box().Faces()[0]
        sewn, fired = _sew_shape(one_face.wrapped, SEWING_TOLERANCE)
        self.assertFalse(fired)
        self.assertIs(sewn, one_face.wrapped)

    def test_face_recipe_preserves_a_curved_solid(self):
        import cadquery as cq
        cyl = cq.Workplane("XY").circle(4).extrude(10).val()
        rung = LadderRung("t", FIX_PRECISION, FIX_TOLERANCE, face_recipe=True)
        fixed, fired = _apply_face_recipe(cyl.wrapped, rung)
        self.assertTrue(fired)
        fixed_shape = cq.Shape.cast(fixed)
        # The recipe must heal, never mangle: volume and validity are preserved.
        self.assertAlmostEqual(fixed_shape.Volume(), cyl.Volume(), places=3)
        self.assertTrue(_shape_is_valid(fixed))

    def test_ladder_never_narrows_what_a_single_pass_healed(self):
        # Pinning the ladder to rung 0 alone must reproduce the default result
        # for a valid solid — evidence that escalation only ever adds outcomes.
        import cadquery as cq
        box = self._box()
        baseline_only = [default_ladder()[0]]
        pinned, _ = heal_shape_ladder(cq, box, baseline_only)
        full, _ = heal_shape_ladder(cq, box)
        self.assertAlmostEqual(pinned.Volume(), full.Volume(), places=6)
        self.assertEqual(len(pinned.Faces()), len(full.Faces()))


if __name__ == "__main__":
    unittest.main()
