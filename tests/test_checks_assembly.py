"""Tests for the standalone assembly / mate DOF solver (checks_assembly).

Covers, per the blueprint's third verifier family:

  * the DOF-count arithmetic (6 * n_parts - removed) and its over / under /
    well classification, on an :class:`AssemblyModel` built directly (no
    backend);
  * mate-satisfaction residuals -> ``unsatisfied-mate`` ERROR;
  * graceful INFO-skip on a dependency-free :class:`backends.stub.StubBackend`
    that has no ``'assembly'`` query, with no crash and no ERROR.
"""

import unittest

from backends.stub import StubBackend
from verify import Severity
from checks_assembly import (
    MATE_DOF, mate_dof, Mate, AssemblyModel, AssemblyCheck,
    assembly_diagnostics, with_assembly,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


class TestMateDofTable(unittest.TestCase):
    def test_table_values(self):
        self.assertEqual(MATE_DOF["rigid"], 6)
        self.assertEqual(MATE_DOF["revolute"], 5)
        self.assertEqual(MATE_DOF["slider"], 5)
        self.assertEqual(MATE_DOF["cylindrical"], 4)
        self.assertEqual(MATE_DOF["planar"], 3)

    def test_mate_dof_lookup_is_case_insensitive(self):
        self.assertEqual(mate_dof("REVOLUTE"), 5)
        self.assertEqual(mate_dof(" Planar "), 3)
        self.assertIsNone(mate_dof("nonsense"))


class TestResidualDofMath(unittest.TestCase):
    def test_two_parts_revolute_is_under_constrained(self):
        # 6*2 - 5 = 7 free DOF -> under-constrained.
        model = AssemblyModel(
            parts=["base", "arm"],
            mates=[Mate(kind="revolute", a="base", b="arm")],
        )
        self.assertEqual(model.residual_dof(), 7)
        self.assertEqual(model.classify(), "under")

        report = AssemblyCheck().check_model(model)
        self.assertIn("under-constrained", _codes(report))
        self.assertIn("assembly-dof", _codes(report))
        # under-constrained is only a WARNING: report stays ok.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_grounding_removes_six_dof(self):
        # Ground the base: 6*2 - 6 - 5 = 1 -> still under-constrained (hinge).
        model = AssemblyModel(
            parts=["base", "arm"],
            mates=[Mate(kind="revolute", a="base", b="arm")],
            grounded=["base"],
        )
        self.assertEqual(model.residual_dof(), 1)
        self.assertEqual(model.classify(), "under")

    def test_well_constrained_zero_dof(self):
        # base grounded (-6) + rigid arm (-6): 6*2 - 6 - 6 = 0 -> well.
        model = AssemblyModel(
            parts=["base", "arm"],
            mates=[Mate(kind="rigid", a="base", b="arm")],
            grounded=["base"],
        )
        self.assertEqual(model.residual_dof(), 0)
        self.assertEqual(model.classify(), "well")
        report = AssemblyCheck().check_model(model)
        self.assertNotIn("over-constrained", _codes(report))
        self.assertNotIn("under-constrained", _codes(report))
        self.assertTrue(report.ok)

    def test_over_constrained_is_error(self):
        # Three rigid mates on two parts: 6*2 - 18 = -6 -> over-constrained.
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="rigid", a="a", b="b"),
                   Mate(kind="rigid", a="a", b="b"),
                   Mate(kind="rigid", a="a", b="b")],
        )
        self.assertEqual(model.residual_dof(), -6)
        self.assertEqual(model.classify(), "over")
        report = AssemblyCheck().check_model(model)
        self.assertIn("over-constrained", _codes(report))
        self.assertFalse(report.ok)  # over-constrained is an ERROR


class TestUnknownAndBadMates(unittest.TestCase):
    def test_unknown_mate_kind_warns_and_is_excluded(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="wobble", a="a", b="b")],
        )
        # Unknown kind removes no DOF: 6*2 - 0 = 12.
        self.assertEqual(model.residual_dof(), 12)
        report = AssemblyCheck().check_model(model)
        self.assertIn("unknown-mate", _codes(report))
        # It is only a WARNING, not fatal.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])

    def test_dangling_part_reference_warns(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="revolute", a="a", b="ghost")],
        )
        report = AssemblyCheck().check_model(model)
        self.assertIn("bad-part-ref", _codes(report))


class TestMateSatisfaction(unittest.TestCase):
    def test_unsatisfied_mate_errors(self):
        # Two anchors that should be coincident, but the arm is translated far.
        model = AssemblyModel(
            parts=["base", "arm"],
            mates=[Mate(kind="revolute", a="base", b="arm",
                        point_a=(0.0, 0.0, 0.0), point_b=(0.0, 0.0, 0.0),
                        tol=0.01, name="pivot")],
            transforms={"arm": {"translate": [10.0, 0.0, 0.0]}},
        )
        # Gap = 10 mm >> tol.
        self.assertAlmostEqual(model.mate_residual(model.mates[0]), 10.0)
        report = AssemblyCheck().check_model(model)
        self.assertIn("unsatisfied-mate", _codes(report))
        self.assertFalse(report.ok)

    def test_satisfied_mate_has_no_error(self):
        # Anchors coincide after placement (both land at origin): no ERROR.
        model = AssemblyModel(
            parts=["base", "arm"],
            mates=[Mate(kind="revolute", a="base", b="arm",
                        point_a=(0.0, 0.0, 0.0), point_b=(-5.0, 0.0, 0.0),
                        tol=1e-6)],
            transforms={"arm": {"translate": [5.0, 0.0, 0.0]}},
        )
        self.assertAlmostEqual(model.mate_residual(model.mates[0]), 0.0)
        report = AssemblyCheck().check_model(model)
        self.assertNotIn("unsatisfied-mate", _codes(report))

    def test_residual_none_when_no_anchors(self):
        model = AssemblyModel(parts=["a", "b"],
                              mates=[Mate(kind="rigid", a="a", b="b")])
        self.assertIsNone(model.mate_residual(model.mates[0]))

    def test_precomputed_residual_is_used(self):
        model = AssemblyModel(
            parts=["a", "b"],
            mates=[Mate(kind="rigid", a="a", b="b", residual=2.5, tol=0.1)],
        )
        self.assertEqual(model.mate_residual(model.mates[0]), 2.5)
        self.assertIn("unsatisfied-mate",
                      _codes(AssemblyCheck().check_model(model)))


class TestFromDict(unittest.TestCase):
    def test_from_dict_round_trip_shape(self):
        raw = {
            "parts": [{"id": "base"}, "arm"],
            "mates": [{"kind": "revolute", "a": "base", "b": "arm"}],
            "transforms": {"arm": {"translate": [1, 2, 3]}},
            "grounded": ["base"],
        }
        model = AssemblyModel.from_dict(raw)
        self.assertEqual(model.parts, ["base", "arm"])
        self.assertEqual(len(model.mates), 1)
        self.assertEqual(model.mates[0].kind, "revolute")
        self.assertEqual(model.grounded, ["base"])
        # 6*2 - 6(grounded) - 5(revolute) = 1.
        self.assertEqual(model.residual_dof(), 1)

    def test_from_empty_dict(self):
        model = AssemblyModel.from_dict({})
        self.assertEqual(model.parts, [])
        self.assertEqual(model.mates, [])


class _AssemblyBackend:
    """Minimal backend that answers query('assembly') like an assembly-aware
    kernel would, so the verifier can run without a real backend."""

    def __init__(self, payload):
        self._payload = payload

    def query(self, q: str) -> dict:
        if q == "assembly":
            return self._payload
        return {}


class TestAssemblyCheckOnBackends(unittest.TestCase):
    def test_stub_info_skips_cleanly(self):
        # The stub answers only 'summary'/'sketch_dof' -> INFO skip, no crash,
        # no ERROR.
        backend = StubBackend()
        report = AssemblyCheck().check(backend, None)
        self.assertIn("assembly-skipped", _codes(report))
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_empty_assembly_query_skips(self):
        report = AssemblyCheck().check(_AssemblyBackend({}), None)
        self.assertIn("assembly-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_backend_payload_over_constrained(self):
        payload = {
            "parts": ["a", "b"],
            "mates": [{"kind": "rigid", "a": "a", "b": "b"},
                      {"kind": "rigid", "a": "a", "b": "b"},
                      {"kind": "rigid", "a": "a", "b": "b"}],
        }
        report = AssemblyCheck().check(_AssemblyBackend(payload), None)
        self.assertIn("over-constrained", _codes(report))
        self.assertFalse(report.ok)

    def test_single_part_is_trivial(self):
        report = AssemblyCheck().check(
            _AssemblyBackend({"parts": ["solo"], "mates": []}), None)
        self.assertIn("assembly-trivial", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_on_broken_backend(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("no query support")

        report = AssemblyCheck().check(_Boom(), None)
        self.assertIn("assembly-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestWithAssembly(unittest.TestCase):
    def test_appends_assembly_check(self):
        base = ["x", "y"]
        result = with_assembly(base)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[-1], AssemblyCheck)
        self.assertEqual(result[-1].name, "assembly")
        self.assertEqual(base, ["x", "y"])  # original untouched

    def test_assembly_diagnostics_is_backend_free(self):
        model = AssemblyModel(parts=["a", "b"],
                              mates=[Mate(kind="planar", a="a", b="b")])
        diags = assembly_diagnostics(model)
        self.assertTrue(any(d.code == "assembly-dof" for d in diags))


if __name__ == "__main__":
    unittest.main()
