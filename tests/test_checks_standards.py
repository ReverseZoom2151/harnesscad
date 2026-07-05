"""Tests for the standalone engineering-standards critic (checks_standards).

Covers:
  * StandardsRules round-trips through to_dict/from_dict.
  * nearest_standard / preferred-number nearest lookup.
  * StandardsCheck flags a Ø7.3 hole as non-standard and passes a Ø8.0 hole.
  * A non-standard fillet radius is flagged; a standard one passes.
  * Graceful degradation: an empty op-DAG on a StubBackend that cannot answer
    'metrics' INFO-skips and never ERRORs.
  * No finding is ever an ERROR (advisory verifier).
"""

import unittest

from cisp.ops import NewSketch, AddCircle, Fillet
from state.opdag import OpDAG
from backends.stub import StubBackend
from verify import Severity
from checks_standards import (
    StandardsRules, StandardsCheck, with_standards, nearest_standard,
)


def _opdag(*ops) -> OpDAG:
    dag = OpDAG()
    for op in ops:
        dag.append(op)
    return dag


def _codes(report):
    return {d.code for d in report.diagnostics}


def _no_error(testcase, report):
    testcase.assertFalse(
        any(d.severity is Severity.ERROR for d in report.diagnostics),
        f"unexpected ERROR: {[d.to_dict() for d in report.diagnostics]}")


class TestStandardsRulesRoundTrip(unittest.TestCase):
    def test_defaults_round_trip(self):
        r = StandardsRules()
        r2 = StandardsRules.from_dict(r.to_dict())
        self.assertEqual(r.to_dict(), r2.to_dict())

    def test_from_dict_overrides(self):
        r = StandardsRules.from_dict({"drill_sizes": [1.0, 2.0, 3.0],
                                      "abs_tol": 0.2})
        self.assertEqual(r.drill_sizes, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(r.abs_tol, 0.2)
        # untouched fields keep defaults
        self.assertEqual(r.fillet_radii, StandardsRules().fillet_radii)

    def test_from_dict_none(self):
        self.assertEqual(StandardsRules.from_dict(None).to_dict(),
                         StandardsRules().to_dict())


class TestNearestStandard(unittest.TestCase):
    def test_nearest_drill(self):
        drills = StandardsRules().drill_sizes
        # 7.3 is closest to 7.5 (not 7.0)
        self.assertEqual(nearest_standard(7.3, drills), 7.5)
        # exact hit
        self.assertEqual(nearest_standard(8.0, drills), 8.0)

    def test_nearest_preferred_number(self):
        preferred = StandardsRules().preferred_numbers()
        # 6.4 rounds to the preferred number 6.3
        self.assertEqual(nearest_standard(6.4, preferred), 6.3)

    def test_empty_series(self):
        self.assertIsNone(nearest_standard(5.0, []))


class TestStandardsCheckHoles(unittest.TestCase):
    def test_non_standard_hole_flagged(self):
        # Ø7.3 hole -> radius 3.65
        dag = _opdag(NewSketch(plane="XY"), AddCircle(sketch="sk1", r=3.65))
        report = StandardsCheck().check(StubBackend(), dag)
        self.assertIn("non-standard-hole", _codes(report))
        _no_error(self, report)
        # suggestion mentions the nearest standard 7.5
        msg = next(d.message for d in report.diagnostics
                   if d.code == "non-standard-hole")
        self.assertIn("7.5", msg)

    def test_standard_hole_passes(self):
        # Ø8.0 hole -> radius 4.0
        dag = _opdag(NewSketch(plane="XY"), AddCircle(sketch="sk1", r=4.0))
        report = StandardsCheck().check(StubBackend(), dag)
        self.assertNotIn("non-standard-hole", _codes(report))
        _no_error(self, report)


class TestStandardsCheckDimensions(unittest.TestCase):
    def test_non_standard_fillet_flagged(self):
        dag = _opdag(Fillet(edges=(), radius=2.7))
        report = StandardsCheck().check(StubBackend(), dag)
        self.assertIn("non-standard-fillet", _codes(report))
        _no_error(self, report)

    def test_standard_fillet_passes(self):
        dag = _opdag(Fillet(edges=(), radius=2.5))
        report = StandardsCheck().check(StubBackend(), dag)
        self.assertNotIn("non-standard-fillet", _codes(report))
        _no_error(self, report)


class TestStandardsMetricsHoles(unittest.TestCase):
    class _MetricsBackend:
        def query(self, q):
            if q == "metrics":
                return {"holes": [7.3]}
            return {}

    def test_metrics_holes_flagged(self):
        # No ops, but a backend that reports a measured Ø7.3 hole via 'metrics'.
        report = StandardsCheck().check(self._MetricsBackend(), OpDAG())
        self.assertIn("non-standard-hole", _codes(report))
        _no_error(self, report)


class TestStandardsDegrade(unittest.TestCase):
    def test_stub_empty_dag_info_skips(self):
        report = StandardsCheck().check(StubBackend(), OpDAG())
        self.assertIn("standards-skipped", _codes(report))
        _no_error(self, report)
        self.assertTrue(all(d.severity is Severity.INFO
                            for d in report.diagnostics))

    def test_none_opdag_does_not_crash(self):
        report = StandardsCheck().check(StubBackend(), None)
        _no_error(self, report)


class TestWithStandards(unittest.TestCase):
    def test_appends_verifier(self):
        base = []
        verifiers = with_standards(base)
        self.assertEqual(len(verifiers), 1)
        self.assertEqual(verifiers[0].name, "standards")
        self.assertEqual(base, [])  # original untouched


if __name__ == "__main__":
    unittest.main()
