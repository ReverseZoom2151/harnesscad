"""Tests for the standalone compliance critic (checks_compliance).

Covers:
  * ComplianceRules round-trips through to_dict/from_dict.
  * ComplianceCheck flags an oversize part (heuristic export-control review) and
    passes a compliant one.
  * ComplianceCheck flags an over-limit additive overhang and passes a compliant
    one.
  * Graceful degradation: a StubBackend that answers neither 'measure' nor
    'metrics' INFO-skips both checks and never ERRORs.
  * No finding is ever an ERROR (advisory verifier, never blocks).
"""

import unittest

from backends.stub import StubBackend
from verify import Severity
from checks_compliance import (
    ComplianceRules, ComplianceCheck, with_compliance,
)


class _FakeBackend:
    """Answers 'measure' / 'metrics' with caller-supplied dicts, like a kernel."""

    def __init__(self, measure=None, metrics=None):
        self._measure = measure
        self._metrics = metrics

    def query(self, q):
        if q == "measure" and self._measure is not None:
            return self._measure
        if q == "metrics" and self._metrics is not None:
            return self._metrics
        return {}


def _codes(report):
    return {d.code for d in report.diagnostics}


def _no_error(testcase, report):
    testcase.assertFalse(
        any(d.severity is Severity.ERROR for d in report.diagnostics),
        f"unexpected ERROR: {[d.to_dict() for d in report.diagnostics]}")


class TestComplianceRulesRoundTrip(unittest.TestCase):
    def test_defaults_round_trip(self):
        r = ComplianceRules()
        r2 = ComplianceRules.from_dict(r.to_dict())
        self.assertEqual(r.to_dict(), r2.to_dict())

    def test_from_dict_overrides(self):
        r = ComplianceRules.from_dict({"itar_size_threshold": 250.0,
                                       "region": "EU",
                                       "max_overhang_angle": 30.0})
        self.assertAlmostEqual(r.itar_size_threshold, 250.0)
        self.assertEqual(r.region, "EU")
        self.assertAlmostEqual(r.max_overhang_angle, 30.0)

    def test_from_dict_none(self):
        self.assertEqual(ComplianceRules.from_dict(None).to_dict(),
                         ComplianceRules().to_dict())


class TestExportControl(unittest.TestCase):
    def test_oversize_flagged(self):
        backend = _FakeBackend(measure={"volume": 1.0, "bbox": [700.0, 50.0, 50.0]})
        report = ComplianceCheck().check(backend, None)
        self.assertIn("export-control-review", _codes(report))
        _no_error(self, report)
        # rationale is clearly heuristic, not authoritative
        msg = next(d.message for d in report.diagnostics
                   if d.code == "export-control-review")
        self.assertIn("HEURISTIC", msg)

    def test_compliant_size_passes(self):
        backend = _FakeBackend(measure={"volume": 1.0, "bbox": [100.0, 50.0, 20.0]})
        report = ComplianceCheck().check(backend, None)
        self.assertNotIn("export-control-review", _codes(report))
        _no_error(self, report)

    def test_regional_limit_flagged(self):
        rules = ComplianceRules(region="EU")  # EU ceiling default 800
        backend = _FakeBackend(measure={"volume": 1.0, "bbox": [850.0, 50.0, 50.0]})
        report = ComplianceCheck(rules).check(backend, None)
        self.assertIn("regional-limit", _codes(report))
        _no_error(self, report)


class TestAdditiveOverhang(unittest.TestCase):
    def test_over_limit_overhang_flagged(self):
        backend = _FakeBackend(metrics={"max_overhang_deg": 60.0})
        report = ComplianceCheck().check(backend, None)
        self.assertIn("unsupported-overhang", _codes(report))
        _no_error(self, report)

    def test_compliant_overhang_passes(self):
        backend = _FakeBackend(metrics={"max_overhang_deg": 30.0})
        report = ComplianceCheck().check(backend, None)
        self.assertNotIn("unsupported-overhang", _codes(report))
        _no_error(self, report)

    def test_overhang_from_angle_list(self):
        backend = _FakeBackend(metrics={"overhang_angles": [10.0, 55.0, 20.0]})
        report = ComplianceCheck().check(backend, None)
        self.assertIn("unsupported-overhang", _codes(report))
        _no_error(self, report)

    def test_below_min_feature_flagged(self):
        backend = _FakeBackend(metrics={"min_feature_mm": 0.2})
        report = ComplianceCheck().check(backend, None)
        self.assertIn("below-min-feature", _codes(report))
        _no_error(self, report)


class TestComplianceDegrade(unittest.TestCase):
    def test_stub_info_skips_both(self):
        report = ComplianceCheck().check(StubBackend(), None)
        codes = _codes(report)
        self.assertIn("export-control-skipped", codes)
        self.assertIn("overhang-skipped", codes)
        _no_error(self, report)
        self.assertTrue(all(d.severity is Severity.INFO
                            for d in report.diagnostics))


class TestWithCompliance(unittest.TestCase):
    def test_appends_verifier(self):
        base = []
        verifiers = with_compliance(base)
        self.assertEqual(len(verifiers), 1)
        self.assertEqual(verifiers[0].name, "compliance")
        self.assertEqual(base, [])


if __name__ == "__main__":
    unittest.main()
