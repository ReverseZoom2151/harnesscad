"""Tests for the conformance-certificate exporter (verifiers.report).

Covers:
  * rolling a mix of passing and failing verifiers into rule-by-rule records with
    an overall pass/fail verdict;
  * markdown + json rendering and a deterministic (stable) content-hash
    signature that changes when the report changes and is reproducible when it
    does not;
  * measurements pulled from query('metrics') and an op-DAG provenance hash;
  * a verifier that raises is recorded as a failing check, not a crash.

Deterministic; no network.
"""

import json
import unittest

from harnesscad.eval.verifiers.verify import Diagnostic, Severity, VerifyReport
from harnesscad.eval.verifiers.conformance_report import ConformanceReport


# --- fake verifiers --------------------------------------------------------
class _PassVerifier:
    name = "always-pass"

    def check(self, backend, opdag):
        return VerifyReport([Diagnostic(
            Severity.INFO, "noted", "everything looks fine", None)])


class _FailVerifier:
    name = "always-fail"

    def check(self, backend, opdag):
        return VerifyReport([
            Diagnostic(Severity.ERROR, "bad-thing", "a hard failure", "here"),
            Diagnostic(Severity.WARNING, "soft-thing", "an advisory", None),
        ])


class _BoomVerifier:
    name = "explodes"

    def check(self, backend, opdag):
        raise RuntimeError("kernel exploded")


class _FakeBackend:
    def __init__(self, metrics=None, digest="deadbeef"):
        self._metrics = metrics or {"bbox": [10.0, 20.0, 5.0], "volume": 1000.0}
        self._digest = digest

    def query(self, q):
        if q == "metrics":
            return dict(self._metrics)
        return {}

    def state_digest(self):
        return self._digest


class TestRollup(unittest.TestCase):
    def _report(self):
        return ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _FailVerifier()])

    def test_overall_verdict_fail_when_any_fails(self):
        report = self._report()
        self.assertEqual(report.verdict, "fail")
        self.assertFalse(report.ok)
        self.assertEqual(report.counts(),
                         {"total": 2, "passed": 1, "failed": 1})

    def test_all_pass_verdict(self):
        report = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _PassVerifier()])
        self.assertEqual(report.verdict, "pass")
        self.assertTrue(report.ok)

    def test_rule_records_carry_verdict_and_where(self):
        report = self._report()
        fail_check = next(c for c in report.checks if c.name == "always-fail")
        self.assertEqual(fail_check.verdict, "fail")
        rules = {r.rule: r for r in fail_check.rules}
        self.assertEqual(rules["bad-thing"].verdict, "fail")
        self.assertEqual(rules["bad-thing"].where, "here")
        self.assertEqual(rules["soft-thing"].verdict, "pass")  # warning != fail

    def test_measurements_and_provenance(self):
        report = self._report()
        self.assertEqual(report.measurements.get("volume"), 1000.0)
        self.assertEqual(report.provenance.get("model_digest"), "deadbeef")
        self.assertTrue(report.provenance.get("opdag_hash"))


class TestRendering(unittest.TestCase):
    def _report(self):
        return ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _FailVerifier()])

    def test_markdown_is_readable_certificate(self):
        md = self._report().to_markdown()
        self.assertIn("# HarnessCAD Conformance Certificate", md)
        self.assertIn("Verdict: FAIL", md)
        self.assertIn("always-pass", md)
        self.assertIn("always-fail", md)
        self.assertIn("bad-thing", md)
        self.assertIn("Signature", md)

    def test_json_round_trips_and_contains_signature(self):
        report = self._report()
        data = json.loads(report.to_json())
        self.assertEqual(data["verdict"], "fail")
        self.assertIn("signature", data)
        self.assertEqual(len(data["checks"]), 2)
        self.assertEqual(data["signature"], report.signature())


class TestSignatureDeterminism(unittest.TestCase):
    def test_signature_is_stable_across_identical_runs(self):
        a = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _FailVerifier()])
        b = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _FailVerifier()])
        self.assertEqual(a.signature(), b.signature())
        self.assertEqual(a.to_json(), b.to_json())

    def test_signature_changes_when_verdict_changes(self):
        passing = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier()])
        failing = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_FailVerifier()])
        self.assertNotEqual(passing.signature(), failing.signature())

    def test_signature_recompute_matches_body(self):
        report = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier()])
        # to_dict embeds the signature; recomputing over the body reproduces it.
        self.assertEqual(report.to_dict()["signature"], report.signature())


class TestGracefulDegradation(unittest.TestCase):
    def test_raising_verifier_recorded_as_failing_check(self):
        report = ConformanceReport.from_verifiers(
            _FakeBackend(), None, [_PassVerifier(), _BoomVerifier()])
        self.assertEqual(report.verdict, "fail")
        boom = next(c for c in report.checks if c.name == "explodes")
        self.assertEqual(boom.verdict, "fail")
        self.assertIn("verifier-error", {r.rule for r in boom.rules})

    def test_backend_without_digest_degrades(self):
        class _NoDigest:
            def query(self, q):
                return {}

        report = ConformanceReport.from_verifiers(
            _NoDigest(), None, [_PassVerifier()])
        self.assertIsNone(report.provenance.get("model_digest"))
        # still produces a valid, hashable certificate.
        self.assertTrue(report.signature())


if __name__ == "__main__":
    unittest.main()
