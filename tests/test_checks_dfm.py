"""Tests for the standalone DFM critic (checks_dfm.DFMCheck).

Two backend regimes, mirroring the contract tests:

  * the dependency-free :class:`backends.stub.StubBackend`, which answers only
    ``'summary'`` / ``'sketch_dof'`` -> the envelope (bbox) checks and the
    face-level stubs must INFO-skip, and nothing may ERROR;
  * a tiny in-test :class:`_MeasuredBackend` that answers ``'measure'`` /
    ``'validity'`` like a real kernel -> a high-aspect-ratio bbox produces a
    WARNING (advisory), still never an ERROR.
"""

import unittest

from cisp.ops import NewSketch, AddRectangle, Extrude
from backends.stub import StubBackend
from verifiers.verify import Severity
from verifiers.dfm import DFMRules, DFMCheck, with_dfm


def _build_plate(backend):
    """Apply ops leaving `backend` holding one extruded rectangular plate."""
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


class _MeasuredBackend:
    """Minimal backend that answers the geometry queries a real kernel would,
    so the envelope checks can run without CadQuery/OCCT installed."""

    def __init__(self, bbox, volume=1.0, solid_present=True):
        self._bbox = bbox
        self._volume = volume
        self._solid_present = solid_present

    def query(self, q: str) -> dict:
        if q == "summary":
            return {"sketch_count": 1, "entity_count": 1,
                    "feature_count": 1, "solid_present": self._solid_present}
        if q == "measure":
            return {"volume": self._volume, "bbox": list(self._bbox)}
        if q == "validity":
            return {"manifold": True, "watertight": True,
                    "is_valid": True, "solid_present": self._solid_present}
        return {}


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


class TestDFMRulesRoundTrip(unittest.TestCase):
    def test_defaults_to_dict_from_dict(self):
        rules = DFMRules()
        self.assertEqual(DFMRules.from_dict(rules.to_dict()), rules)

    def test_custom_round_trip(self):
        rules = DFMRules(min_wall_thickness=2.5, max_aspect_ratio=8.0,
                         min_hole_diameter=3.0, min_draft_angle=2.0)
        restored = DFMRules.from_dict(rules.to_dict())
        self.assertEqual(restored, rules)
        self.assertEqual(restored.max_aspect_ratio, 8.0)

    def test_from_dict_none_and_partial(self):
        self.assertEqual(DFMRules.from_dict(None), DFMRules())
        # Unspecified keys fall back to defaults.
        partial = DFMRules.from_dict({"max_aspect_ratio": 5.0})
        self.assertEqual(partial.max_aspect_ratio, 5.0)
        self.assertEqual(partial.min_wall_thickness,
                         DFMRules().min_wall_thickness)


class TestDFMCheckOnStub(unittest.TestCase):
    """The stub answers only 'summary'/'sketch_dof': envelope checks INFO-skip,
    face-level stubs INFO-skip, and there is never an ERROR."""

    def test_stub_info_skips_and_no_error(self):
        backend = _build_plate(StubBackend())
        report = DFMCheck().check(backend, None)

        # Nothing this check emits may be an ERROR.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

        # No real measurement was possible -> no WARNINGs, only INFO.
        self.assertEqual(_by_severity(report, Severity.WARNING), [])
        codes = _codes(report)
        self.assertIn("envelope-skipped", codes)
        self.assertIn("dfm-not-yet-measurable", codes)


class TestDFMCheckOnMeasuredBackend(unittest.TestCase):
    def test_high_aspect_ratio_warns(self):
        # 200 x 4 x 4 -> aspect 50 > default 20.
        backend = _MeasuredBackend(bbox=(200.0, 4.0, 4.0), volume=3200.0)
        report = DFMCheck().check(backend, None)

        warnings = _by_severity(report, Severity.WARNING)
        codes = {d.code for d in warnings}
        self.assertIn("high-aspect-ratio", codes)

        # Advisory only: never an ERROR, report stays ok.
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertTrue(report.ok)

    def test_healthy_envelope_no_warning(self):
        # A tidy 20 x 10 x 5 plate: aspect 4 < 20, dims within limits.
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0)
        report = DFMCheck().check(backend, None)
        codes = _codes(report)
        self.assertNotIn("high-aspect-ratio", codes)
        self.assertNotIn("thin-envelope", codes)
        self.assertNotIn("oversized", codes)
        # Face-level hooks still advertised (solid present).
        self.assertIn("dfm-not-yet-measurable", codes)
        self.assertTrue(report.ok)

    def test_thin_envelope_and_oversized(self):
        # 2000 (too big) x 5 x 0.2 (too thin) -> oversized + thin-envelope +
        # also high aspect ratio.
        backend = _MeasuredBackend(bbox=(2000.0, 5.0, 0.2), volume=1.0)
        report = DFMCheck().check(backend, None)
        codes = _codes(report)
        self.assertIn("thin-envelope", codes)
        self.assertIn("oversized", codes)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])

    def test_custom_rules_change_outcome(self):
        # 200 x 4 x 4 -> aspect 50. With a loose max of 100 it should NOT warn.
        backend = _MeasuredBackend(bbox=(200.0, 4.0, 4.0), volume=3200.0)
        report = DFMCheck(DFMRules(max_aspect_ratio=100.0)).check(backend, None)
        self.assertNotIn("high-aspect-ratio", _codes(report))


class TestNeverEmitsError(unittest.TestCase):
    """DFM is advisory: no backend/geometry should ever yield an ERROR."""

    def test_various_backends_never_error(self):
        cases = [
            _build_plate(StubBackend()),
            _MeasuredBackend(bbox=(200.0, 4.0, 4.0)),           # high aspect
            _MeasuredBackend(bbox=(0.1, 0.1, 0.1)),             # tiny
            _MeasuredBackend(bbox=(0.0, 0.0, 0.0), solid_present=False),  # no solid
            _MeasuredBackend(bbox=(5000.0, 5000.0, 5000.0)),    # huge
        ]
        for backend in cases:
            report = DFMCheck().check(backend, None)
            self.assertEqual(_by_severity(report, Severity.ERROR), [],
                             f"unexpected ERROR for {backend}")
            self.assertTrue(report.ok)


class TestWithDFM(unittest.TestCase):
    def test_with_dfm_appends_dfm_check(self):
        base = ["a", "b"]
        result = with_dfm(base)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[-1], DFMCheck)
        self.assertEqual(result[-1].name, "dfm")
        # Original list untouched.
        self.assertEqual(base, ["a", "b"])

    def test_with_dfm_passes_rules(self):
        rules = DFMRules(max_aspect_ratio=3.0)
        result = with_dfm([], rules=rules)
        self.assertIs(result[-1].rules, rules)


if __name__ == "__main__":
    unittest.main()
