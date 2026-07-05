"""Tests for ReferenceMatchCheck — the reference-match verifier.

All paths use fake metrics so no OCCT is required: a large volume/bbox delta is
flagged (WARNING/ERROR), a near-match passes, and an absent or unmeasurable
reference / model INFO-skips.
"""

import unittest

from verify import Severity
from checks_reference import ReferenceMatchCheck, with_reference
from ingest import ImportedPart


def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_sev(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


class _Backend:
    """Fake generated model answering 'metrics' like a real kernel."""

    def __init__(self, volume=1000.0, bbox=(20.0, 10.0, 5.0)):
        self._volume = volume
        self._bbox = bbox

    def query(self, q):
        if q in ("metrics", "measure"):
            return {"volume": self._volume, "bbox": list(self._bbox)}
        return {}


def _ref(volume=1000.0, bbox=(20.0, 10.0, 5.0)):
    return ImportedPart(
        path="ref.step", fmt="step", available=True,
        metrics={"volume": volume, "bbox": list(bbox)}, bbox=list(bbox))


class TestNearMatchPasses(unittest.TestCase):
    def test_exact_match_ok(self):
        report = ReferenceMatchCheck(_ref()).check(_Backend())
        self.assertTrue(report.ok, [d.to_dict() for d in report.diagnostics])
        self.assertIn("reference-match", _codes(report))

    def test_within_tolerance_ok(self):
        # 1% volume / bbox drift, default warn_tol is 2%.
        report = ReferenceMatchCheck(_ref()).check(
            _Backend(volume=1010.0, bbox=(20.1, 10.05, 5.02)))
        self.assertTrue(report.ok)
        self.assertEqual(_by_sev(report, Severity.ERROR), [])


class TestDeltaFlagged(unittest.TestCase):
    def test_large_volume_delta_errors(self):
        # 2x volume -> well beyond error_tol.
        report = ReferenceMatchCheck(_ref(volume=1000.0)).check(
            _Backend(volume=2000.0))
        self.assertFalse(report.ok)
        self.assertIn("volume-mismatch", _codes(report))

    def test_moderate_delta_warns(self):
        # 5% volume drift: past warn_tol (2%) but under error_tol (10%).
        report = ReferenceMatchCheck(_ref()).check(_Backend(volume=1050.0))
        self.assertTrue(report.ok)  # WARNING does not fail the report
        self.assertIn("volume-drift", _codes(report))
        self.assertTrue(_by_sev(report, Severity.WARNING))

    def test_bbox_delta_errors(self):
        report = ReferenceMatchCheck(_ref()).check(
            _Backend(bbox=(200.0, 10.0, 5.0)))
        self.assertFalse(report.ok)
        self.assertIn("bbox-mismatch", _codes(report))

    def test_custom_tolerances(self):
        # Tighten error_tol so a 5% drift errors.
        report = ReferenceMatchCheck(
            _ref(), warn_tol=0.01, error_tol=0.03).check(
            _Backend(volume=1050.0))
        self.assertFalse(report.ok)


class TestGracefulDegradation(unittest.TestCase):
    def test_absent_reference_info_skips(self):
        report = ReferenceMatchCheck(None).check(_Backend())
        self.assertTrue(report.ok)
        self.assertIn("reference-unavailable", _codes(report))

    def test_unavailable_imported_part_info_skips(self):
        ref = ImportedPart(path="x.step", fmt="step", available=False,
                           note="cadquery unavailable")
        report = ReferenceMatchCheck(ref).check(_Backend())
        self.assertTrue(report.ok)
        self.assertIn("reference-unavailable", _codes(report))

    def test_unmeasurable_model_info_skips(self):
        class _Blind:
            def query(self, q):
                return {}
        report = ReferenceMatchCheck(_ref()).check(_Blind())
        self.assertTrue(report.ok)
        self.assertIn("measurement-unavailable", _codes(report))

    def test_dict_reference_supported(self):
        report = ReferenceMatchCheck(
            {"volume": 1000.0, "bbox": [20.0, 10.0, 5.0]}).check(_Backend())
        self.assertTrue(report.ok)


class TestHelpers(unittest.TestCase):
    def test_with_reference_appends(self):
        verifiers = with_reference([], _ref())
        self.assertEqual(len(verifiers), 1)
        self.assertEqual(verifiers[0].name, "reference-match")

    def test_hausdorff_metric_scored_when_present(self):
        ref = {"volume": 1000.0, "bbox": [20.0, 10.0, 5.0],
               "hausdorff": 5.0, "hausdorff_ref_size": 20.0}
        # 25% Hausdorff -> error.
        report = ReferenceMatchCheck(ref).check(_Backend())
        self.assertIn("hausdorff-mismatch", _codes(report))


if __name__ == "__main__":
    unittest.main()
