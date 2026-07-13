"""Tests for RequirementsCheck — the standalone requirements verifier.

Two regimes, mirroring the Contract tests:
  * a tiny fake backend that answers 'summary'/'measure' -> count and dimension
    requirements run for real (pass when met, ERROR when unmet);
  * the dependency-free StubBackend, which cannot answer 'measure' -> dimension
    requirements INFO-skip rather than failing.
"""

import unittest

from harnesscad.eval.verifiers.verify import Severity
from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.domain.spec.formalize import Requirement, RequirementSet, formalize
from harnesscad.eval.verifiers.requirements import RequirementsCheck, with_requirements


def _codes(report):
    return {d.code for d in report.diagnostics}


def _errors(report):
    return [d for d in report.diagnostics if d.severity is Severity.ERROR]


def _infos(report):
    return [d for d in report.diagnostics if d.severity is Severity.INFO]


class _FakeBackend:
    """Answers the numeric queries a real kernel would, for offline testing."""

    def __init__(self, feature_count=4, bbox=(100.0, 50.0, 8.0), volume=40000.0):
        self._feature_count = feature_count
        self._bbox = bbox
        self._volume = volume

    def query(self, q: str) -> dict:
        if q == "summary":
            return {"feature_count": self._feature_count, "solid_present": True}
        if q == "measure":
            return {"bbox": list(self._bbox), "volume": self._volume}
        return {}


def _build_plate(backend):
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


class TestCountRequirement(unittest.TestCase):
    def _reqset(self):
        return RequirementSet(requirements=[
            Requirement(kind="count", target=4, label="hole",
                        source_phrase="4 holes")])

    def test_passes_when_model_has_enough_features(self):
        report = RequirementsCheck(self._reqset()).check(
            _FakeBackend(feature_count=4))
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        self.assertEqual(_errors(report), [])

    def test_errors_when_model_has_too_few(self):
        report = RequirementsCheck(self._reqset()).check(
            _FakeBackend(feature_count=2))
        self.assertFalse(report.ok)
        self.assertIn("count-unmet", _codes(report))

    def test_label_specific_metric_preferred(self):
        # a 'metrics' query with a hole-specific count is used over feature_count
        class _M:
            def query(self, q):
                if q == "metrics":
                    return {"hole_count": 4}
                if q == "summary":
                    return {"feature_count": 1}  # would fail if used
                return {}

        report = RequirementsCheck(self._reqset()).check(_M())
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])


class TestDimensionRequirement(unittest.TestCase):
    def test_passes_within_tolerance(self):
        rs = RequirementSet(requirements=[
            Requirement(kind="dimension", target=100.0, tolerance=0.5,
                        label="length")])
        report = RequirementsCheck(rs).check(_FakeBackend(bbox=(100.2, 50, 8)))
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])

    def test_errors_out_of_tolerance(self):
        rs = RequirementSet(requirements=[
            Requirement(kind="dimension", target=100.0, tolerance=0.1,
                        label="length")])
        report = RequirementsCheck(rs).check(_FakeBackend(bbox=(999.0, 50, 8)))
        self.assertFalse(report.ok)
        self.assertIn("dimension-unmet", _codes(report))

    def test_default_tolerance_from_requirementset(self):
        rs = RequirementSet(requirements=[
            Requirement(kind="dimension", target=100.0, label="length"),
            Requirement(kind="tolerance", target=1.0)])
        report = RequirementsCheck(rs).check(_FakeBackend(bbox=(100.5, 50, 8)))
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])


class TestGracefulDegradation(unittest.TestCase):
    def test_dimension_info_skips_on_stub(self):
        # The stub answers 'summary' but not 'measure' -> dimension is skipped
        # (INFO), never an ERROR.
        backend = _build_plate(StubBackend())
        rs = RequirementSet(requirements=[
            Requirement(kind="dimension", target=100.0, label="length")])
        report = RequirementsCheck(rs).check(backend)
        self.assertTrue(report.ok)
        self.assertIn("req-skipped", {d.code for d in _infos(report)})

    def test_material_requirement_info_skips(self):
        rs = RequirementSet(requirements=[
            Requirement(kind="material", target="aluminium", label="material")])
        report = RequirementsCheck(rs).check(_FakeBackend())
        self.assertTrue(report.ok)
        self.assertIn("req-unmeasurable", {d.code for d in _infos(report)})


class TestEndToEnd(unittest.TestCase):
    def test_formalize_then_check(self):
        rs = formalize("an aluminium plate 100mm x 50mm x 8mm with 4 holes")
        report = RequirementsCheck(rs).check(
            _FakeBackend(feature_count=4, bbox=(100.0, 50.0, 8.0)))
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        # material is INFO-skipped, dimensions + count pass
        self.assertIn("req-unmeasurable", {d.code for d in _infos(report)})

    def test_with_requirements_appends_verifier(self):
        rs = formalize("a plate with 4 holes")
        base = []
        verifiers = with_requirements(base, rs)
        self.assertEqual(len(verifiers), 1)
        self.assertEqual(verifiers[0].name, "requirements")


if __name__ == "__main__":
    unittest.main()
