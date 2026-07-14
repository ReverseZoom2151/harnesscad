"""Tests for the model-completeness gate (verifiers.completeness).

Covers intrinsic metadata coverage (distinct from prompt-conformance):

  * a body missing a material is an ERROR ``missing-metadata``; a fully-specified
    model (name, units, material) passes cleanly;
  * per-hole tolerance/thread and per-critical-dimension tolerance gaps ERROR;
  * an un-enumerable category (no dimension list) INFO-skips, never ERRORs;
  * the dependency-free StubBackend degrades without crashing;
  * CompletenessRules round-trips through from_dict/to_dict and toggles families.

Deterministic; no network.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.verifiers.completeness import (
    CompletenessRules, CompletenessCheck, with_completeness,
)


def _codes(report):
    return {d.code for d in report.diagnostics}


def _errors(report):
    return [d for d in report.diagnostics if d.severity is Severity.ERROR]


def _wheres(report):
    return {d.where for d in report.diagnostics if d.code == "missing-metadata"}


class _MetaBackend:
    """A metadata-aware backend: answers 'summary' and 'metrics' with an
    injectable part-metadata view."""

    def __init__(self, summary=None, metrics=None):
        self._summary = summary or {}
        self._metrics = metrics or {}

    def query(self, q: str) -> dict:
        if q == "summary":
            return dict(self._summary)
        if q == "metrics":
            return dict(self._metrics)
        return {}


class TestMaterialCoverage(unittest.TestCase):
    def test_body_missing_material_errors(self):
        backend = _MetaBackend(metrics={
            "name": "bracket", "units": "mm",
            "bodies": [{"id": "b1"}],  # no material
        })
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("missing-metadata", _codes(report))
        self.assertIn("body[b1].material", _wheres(report))

    def test_fully_specified_model_passes(self):
        backend = _MetaBackend(metrics={
            "name": "bracket",
            "units": "mm",
            "bodies": [{"id": "b1", "material": "aluminium 6061"}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        self.assertEqual(_errors(report), [])

    def test_part_level_material_covers_implicit_body(self):
        # No 'bodies' list, but solid_present + a part-level material -> pass.
        backend = _MetaBackend(
            summary={"solid_present": True},
            metrics={"name": "p", "units": "mm", "material": "steel"},
        )
        report = CompletenessCheck().check(backend, None)
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])

    def test_implicit_body_without_material_errors(self):
        backend = _MetaBackend(
            summary={"solid_present": True},
            metrics={"name": "p", "units": "mm"},
        )
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("body.material", _wheres(report))


class TestPartFields(unittest.TestCase):
    def test_missing_name_and_units_error(self):
        backend = _MetaBackend(metrics={
            "material": "steel",
            "bodies": [{"id": "b1", "material": "steel"}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("part.name", _wheres(report))
        self.assertIn("part.units", _wheres(report))

    def test_toggling_families_off_suppresses_gaps(self):
        rules = CompletenessRules(
            require_part_name=False, require_units=False,
            require_dimension_tolerance=False)
        backend = _MetaBackend(metrics={
            "bodies": [{"id": "b1", "material": "steel"}],
        })
        report = CompletenessCheck(rules).check(backend, None)
        # name/units/dimension families disabled -> only material remains, and it
        # is satisfied -> clean pass.
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        self.assertNotIn("part.name", _wheres(report))


class TestHoleCoverage(unittest.TestCase):
    def test_hole_without_spec_errors(self):
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
            "holes": [{"diameter": 5.0}],  # no tolerance/thread
        })
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("hole[0]", _wheres(report))

    def test_hole_with_thread_passes(self):
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
            "holes": [{"diameter": 5.0, "thread": "M6x1.0"},
                      {"diameter": 3.2, "tolerance": 0.1}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])

    def test_ops_derived_hole_flagged(self):
        # A Hole op carries no tolerance/thread -> a genuine metadata gap.
        from harnesscad.core.cisp.ops import Hole
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
        })
        opdag = [Hole(face_or_sketch="sk1", diameter=5.0)]
        report = CompletenessCheck().check(backend, opdag)
        self.assertFalse(report.ok)
        self.assertIn("hole#0", _wheres(report))


class TestDimensionCoverage(unittest.TestCase):
    def test_dimension_missing_tolerance_errors(self):
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
            "critical_dimensions": [{"label": "length", "value": 100.0}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("dimension[length]", _wheres(report))

    def test_dimension_with_tolerance_passes(self):
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
            "critical_dimensions": [
                {"label": "length", "value": 100.0, "tolerance": 0.1}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])

    def test_no_dimension_list_is_unmeasurable_info(self):
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm",
            "bodies": [{"id": "b1", "material": "steel"}],
        })
        report = CompletenessCheck().check(backend, None)
        self.assertTrue(report.ok)
        self.assertIn("completeness-unmeasurable", _codes(report))


class TestGracefulDegradation(unittest.TestCase):
    def test_a_backend_with_no_metadata_surface_abstains_rather_than_failing(self):
        """No metadata surface -> UNMEASURABLE, not "incomplete".

        The stub (and the F-rep backend, and every backend the op vocabulary can
        actually drive) cannot report a name, units, a material or a hole
        callout for ANY part. This check used to call that a metadata GAP and
        raise a hard ERROR -- on 16 of 16 parts in the fleet audit, correct and
        broken alike. Precision 0.50 against a base rate of 0.50: zero
        information, and it inflated the fleet's headline recall besides.

        A rule that cannot tell a good part from a bad one must say so, not
        fail both. `completeness-unmeasurable` is the sentence it already had.
        """
        report = CompletenessCheck().check(StubBackend(), None)
        self.assertNotIn("missing-metadata", _codes(report))
        self.assertIn("completeness-unmeasurable", _codes(report))
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])

    def test_a_metadata_aware_backend_still_gets_every_error(self):
        """Abstention is scoped to ignorance. Where the rule can see, it fires."""
        backend = _MetaBackend(metrics={
            "name": "p", "units": "mm", "solid_present": True,
        })
        report = CompletenessCheck().check(backend, None)
        self.assertFalse(report.ok)
        self.assertIn("body.material", _wheres(report))

    def test_broken_backend_does_not_crash(self):
        class _Boom:
            def query(self, q):
                raise RuntimeError("no query support")

        report = CompletenessCheck().check(_Boom(), None)
        # nothing measurable at all -> clean skip.
        self.assertIn("completeness-skipped", _codes(report))
        self.assertTrue(report.ok)


class TestRulesAndWiring(unittest.TestCase):
    def test_rules_round_trip(self):
        rules = CompletenessRules(require_units=False, require_hole_spec=False)
        back = CompletenessRules.from_dict(rules.to_dict())
        self.assertEqual(back.to_dict(), rules.to_dict())
        self.assertFalse(back.require_units)
        self.assertFalse(back.require_hole_spec)

    def test_with_completeness_appends(self):
        base = ["x", "y"]
        result = with_completeness(base)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[-1].name, "completeness")
        self.assertEqual(base, ["x", "y"])  # original untouched


if __name__ == "__main__":
    unittest.main()
