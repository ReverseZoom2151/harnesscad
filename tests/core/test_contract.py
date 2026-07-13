"""Tests for the Contract layer (the "Contractor" model).

Two backend regimes are exercised:

  * the dependency-free :class:`backends.stub.StubBackend`, which answers only
    ``'summary'`` / ``'sketch_dof'`` -> dimension/volume/topology checks must
    INFO-skip while feature-count checks still run;
  * a tiny in-test fake backend that *does* answer ``'measure'`` /
    ``'validity'`` -> dimension/volume/topology checks run for real, so an
    out-of-tolerance contract produces an ERROR.
"""

import unittest

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.core.contract import (
    Contract, Tolerance, MassSpec, ContractCheck,
    contract_from_brief_schema, register_predicate,
)


def _build_plate(backend):
    """Apply ops that leave `backend` holding one extruded rectangular plate."""
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


class _MeasuredBackend:
    """Minimal backend that answers the geometry queries a real kernel would,
    so tolerance/topology checks can run without CadQuery/OCCT installed."""

    def __init__(self, bbox, volume, manifold=True, watertight=True,
                 is_valid=True, feature_count=1):
        self._bbox = bbox
        self._volume = volume
        self._manifold = manifold
        self._watertight = watertight
        self._is_valid = is_valid
        self._feature_count = feature_count

    def query(self, q: str) -> dict:
        if q == "summary":
            return {"sketch_count": 1, "entity_count": 1,
                    "feature_count": self._feature_count, "solid_present": True}
        if q == "measure":
            return {"volume": self._volume, "bbox": list(self._bbox)}
        if q == "validity":
            return {"manifold": self._manifold, "watertight": self._watertight,
                    "is_valid": self._is_valid, "solid_present": True}
        return {}


def _codes(report):
    return {d.code for d in report.diagnostics}


def _errors(report):
    return [d for d in report.diagnostics if d.severity is Severity.ERROR]


def _infos(report):
    return [d for d in report.diagnostics if d.severity is Severity.INFO]


class TestContractOnStub(unittest.TestCase):
    """A matching contract against the stub: feature check passes, geometry
    checks INFO-skip (the stub has no 'measure'/'validity')."""

    def _contract(self):
        return Contract(
            name="plate",
            bbox={"x": Tolerance(20.0, 0.1),
                  "y": Tolerance(10.0, 0.1),
                  "z": Tolerance(5.0, 0.1)},
            volume=Tolerance(1000.0, 1.0),
            min_features=1,
            require_manifold=True,
            no_self_intersections=True,
        )

    def test_passes_and_info_skips(self):
        backend = _build_plate(StubBackend())
        report = ContractCheck(self._contract()).check(backend, opdag=None)

        # No ERROR: the stub answers the feature-count check (>=1) and skips the
        # rest rather than failing.
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        self.assertEqual(_errors(report), [])

        # The geometry checks it cannot answer are recorded as INFO skips.
        info_codes = {d.code for d in _infos(report)}
        self.assertIn("dim-skipped", info_codes)
        self.assertIn("measure-skipped", info_codes)
        self.assertIn("topology-skipped", info_codes)

    def test_feature_count_error_on_stub(self):
        # Stub *can* answer feature_count, so a wrong count is a real ERROR even
        # without a geometry kernel.
        backend = _build_plate(StubBackend())
        contract = Contract(name="plate", min_features=5)
        report = ContractCheck(contract).check(backend, opdag=None)
        self.assertFalse(report.ok)
        self.assertIn("too-few-features", _codes(report))


class TestContractOnMeasuredBackend(unittest.TestCase):
    """With a backend that answers 'measure'/'validity', tolerance and topology
    checks run for real."""

    def test_matching_contract_passes(self):
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0)
        contract = Contract(
            name="plate",
            bbox={"x": Tolerance(20.0, 0.1),
                  "y": Tolerance(10.0, 0.1),
                  "z": Tolerance(5.0, 0.1)},
            volume=Tolerance(1000.0, 1.0),
            min_features=1,
            require_manifold=True,
            no_self_intersections=True,
        )
        report = ContractCheck(contract).check(backend, opdag=None)
        self.assertTrue(report.ok, [d.to_dict() for d in _errors(report)])
        self.assertEqual(_infos(report), [])  # nothing skipped: all answerable

    def test_bbox_out_of_tolerance_errors(self):
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0)
        contract = Contract(name="plate", bbox={"x": Tolerance(999.0, 0.1)})
        report = ContractCheck(contract).check(backend, opdag=None)
        self.assertFalse(report.ok)
        errs = _errors(report)
        self.assertTrue(any(d.code == "dim-out-of-tol" for d in errs))
        self.assertTrue(any(d.where == "x" for d in errs))

    def test_volume_and_mass_out_of_tolerance(self):
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0)
        contract = Contract(
            name="plate",
            volume=Tolerance(500.0, 1.0),           # actual 1000 -> error
            mass=MassSpec(target=1.0, tol=0.1, density=0.00785),  # ~7.85 -> error
        )
        report = ContractCheck(contract).check(backend, opdag=None)
        self.assertFalse(report.ok)
        self.assertIn("volume-out-of-tol", _codes(report))
        self.assertIn("mass-out-of-tol", _codes(report))

    def test_non_manifold_errors(self):
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0,
                                   manifold=False, watertight=False,
                                   is_valid=False)
        contract = Contract(name="plate", require_manifold=True,
                            no_self_intersections=True)
        report = ContractCheck(contract).check(backend, opdag=None)
        self.assertFalse(report.ok)
        codes = _codes(report)
        self.assertIn("not-manifold", codes)
        self.assertIn("not-watertight", codes)
        self.assertIn("self-intersections", codes)


class TestPredicates(unittest.TestCase):
    def test_named_predicate_pass_and_fail(self):
        register_predicate("has_solid",
                           lambda backend, opdag:
                           backend.query("summary")["solid_present"])
        backend = _build_plate(StubBackend())

        ok_report = ContractCheck(
            Contract(name="p", predicates=["has_solid"])).check(backend, None)
        self.assertTrue(ok_report.ok)

        register_predicate("always_false", lambda backend, opdag: False)
        bad_report = ContractCheck(
            Contract(name="p", predicates=["always_false"])).check(backend, None)
        self.assertFalse(bad_report.ok)
        self.assertIn("predicate-failed", _codes(bad_report))

    def test_unregistered_predicate_info_skips(self):
        backend = _build_plate(StubBackend())
        report = ContractCheck(
            Contract(name="p", predicates=["never_registered"])).check(
            backend, None)
        self.assertTrue(report.ok)
        self.assertIn("predicate-skipped", {d.code for d in _infos(report)})


class TestHoleCountDegrades(unittest.TestCase):
    def test_hole_count_info_skips(self):
        backend = _MeasuredBackend(bbox=(20.0, 10.0, 5.0), volume=1000.0)
        report = ContractCheck(
            Contract(name="p", hole_count=4)).check(backend, None)
        self.assertTrue(report.ok)  # no backend hole query -> INFO, not ERROR
        self.assertIn("hole-count-skipped", {d.code for d in _infos(report)})


class TestRoundTrip(unittest.TestCase):
    def test_to_dict_from_dict(self):
        contract = Contract(
            name="bracket",
            description="an L bracket",
            bbox={"x": Tolerance(20.0, 0.1),
                  "y": Tolerance(10.0, 0.05),
                  "z": Tolerance(5.0, 0.1)},
            volume=Tolerance(1000.0, 2.0),
            mass=MassSpec(target=7.85, tol=0.1, density=0.00785),
            min_features=2,
            feature_count=3,
            hole_count=4,
            require_manifold=True,
            no_self_intersections=True,
            predicates=["has_solid"],
        )
        d = contract.to_dict()
        # JSON-serialisable
        import json
        restored = Contract.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(restored.to_dict(), d)
        self.assertEqual(restored.name, "bracket")
        self.assertEqual(restored.bbox["y"].tol, 0.05)
        self.assertEqual(restored.mass.density, 0.00785)
        self.assertEqual(restored.predicates, ["has_solid"])

    def test_empty_contract_round_trip(self):
        contract = Contract(name="empty")
        restored = Contract.from_dict(contract.to_dict())
        self.assertEqual(restored.to_dict(), {"name": "empty",
                                              "description": ""})


class TestSchema(unittest.TestCase):
    def test_schema_shape(self):
        schema = contract_from_brief_schema()
        self.assertEqual(schema["title"], "Contract")
        self.assertEqual(schema["type"], "object")
        props = schema["properties"]
        for key in ("name", "bbox", "volume", "mass", "min_features",
                    "feature_count", "hole_count", "require_manifold",
                    "no_self_intersections", "predicates"):
            self.assertIn(key, props)
        self.assertIn("name", schema["required"])
        # schema itself must be JSON-serialisable
        import json
        json.dumps(schema)


if __name__ == "__main__":
    unittest.main()
