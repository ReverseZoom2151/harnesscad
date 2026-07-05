"""Tests for the real-geometry CadQueryBackend (OCCT).

These build actual B-rep solids, so they require cadquery / cadquery-ocp. When
cadquery is not importable the whole suite is skipped (the backend module itself
still imports fine — cadquery is imported lazily inside its methods).
"""

import unittest

from backends.cadquery_backend import CadQueryBackend
from checks_geometry import BRepValidityCheck
from cisp.ops import (
    NewSketch, AddRectangle, AddCircle, Extrude, Fillet, Boolean,
)
from verify import Severity


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()


def _build_plate(w=20.0, h=10.0, t=5.0) -> CadQueryBackend:
    """Sketch a rectangle and extrude it into a real plate solid."""
    b = CadQueryBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok
    assert b.apply(Extrude(sketch="sk1", distance=t)).ok
    return b


# The backend module must import even without cadquery installed.
class TestModuleImportsWithoutCadquery(unittest.TestCase):
    def test_backend_constructs_without_kernel(self):
        b = CadQueryBackend()
        self.assertEqual(b.query("summary")["feature_count"], 0)
        self.assertFalse(b.query("validity")["solid_present"])


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestRealPlate(unittest.TestCase):
    def test_plate_is_valid_solid(self):
        b = _build_plate()
        summary = b.query("summary")
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)

        v = b.query("validity")
        self.assertTrue(v["solid_present"])
        self.assertTrue(v["manifold"])
        self.assertTrue(v["watertight"])
        self.assertTrue(v["is_valid"])

    def test_measure_matches_nominal(self):
        b = _build_plate(w=20.0, h=10.0, t=5.0)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 20.0 * 10.0 * 5.0, places=3)
        self.assertEqual([round(d, 3) for d in m["bbox"]], [20.0, 10.0, 5.0])

    def test_export_step_is_real(self):
        b = _build_plate()
        step = b.export("step")
        self.assertTrue(step)
        self.assertIn("ISO-10303", step)

    def test_deterministic_digest(self):
        d1 = _build_plate().state_digest()
        d2 = _build_plate().state_digest()
        self.assertEqual(d1, d2)

    def test_different_geometry_differs_in_digest(self):
        self.assertNotEqual(
            _build_plate(w=20.0).state_digest(),
            _build_plate(w=30.0).state_digest(),
        )


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestBlockAndCorrect(unittest.TestCase):
    def test_bad_reference_does_not_mutate(self):
        b = CadQueryBackend()
        before = b.state_digest()
        res = b.apply(Extrude(sketch="nope", distance=5.0))
        self.assertFalse(res.ok)
        self.assertTrue(res.diagnostics)
        self.assertEqual(res.diagnostics[0].severity, Severity.ERROR)
        self.assertEqual(b.state_digest(), before)

    def test_oversized_fillet_blocks_without_mutating(self):
        b = _build_plate()
        before = b.state_digest()
        res = b.apply(Fillet(edges=(), radius=999.0))  # bigger than the plate
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)  # kernel failure rolled back

    def test_bad_circle_radius_blocks(self):
        b = CadQueryBackend()
        b.apply(NewSketch())
        res = b.apply(AddCircle(sketch="sk1", r=-1.0))
        self.assertFalse(res.ok)

    def test_boolean_requires_two_solids(self):
        b = _build_plate()
        res = b.apply(Boolean(kind="cut"))
        self.assertFalse(res.ok)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestFeatures(unittest.TestCase):
    def test_real_fillet_keeps_solid_valid(self):
        b = _build_plate()
        res = b.apply(Fillet(edges=(), radius=1.0))
        self.assertTrue(res.ok)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_boolean_union_produces_single_valid_solid(self):
        b = CadQueryBackend()
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk2", x=5.0, y=2.0, w=5.0, h=5.0))
        b.apply(Extrude(sketch="sk2", distance=8.0))
        res = b.apply(Boolean(kind="union"))
        self.assertTrue(res.ok)
        v = b.query("validity")
        self.assertTrue(v["is_valid"])
        self.assertEqual(b.state_digest(), b.state_digest())  # stable


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestDofTracking(unittest.TestCase):
    def test_constraints_reduce_nominal_dof_like_stub(self):
        from cisp.ops import Constrain
        b = CadQueryBackend()
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        # rectangle contributes 4 DOF
        self.assertEqual(b.query("sketch_dof")["sk1"], 4)
        b.apply(Constrain(kind="distance", a="e1", value=10.0))
        self.assertEqual(b.query("sketch_dof")["sk1"], 3)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestBRepValidityCheck(unittest.TestCase):
    def test_valid_plate_passes(self):
        b = _build_plate()
        report = BRepValidityCheck().check(b, None)
        self.assertTrue(report.ok)

    def test_no_solid_is_noop(self):
        b = CadQueryBackend()
        report = BRepValidityCheck().check(b, None)
        self.assertTrue(report.ok)
        self.assertEqual(report.diagnostics, [])


if __name__ == "__main__":
    unittest.main()
