"""Tests for the analytic simulation critic (verifiers.simulation).

Coverage:
  * pure closed forms match textbook values (cantilever sigma & deflection, hoop
    stress, Kt hole lookup, Euler load);
  * SimulationCheck flags ERROR when overstressed / buckling and passes when safe;
  * INFO-skip on the dependency-free stub (no measurable geometry, no load case);
  * needs-fea INFO where a case is not analytically reducible;
  * never crashes without an FEA solver; deterministic.
"""

import unittest

from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude
from harnesscad.io.backends.stub import StubBackend
from harnesscad.eval.verifiers.verify import Severity
from harnesscad.eval.verifiers.simulation import (
    LoadCase, SimRules, SimulationCheck, FEASolver, with_simulation,
    rectangular_section, circular_section,
    beam_bending_stress, beam_max_deflection, beam_max_moment,
    hoop_stress, longitudinal_stress,
    kt_hole, kt_fillet,
    euler_critical_load, slenderness_ratio, radius_of_gyration,
    buckling_transition_slenderness, johnson_critical_stress,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _codes(report):
    return {d.code for d in report.diagnostics}


def _by_severity(report, sev):
    return [d for d in report.diagnostics if d.severity is sev]


class _MeasuredBackend:
    """Minimal backend answering 'metrics'/'measure' like a real kernel."""

    def __init__(self, bbox, volume=1.0, solid_present=True):
        self._bbox = list(bbox)
        self._volume = volume
        self._solid_present = solid_present

    def query(self, q: str) -> dict:
        if q == "summary":
            return {"sketch_count": 1, "entity_count": 1, "feature_count": 1,
                    "solid_present": self._solid_present}
        if q in ("measure", "metrics"):
            return {"volume": self._volume, "bbox": list(self._bbox)}
        return {}


def _build_plate(backend):
    for op in (NewSketch(plane="XY"),
               AddRectangle(sketch="sk1", w=20.0, h=10.0),
               Extrude(sketch="sk1", distance=5.0)):
        res = backend.apply(op)
        assert res.ok, res.diagnostics
    return backend


# --------------------------------------------------------------------------- #
# Pure formulas vs textbook values
# --------------------------------------------------------------------------- #
class TestClosedForms(unittest.TestCase):
    def test_rectangular_section(self):
        I, c, Z, A = rectangular_section(10.0, 20.0)
        self.assertAlmostEqual(I, 10.0 * 20.0 ** 3 / 12.0)   # 6666.667
        self.assertAlmostEqual(c, 10.0)
        self.assertAlmostEqual(Z, 10.0 * 20.0 ** 2 / 6.0)    # 666.667
        self.assertAlmostEqual(A, 200.0)

    def test_cantilever_stress_and_deflection(self):
        # Steel cantilever L=100, b=10, h=20, end load P=1000 N.
        I, c, _Z, _A = rectangular_section(10.0, 20.0)
        sigma = beam_bending_stress(1000.0, 100.0, I, c, "cantilever")
        # M = P L = 1e5 N*mm; sigma = M c / I = 1e5*10/6666.67 = 150 MPa.
        self.assertAlmostEqual(beam_max_moment(1000.0, 100.0, "cantilever"), 1e5)
        self.assertAlmostEqual(sigma, 150.0, places=4)
        # delta = P L^3 / (3 E I) = 1e9 / (3*2e5*6666.67) = 0.25 mm.
        delta = beam_max_deflection(1000.0, 100.0, 200000.0, I, "cantilever")
        self.assertAlmostEqual(delta, 0.25, places=6)

    def test_simply_supported_central(self):
        I, c, _Z, _A = rectangular_section(10.0, 20.0)
        # M = P L / 4 = 25000; sigma = 25000*10/6666.67 = 37.5 MPa.
        sigma = beam_bending_stress(1000.0, 100.0, I, c, "simply_supported")
        self.assertAlmostEqual(sigma, 37.5, places=4)

    def test_hoop_stress_pr_over_t(self):
        # p=2 MPa, r=100, t=5 -> hoop = 40 MPa, long = 20 MPa.
        self.assertAlmostEqual(hoop_stress(2.0, 100.0, 5.0), 40.0)
        self.assertAlmostEqual(longitudinal_stress(2.0, 100.0, 5.0), 20.0)

    def test_kt_hole_lookup(self):
        # d/w = 0.5 -> Peterson polynomial ~ 2.159.
        self.assertAlmostEqual(kt_hole(0.5), 2.15875, places=4)
        # small hole -> classical 3.0.
        self.assertAlmostEqual(kt_hole(0.0), 3.0, places=6)
        self.assertGreater(kt_hole(0.01), 2.9)

    def test_kt_fillet_inglis(self):
        # depth = (D-d)/2 = (30-20)/2 = 5, r = 5 -> Kt = 1 + 2*sqrt(1) = 3.
        self.assertAlmostEqual(kt_fillet(30.0, 20.0, 5.0), 3.0, places=6)

    def test_euler_critical_load(self):
        # Steel pinned-pinned column L=1000, circular d=10.
        I, _c, _Z, A = circular_section(10.0)
        self.assertAlmostEqual(radius_of_gyration(I, A), 2.5, places=6)
        self.assertAlmostEqual(slenderness_ratio(1000.0, I, A, "pinned-pinned"),
                               400.0, places=4)
        p_cr = euler_critical_load(200000.0, I, 1000.0, "pinned-pinned")
        # pi^2 * 2e5 * 490.87 / 1e6 ~ 969 N.
        self.assertAlmostEqual(p_cr, 968.9, delta=1.0)

    def test_johnson_reduces_to_yield_for_stub(self):
        self.assertAlmostEqual(johnson_critical_stress(250.0, 0.0, 200000.0), 250.0)

    def test_buckling_transition(self):
        # sqrt(2 pi^2 E / Sy) for E=2e5, Sy=250 ~ 125.7.
        self.assertAlmostEqual(
            buckling_transition_slenderness(200000.0, 250.0), 125.66, delta=0.1)


# --------------------------------------------------------------------------- #
# SimulationCheck — beam bending
# --------------------------------------------------------------------------- #
class TestBeamBendingCheck(unittest.TestCase):
    def _beam_case(self, force, yield_strength, sf=2.0):
        return LoadCase(force=force, yield_strength=yield_strength, safety_factor=sf,
                        support="cantilever", where="free end", analysis="beam_bending",
                        geometry={"span": 100.0, "section_b": 10.0, "section_h": 20.0})

    def test_overstressed_is_error(self):
        # P=5000 -> sigma=750 MPa >> Sy=250 -> FoS 0.33 -> ERROR.
        report = SimulationCheck(self._beam_case(5000.0, 250.0)).check(None, None)
        errs = _by_severity(report, Severity.ERROR)
        self.assertTrue(any(d.code == "overstressed" for d in errs))
        self.assertFalse(report.ok)

    def test_safe_beam_passes(self):
        # P=1000 -> sigma=150 MPa; Sy=400 -> FoS 2.67 >= 2 -> pass, no error.
        report = SimulationCheck(self._beam_case(1000.0, 400.0)).check(None, None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertIn("sim-pass", _codes(report))
        self.assertTrue(report.ok)

    def test_marginal_warns(self):
        # sigma=150; Sy=315 -> FoS 2.1, within [2.0, 2.2) -> WARNING marginal.
        report = SimulationCheck(self._beam_case(1000.0, 315.0)).check(None, None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertIn("marginal", _codes(report))

    def test_deflection_limit_error(self):
        # delta = 0.25 mm; limit 0.1 -> over-deflected ERROR.
        lc = LoadCase(force=1000.0, yield_strength=400.0, safety_factor=2.0,
                      support="cantilever", deflection_limit=0.1,
                      analysis="beam_bending",
                      geometry={"span": 100.0, "section_b": 10.0, "section_h": 20.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("over-deflected", _codes(report))
        self.assertFalse(report.ok)

    def test_geometry_derived_from_bbox(self):
        # No explicit section: bbox 100 x 20 x 10 -> span 100, weak section 20x10.
        lc = LoadCase(force=1000.0, yield_strength=400.0, support="cantilever",
                      analysis="beam_bending")
        backend = _MeasuredBackend(bbox=(100.0, 20.0, 10.0), volume=20000.0)
        report = SimulationCheck(lc).check(backend, None)
        # Must produce a real stress finding, not a skip.
        self.assertTrue(any(c in _codes(report)
                            for c in ("sim-pass", "overstressed", "marginal")))
        self.assertEqual([d for d in report.diagnostics
                          if d.code == "simulation-skipped"], [])


# --------------------------------------------------------------------------- #
# SimulationCheck — pressure vessel
# --------------------------------------------------------------------------- #
class TestPressureVesselCheck(unittest.TestCase):
    def test_hoop_stress_pass(self):
        # p=2, r=100, t=5 -> hoop 40 MPa; Sy=250, SF=2 -> FoS 6.25 -> pass.
        lc = LoadCase(pressure=2.0, yield_strength=250.0, safety_factor=2.0,
                      analysis="pressure_vessel",
                      geometry={"radius": 100.0, "wall_thickness": 5.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("sim-pass", _codes(report))
        self.assertTrue(report.ok)

    def test_hoop_stress_overstressed(self):
        # p=20 -> hoop 400 MPa > Sy 250 -> ERROR.
        lc = LoadCase(pressure=20.0, yield_strength=250.0, safety_factor=2.0,
                      analysis="pressure_vessel",
                      geometry={"radius": 100.0, "wall_thickness": 5.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("overstressed", _codes(report))
        self.assertFalse(report.ok)

    def test_thick_wall_needs_fea(self):
        # r/t = 2 < 10 -> thin-wall invalid -> needs-fea, no fabricated pass.
        lc = LoadCase(pressure=2.0, yield_strength=250.0,
                      analysis="pressure_vessel",
                      geometry={"radius": 10.0, "wall_thickness": 5.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("needs-fea", _codes(report))
        self.assertNotIn("sim-pass", _codes(report))

    def test_missing_geometry_needs_fea(self):
        lc = LoadCase(pressure=2.0, analysis="pressure_vessel")
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("needs-fea", _codes(report))


# --------------------------------------------------------------------------- #
# SimulationCheck — stress concentration
# --------------------------------------------------------------------------- #
class TestStressConcentrationCheck(unittest.TestCase):
    def test_hole_peak_stress(self):
        # F=10000 N, plate w=40, d=20, t=5 -> net area (40-20)*5=100 mm^2;
        # nominal = 100 MPa; Kt(0.5)=2.159 -> peak 215.9 MPa; Sy=250 -> FoS 1.16.
        lc = LoadCase(force=10000.0, yield_strength=250.0, safety_factor=2.0,
                      analysis="stress_concentration", where="at hole",
                      geometry={"hole_diameter": 20.0, "plate_width": 40.0,
                                "thickness": 5.0})
        report = SimulationCheck(lc).check(None, None)
        # FoS 1.16 < 2 -> overstressed.
        self.assertIn("overstressed", _codes(report))

    def test_fillet_uses_inglis(self):
        lc = LoadCase(force=1000.0, yield_strength=400.0, safety_factor=2.0,
                      analysis="stress_concentration",
                      geometry={"width_large": 30.0, "width_small": 20.0,
                                "fillet_radius": 5.0, "thickness": 5.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertTrue(any(c in _codes(report)
                            for c in ("sim-pass", "overstressed", "marginal")))

    def test_missing_geometry_needs_fea(self):
        lc = LoadCase(force=1000.0, analysis="stress_concentration")
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("needs-fea", _codes(report))


# --------------------------------------------------------------------------- #
# SimulationCheck — buckling
# --------------------------------------------------------------------------- #
class TestBucklingCheck(unittest.TestCase):
    def test_slender_column_buckles(self):
        # d=10 circular, L=1000, pinned-pinned -> P_cr ~ 969 N; axial 5000 N
        # -> FoS 0.19 -> buckling ERROR.
        lc = LoadCase(force=5000.0, yield_strength=250.0, safety_factor=2.0,
                      end_condition="pinned-pinned", analysis="buckling",
                      where="column", geometry={"length": 1000.0, "diameter": 10.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("buckling", _codes(report))
        self.assertFalse(report.ok)

    def test_stocky_column_passes(self):
        # Same section, tiny load -> passes.
        lc = LoadCase(force=100.0, yield_strength=250.0, safety_factor=2.0,
                      end_condition="pinned-pinned", analysis="buckling",
                      geometry={"length": 1000.0, "diameter": 10.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertIn("sim-pass", _codes(report))


# --------------------------------------------------------------------------- #
# Graceful degradation / never-crash
# --------------------------------------------------------------------------- #
class TestDegradation(unittest.TestCase):
    def test_no_load_case_info_skips(self):
        report = SimulationCheck(None).check(_build_plate(StubBackend()), None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertIn("simulation-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_stub_no_geometry_info_skips(self):
        # Stub answers no 'measure'/'metrics'; load case carries no geometry.
        lc = LoadCase(force=1000.0, analysis="beam_bending")
        report = SimulationCheck(lc).check(_build_plate(StubBackend()), None)
        self.assertEqual(_by_severity(report, Severity.ERROR), [])
        self.assertIn("simulation-skipped", _codes(report))
        self.assertTrue(report.ok)

    def test_never_crashes_without_solver(self):
        # A grab-bag of odd cases must never raise and never ERROR-out spuriously.
        cases = [
            LoadCase(),                                            # nothing
            LoadCase(force=1000.0, support="weird", analysis="beam_bending",
                     geometry={"span": 100.0, "section_b": 10.0, "section_h": 20.0}),
            LoadCase(force=1000.0, end_condition="weird", analysis="buckling",
                     geometry={"length": 100.0, "diameter": 10.0}),
            LoadCase(torque=5000.0),                               # torque -> no auto check
        ]
        for lc in cases:
            report = SimulationCheck(lc).check(None, None)
            self.assertIsNotNone(report)
            # These specific cases fabricate nothing: either skip or needs-fea.
            self.assertNotIn("overstressed", _codes(report))

    def test_bad_support_is_needs_fea(self):
        lc = LoadCase(force=1000.0, support="fixed-both-ends", analysis="beam_bending",
                      geometry={"span": 100.0, "section_b": 10.0, "section_h": 20.0})
        report = SimulationCheck(lc).check(None, None)
        self.assertIn("needs-fea", _codes(report))


# --------------------------------------------------------------------------- #
# Round-trip + wiring + solver seam
# --------------------------------------------------------------------------- #
class TestLoadCaseRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        lc = LoadCase(force=1000.0, pressure=2.0, yield_strength=300.0,
                      youngs_modulus=210000.0, safety_factor=1.5, where="tip",
                      support="simply_supported", end_condition="fixed-free",
                      analysis="beam_bending",
                      deflection_limit=0.5, geometry={"span": 50.0})
        restored = LoadCase.from_dict(lc.to_dict())
        self.assertEqual(restored.force, 1000.0)
        self.assertEqual(restored.pressure, 2.0)
        self.assertEqual(restored.yield_strength, 300.0)
        self.assertEqual(restored.safety_factor, 1.5)
        self.assertEqual(restored.support, "simply_supported")
        self.assertEqual(restored.analysis, "beam_bending")
        self.assertEqual(restored.deflection_limit, 0.5)
        self.assertEqual(restored.geometry, {"span": 50.0})

    def test_from_dict_defaults(self):
        lc = LoadCase.from_dict({})
        self.assertIsNone(lc.force)
        self.assertEqual(lc.yield_strength, 250.0)
        self.assertEqual(lc.safety_factor, 2.0)

    def test_analyses_auto(self):
        self.assertEqual(LoadCase(force=1.0).analyses(), ["beam_bending"])
        self.assertEqual(LoadCase(pressure=1.0).analyses(), ["pressure_vessel"])
        self.assertEqual(set(LoadCase(force=1.0, pressure=1.0).analyses()),
                         {"beam_bending", "pressure_vessel"})
        self.assertEqual(LoadCase().analyses(), [])
        self.assertEqual(LoadCase(force=1.0, analysis="buckling").analyses(),
                         ["buckling"])


class TestSimRulesRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        rules = SimRules(marginal_band=1.25, thin_wall_ratio=8.0)
        self.assertEqual(SimRules.from_dict(rules.to_dict()), rules)

    def test_from_dict_none(self):
        self.assertEqual(SimRules.from_dict(None), SimRules())


class TestWiringAndSeam(unittest.TestCase):
    def test_with_simulation_appends(self):
        base = ["a", "b"]
        result = with_simulation(base, LoadCase(force=1.0))
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[-1], SimulationCheck)
        self.assertEqual(result[-1].name, "simulation")
        self.assertEqual(base, ["a", "b"])

    def test_fea_solver_protocol_is_runtime_checkable(self):
        # A conforming object satisfies the protocol; ships NO real solver.
        class _DummySolver:
            name = "dummy"

            def mesh(self, shape):
                return shape

            def solve(self, mesh, load_case):
                return {}

        self.assertIsInstance(_DummySolver(), FEASolver)
        self.assertNotIsInstance(object(), FEASolver)

    def test_determinism(self):
        lc = LoadCase(force=1234.0, yield_strength=250.0, analysis="beam_bending",
                      geometry={"span": 100.0, "section_b": 10.0, "section_h": 20.0})
        r1 = SimulationCheck(lc).check(None, None)
        r2 = SimulationCheck(lc).check(None, None)
        self.assertEqual([d.to_dict() for d in r1.diagnostics],
                         [d.to_dict() for d in r2.diagnostics])


if __name__ == "__main__":
    unittest.main()
