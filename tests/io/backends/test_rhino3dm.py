"""Tests for the Rhino3dmBackend -- an openNURBS geometry + IO voice.

The backend implements only what rhino3dm can genuinely do (extrude a single
rectangle or circle into the first solid, primitives, mesh measurement) and
REFUSES everything else with a typed ``unsupported-op`` -- never a wrong number,
never a silent drop. These tests assert exactly that split, plus the two things
that make it useful: its measurements agree with the analytic truth, and it flows
through the format registry and the differential oracle.

Requires the optional ``rhino3dm`` wheel; skipped cleanly when it is absent.
"""

import math
import os
import tempfile
import unittest

from harnesscad.io.formats import threedm
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle, AddCircle, AddLine, Extrude, Fillet, Chamfer,
    Boolean, Revolve, Hole, Shell, Draft, Loft, Sweep, LinearPattern,
    CircularPattern, Mirror, AddInstance, Mate, SetParam,
)
from harnesscad.io import gate


HAVE_R3 = threedm.RHINO3DM_AVAILABLE

if HAVE_R3:
    from harnesscad.io.backends.rhino3dm import Rhino3dmBackend


def _box_backend(w=10.0, h=20.0, d=5.0, plane="XY"):
    b = Rhino3dmBackend()
    assert b.apply(NewSketch(plane)).ok
    assert b.apply(AddRectangle("sk1", 0, 0, w, h)).ok
    assert b.apply(Extrude("sk1", d)).ok
    return b


@unittest.skipUnless(HAVE_R3, "rhino3dm is not installed")
class Rhino3dmBackendBuildTest(unittest.TestCase):
    def test_box_extrude_volume_and_bbox(self) -> None:
        b = _box_backend(10, 20, 5)
        m = b.query("metrics")
        self.assertAlmostEqual(m["volume"], 10 * 20 * 5, places=6)
        self.assertEqual([round(c, 6) for c in m["bbox"]], [10.0, 20.0, 5.0])
        v = b.query("validity")
        self.assertTrue(v["solid_present"])
        self.assertTrue(v["watertight"])
        self.assertEqual(v["genus"], 0)

    def test_box_volume_matches_analytic_and_passes_gate(self) -> None:
        # NB: HarnessSession's constructor resets its backend, so we build the
        # backend directly and gate the backend itself (it carries its own op log
        # for the declared-intent replay).
        b = _box_backend(12, 8, 4)
        report = gate.check(b)
        self.assertTrue(report.ok, [f.check for f in report.failures])
        self.assertAlmostEqual(report.measurement["volume"], 12 * 8 * 4, places=6)
        self.assertTrue(report.measurement["watertight"])
        self.assertTrue(report.measurement["manifold"])
        self.assertGreater(report.measurement["signed_volume"], 0.0)  # outward
        self.assertEqual(report.measurement["declared_intent"], "checked")

    def test_cylinder_extrude_volume_matches_analytic(self) -> None:
        b = Rhino3dmBackend()
        b.apply(NewSketch("XY"))
        b.apply(AddCircle("sk1", 0, 0, 3))
        b.apply(Extrude("sk1", 7))
        m = b.query("metrics")
        self.assertAlmostEqual(m["volume"], math.pi * 9 * 7, places=6)
        # bbox: 2r x 2r x h.
        self.assertEqual([round(c, 4) for c in m["bbox"]], [6.0, 6.0, 7.0])
        verts, faces = b.mesh()
        gm = gate.measure(verts, faces)
        self.assertTrue(gm["watertight"])
        self.assertGreater(gm["signed_volume"], 0.0)

    def test_plane_orients_the_extrusion(self) -> None:
        # XZ sketch => extrude along Y: bbox is (w, d, h).
        b = _box_backend(10, 20, 5, plane="XZ")
        self.assertEqual([round(c, 6) for c in b.query("metrics")["bbox"]],
                         [10.0, 5.0, 20.0])
        # YZ sketch => extrude along X: bbox is (d, w, h).
        b2 = _box_backend(10, 20, 5, plane="YZ")
        self.assertEqual([round(c, 6) for c in b2.query("metrics")["bbox"]],
                         [5.0, 10.0, 20.0])


@unittest.skipUnless(HAVE_R3, "rhino3dm is not installed")
class Rhino3dmBackendRefusalTest(unittest.TestCase):
    """Every op rhino3dm cannot do must be refused with a typed error."""

    def _built(self):
        return _box_backend(10, 10, 5)

    def test_kernel_ops_are_refused_with_typed_error(self) -> None:
        refused = [
            Fillet((), 1.0), Chamfer((), 1.0), Shell((), 2.0),
            Draft((), 5.0, "XY"), Hole("sk1"), Boolean("cut", "a", "b"),
            Revolve("sk1"), Loft(("sk1", "sk2")), Sweep("sk1", "sk2"),
            LinearPattern("f1"), CircularPattern("f1"), Mirror("f1"),
            AddInstance("f1"), Mate("rigid", "a", "b"),
        ]
        for op in refused:
            b = self._built()
            r = b.apply(op)
            self.assertFalse(r.ok, f"{type(op).__name__} should be refused")
            codes = [d.code for d in r.diagnostics]
            self.assertIn("unsupported-op", codes,
                          f"{type(op).__name__} refused with {codes}")

    def test_refused_op_taints_the_measurement(self) -> None:
        # Before the refusal the box measures honestly.
        b = self._built()
        self.assertAlmostEqual(b.query("metrics")["volume"], 10 * 10 * 5, places=6)
        # A refused unsupported op means the requested part was NEVER built, so the
        # measurement must REFUSE (volume/bbox None) -- it must NOT keep reporting
        # the pre-op solid as if it were the finished part (the silent-wrong-part
        # failure the oracle exists to catch).
        r = b.apply(Shell((), 2.0))              # refused: rhino3dm cannot shell
        self.assertFalse(r.ok)
        for q in ("metrics", "measure"):
            self.assertIsNone(b.query(q)["volume"], q)
            self.assertIsNone(b.query(q)["bbox"], q)

    def test_hole_refuses_measurement_not_the_unbored_volume(self) -> None:
        # The WORST case: a plate with a hole must NOT come back as the un-bored
        # plate volume (24000 for a 60x40x10 plate). rhino3dm has no boolean, so
        # the hole is refused and the whole measurement is refused with it.
        b = _box_backend(60, 40, 10)
        self.assertAlmostEqual(b.query("metrics")["volume"], 24000.0, places=6)
        r = b.apply(Hole(">Z", 30.0, 20.0, 10.0, None, True, "simple"))
        self.assertFalse(r.ok)
        self.assertIn("unsupported-op", [d.code for d in r.diagnostics])
        self.assertIsNone(b.query("metrics")["volume"])   # NOT 24000
        self.assertIsNone(b.query("measure")["volume"])

    def test_revolve_refuses_measurement_not_zero(self) -> None:
        # A revolve used to leak 0.0 (no solid). It must refuse, not report zero.
        b = _box_backend(10, 10, 5)
        r = b.apply(Revolve("sk1"))
        self.assertFalse(r.ok)
        self.assertIsNone(b.query("measure")["volume"])

    def test_supported_op_still_measures_after_a_taint_free_stream(self) -> None:
        # A stream with NO refused op measures normally -- the taint is not a
        # blanket off-switch, only a refused unsupported op trips it.
        b = _box_backend(12, 8, 4)
        self.assertAlmostEqual(b.query("measure")["volume"], 12 * 8 * 4, places=6)
        self.assertEqual([round(c, 6) for c in b.query("measure")["bbox"]],
                         [12.0, 8.0, 4.0])

    def test_second_extrude_refused_no_boolean(self) -> None:
        b = self._built()
        r = b.apply(Extrude("sk1", 3))
        self.assertFalse(r.ok)
        self.assertIn("unsupported-op", [d.code for d in r.diagnostics])

    def test_multi_profile_or_line_profile_refused(self) -> None:
        b = Rhino3dmBackend()
        b.apply(NewSketch("XY"))
        b.apply(AddLine("sk1", 0, 0, 5, 0))     # not a closed profile we extrude
        r = b.apply(Extrude("sk1", 5))
        self.assertFalse(r.ok)
        self.assertIn("unsupported-op", [d.code for d in r.diagnostics])

    def test_bad_values_still_typed(self) -> None:
        b = Rhino3dmBackend()
        b.apply(NewSketch("XY"))
        self.assertFalse(b.apply(AddRectangle("sk1", 0, 0, -1, 5)).ok)
        b.apply(AddRectangle("sk1", 0, 0, 10, 10))
        self.assertFalse(b.apply(Extrude("sk1", 0)).ok)   # zero distance


@unittest.skipUnless(HAVE_R3, "rhino3dm is not installed")
class Rhino3dmBackendIoAndDeterminismTest(unittest.TestCase):
    def test_export_3dm_stl_obj(self) -> None:
        b = _box_backend()
        data = b.export("3dm")
        self.assertIsInstance(data, bytes)
        self.assertTrue(len(data) > 0)
        self.assertIn("solid", b.export("stl"))
        self.assertIn("v ", b.export("obj"))
        with self.assertRaises(ValueError):
            b.export("iges")

    def test_export_3dm_round_trips_through_codec(self) -> None:
        b = _box_backend(10, 20, 5)
        data = b.export("3dm")
        tmp = os.path.join(tempfile.mkdtemp(), "backend.3dm")
        with open(tmp, "wb") as fh:
            fh.write(data)
        verts, tris, unit = threedm.read_3dm(tmp)
        self.assertEqual(unit, "millimeter")
        m = gate.measure(verts, tris)
        self.assertAlmostEqual(m["volume"], 10 * 20 * 5, places=6)

    def test_digest_is_deterministic_and_edit_sensitive(self) -> None:
        a = _box_backend(10, 20, 5)
        b = _box_backend(10, 20, 5)
        self.assertEqual(a.state_digest(), b.state_digest())
        c = _box_backend(10, 20, 6)
        self.assertNotEqual(a.state_digest(), c.state_digest())

    def test_setparam_edit_replays(self) -> None:
        b = _box_backend(10, 20, 5)
        self.assertAlmostEqual(b.query("metrics")["volume"], 1000.0, places=6)
        r = b.apply(SetParam(2, "distance", 8))     # op index 2 is the Extrude
        self.assertTrue(r.ok)
        self.assertAlmostEqual(b.query("metrics")["volume"], 1600.0, places=6)

    def test_unavailable_backend_raises_backend_unavailable(self) -> None:
        # Simulate absence by monkeypatching the availability flag.
        import harnesscad.io.formats.threedm as td
        saved = td.RHINO3DM_AVAILABLE
        td.RHINO3DM_AVAILABLE = False
        try:
            with self.assertRaises(BackendUnavailable):
                Rhino3dmBackend()
        finally:
            td.RHINO3DM_AVAILABLE = saved


@unittest.skipUnless(HAVE_R3, "rhino3dm is not installed")
class Rhino3dmBackendOracleTest(unittest.TestCase):
    """It must flow into the differential oracle as an independent voice."""

    def test_registered_in_probe(self) -> None:
        from harnesscad.eval.selftest import probe
        self.assertIn("rhino3dm", probe.BACKENDS)
        self.assertIn("rhino3dm", probe.GEOMETRIC_BACKENDS)
        self.assertIn("rhino3dm", probe.TOLERANCES)

    def test_resolves_to_real_backend_not_stub(self) -> None:
        from harnesscad.eval.selftest import probe
        backend, skip = probe.resolve("rhino3dm")
        self.assertIsNotNone(backend, skip)
        self.assertEqual(type(backend).__name__, "Rhino3dmBackend")

    def test_agrees_with_frep_on_a_box(self) -> None:
        from harnesscad.eval.selftest import differential
        ops = (NewSketch("XY"), AddRectangle("sk1", 0, 0, 10, 20),
               Extrude("sk1", 5))
        case = differential.compare("box", ops, backends=["frep", "rhino3dm"])
        self.assertIn("rhino3dm", case.consensus)
        self.assertEqual(len(case.disagreements), 0)

    def test_refuses_unsupported_op_as_capability_gap(self) -> None:
        from harnesscad.eval.selftest import probe
        ops = [NewSketch("XY"), AddRectangle("sk1", 0, 0, 10, 20),
               Extrude("sk1", 5), Fillet((), 1.0)]
        obs = probe.observe("rhino3dm", ops)
        self.assertFalse(obs.ok)
        self.assertEqual(obs.rejected, "fillet")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
