"""Tests for the FreeCADBackend — real parametric B-rep through FreeCAD (OCCT).

FreeCAD is an application, not a wheel: these run only where `freecadcmd` is on
the PATH / in a standard install location / named by HARNESSCAD_FREECAD. When it
is absent the suite skips cleanly (the backend module still imports, and the
constructor raises BackendUnavailable rather than crashing the harness).

A real B-rep kernel is EXACT, so the assertions here are tight: a boolean cut
must remove precisely the analytic volume, not "close to" it.
"""

import math
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Chamfer, CircularPattern, Draft, Extrude,
    Fillet, Hole, LinearPattern, Loft, Mirror, NewSketch, Revolve, SetParam,
    Shell, Sweep,
)
from harnesscad.io.backends.base import BackendUnavailable
from harnesscad.io.backends.freecad import OP_TO_FREECAD, FreeCADBackend

HAVE_FC = FreeCADBackend.available()

# A 40 x 24 x 8 plate with two 6mm through-holes -- the harness's reference part.
PLATE_W, PLATE_H, PLATE_T = 40.0, 24.0, 8.0
HOLE_D = 6.0
PLATE_VOLUME = PLATE_W * PLATE_H * PLATE_T
ANALYTIC_2HOLE = PLATE_VOLUME - 2.0 * math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T


def _plate(w=PLATE_W, h=PLATE_H, t=PLATE_T) -> FreeCADBackend:
    b = FreeCADBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok
    assert b.apply(Extrude(sketch="sk1", distance=t)).ok
    return b


class TestAvailability(unittest.TestCase):
    """The backend must behave sanely whether or not FreeCAD is installed."""

    def test_available_never_raises(self):
        self.assertIsInstance(FreeCADBackend.available(), bool)

    def test_missing_freecad_raises_backend_unavailable(self):
        """The graceful-absence contract, exercised even WHERE FreeCAD is installed.

        REGRESSION: BackendUnavailable takes (tool, message, searched); raising it
        with a single argument made the absent path die with a TypeError instead,
        which would have taken the whole harness down on any machine without
        FreeCAD -- the exact case this backend exists to survive.
        """
        import harnesscad.io.backends.freecad as mod
        names, patterns = mod.EXECUTABLE_NAMES, mod.EXECUTABLE_PATTERNS
        mod.EXECUTABLE_NAMES = ("harnesscad-no-such-binary",)
        mod.EXECUTABLE_PATTERNS = ()
        try:
            self.assertFalse(mod.FreeCADBackend.available())
            with self.assertRaises(BackendUnavailable) as ctx:
                mod.FreeCADBackend()
            self.assertEqual(ctx.exception.tool, "freecad")
            self.assertTrue(ctx.exception.searched)
        finally:
            mod.EXECUTABLE_NAMES, mod.EXECUTABLE_PATTERNS = names, patterns

    def test_absent_freecad_falls_back_to_stub_in_the_server(self):
        """A missing kernel must degrade the CISP server, never crash it."""
        import harnesscad.io.backends.freecad as mod
        from harnesscad.io.surfaces.server import CISPServer
        names, patterns = mod.EXECUTABLE_NAMES, mod.EXECUTABLE_PATTERNS
        mod.EXECUTABLE_NAMES = ("harnesscad-no-such-binary",)
        mod.EXECUTABLE_PATTERNS = ()
        try:
            server = CISPServer(backend="freecad")
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("freecad backend unavailable", server.backend_note)
        finally:
            mod.EXECUTABLE_NAMES, mod.EXECUTABLE_PATTERNS = names, patterns

    def test_every_op_maps_to_a_real_freecad_operation(self):
        """The op->FreeCAD map must name operations the 53-op catalogue has."""
        from harnesscad.io.adapters.freecad_catalog import default_catalog
        catalog = default_catalog()
        for tag, name in OP_TO_FREECAD.items():
            if not name:
                continue
            self.assertIn(name, catalog,
                          "op '%s' maps to '%s', absent from the catalogue"
                          % (tag, name))


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestPlate(unittest.TestCase):
    def test_plate_builds_a_valid_solid(self):
        b = _plate()
        summary = b.query("summary")
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)
        v = b.query("validity")
        self.assertTrue(v["is_valid"])
        self.assertTrue(v["manifold"])
        self.assertTrue(v["watertight"])
        self.assertEqual(v["solids"], 1)
        self.assertEqual(v["faces"], 6)  # a box has exactly six faces

    def test_plate_volume_is_exact(self):
        m = _plate().query("measure")
        self.assertAlmostEqual(m["volume"], PLATE_VOLUME, places=6)
        self.assertEqual([round(d, 6) for d in m["bbox"]],
                         [PLATE_W, PLATE_H, PLATE_T])

    def test_deterministic_digest(self):
        self.assertEqual(_plate().state_digest(), _plate().state_digest())

    def test_different_geometry_differs_in_digest(self):
        self.assertNotEqual(_plate(w=40.0).state_digest(),
                            _plate(w=50.0).state_digest())


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestExactBoolean(unittest.TestCase):
    """A real B-rep kernel removes EXACTLY the analytic volume."""

    def test_two_through_holes_remove_exactly_the_analytic_volume(self):
        b = _plate()
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                                     diameter=HOLE_D, through=True)).ok)
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=30.0, y=12.0,
                                     diameter=HOLE_D, through=True)).ok)
        volume = b.query("measure")["volume"]
        # Tight: a B-rep cylinder is an analytic surface, not a facetted one.
        self.assertAlmostEqual(volume, ANALYTIC_2HOLE, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_boolean_cut_removes_exactly_the_cylinder(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
        b.apply(Extrude(sketch="sk1", distance=PLATE_T))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=10.0, cy=12.0, r=HOLE_D / 2.0))
        b.apply(Extrude(sketch="sk2", distance=PLATE_T))
        self.assertTrue(b.apply(Boolean(kind="cut")).ok)
        expected = PLATE_VOLUME - math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_boolean_union_is_exact(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk2", x=20.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk2", distance=5.0))
        self.assertTrue(b.apply(Boolean(kind="union")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 1000.0, places=6)

    def test_revolve_is_exact(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddRectangle(sketch="sk1", x=10.0, y=0.0, w=5.0, h=20.0))
        self.assertTrue(b.apply(Revolve(sketch="sk1", angle=360.0,
                                        axis=(0, 0, 0, 0, 1, 0))).ok)
        expected = math.pi * (15.0 ** 2 - 10.0 ** 2) * 20.0  # an exact tube
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_shell_is_exact(self):
        b = _plate()
        self.assertTrue(b.apply(Shell(faces=(), thickness=1.0)).ok)
        # top face removed, walls 1mm: outer minus the inner cavity
        expected = PLATE_VOLUME - (PLATE_W - 2) * (PLATE_H - 2) * (PLATE_T - 1)
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestEdgeBlends(unittest.TestCase):
    """Fillet/chamfer must actually ROUND edges, not merely be accepted."""

    def test_fillet_rounds_every_edge(self):
        base = _plate().query("metrics")
        b = _plate()
        self.assertTrue(b.apply(Fillet(edges=(), radius=1.0)).ok)
        filleted = b.query("metrics")
        # A rounded box gains a face per edge and a patch per corner...
        self.assertGreater(filleted["faces"], base["faces"])
        # ...and loses exactly the material the rounding cuts away.
        self.assertLess(filleted["volume"], base["volume"])
        self.assertTrue(b.query("validity")["is_valid"])

    def test_chamfer_cuts_every_edge(self):
        base = _plate().query("metrics")
        b = _plate()
        self.assertTrue(b.apply(Chamfer(edges=(), distance=1.0)).ok)
        chamfered = b.query("metrics")
        self.assertGreater(chamfered["faces"], base["faces"])
        self.assertLess(chamfered["volume"], base["volume"])
        # A chamfer removes a straight wedge; a fillet leaves a rounded corner,
        # so the chamfer must remove strictly MORE than the equal-size fillet.
        b2 = _plate()
        b2.apply(Fillet(edges=(), radius=1.0))
        self.assertLess(chamfered["volume"], b2.query("metrics")["volume"])


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestPatterns(unittest.TestCase):
    def test_linear_pattern_is_exact(self):
        b = _plate(10.0, 10.0, 5.0)
        self.assertTrue(b.apply(LinearPattern(count=3, spacing=20.0,
                                              direction=(1, 0, 0))).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 3 * 500.0, places=6)

    def test_circular_pattern_is_exact(self):
        b = _plate(10.0, 10.0, 5.0)
        self.assertTrue(b.apply(CircularPattern(count=4, angle=360.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 4 * 500.0, places=6)

    def test_mirror_is_exact(self):
        b = _plate(10.0, 10.0, 5.0)
        self.assertTrue(b.apply(Mirror(plane="YZ")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 2 * 500.0, places=6)


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestExport(unittest.TestCase):
    def test_step_export_is_real(self):
        step = _plate().export("step")
        self.assertIn("ISO-10303", step)
        self.assertIn("MANIFOLD_SOLID_BREP", step)

    def test_brep_export_is_the_native_kernel_format(self):
        self.assertIn("CASCADE", _plate().export("brep"))

    def test_unsupported_format_is_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            _plate().export("dwg")
        self.assertIn("cannot export", str(ctx.exception))

    def test_export_with_no_solid_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            FreeCADBackend().export("step")


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestBlockAndCorrect(unittest.TestCase):
    def test_bad_reference_does_not_mutate(self):
        b = FreeCADBackend()
        before = b.state_digest()
        res = b.apply(Extrude(sketch="nope", distance=5.0))
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)

    def test_unsupported_ops_are_typed_not_silent(self):
        for op in (Draft(faces=(), angle=5.0),
                   Loft(sketches=("sk1", "sk2")),
                   Sweep(sketch="sk1", path="sk2")):
            b = _plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok, "%s must not silently no-op" % type(op).__name__)
            self.assertEqual(res.diagnostics[0].code, "unsupported-op")
            self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestParametricExpressions(unittest.TestCase):
    """SetParam accepts FreeCAD's own expression grammar (freecad_expressions)."""

    def test_numeric_set_param_rebuilds(self):
        b = _plate()
        self.assertTrue(b.apply(SetParam(target=2, param="distance", value=16.0)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               PLATE_W * PLATE_H * 16.0, places=6)

    def test_expression_with_units_is_evaluated(self):
        b = _plate()
        self.assertTrue(b.apply(SetParam(target=2, param="distance",
                                         value="2 * 4 + 8mm")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               PLATE_W * PLATE_H * 16.0, places=6)

    def test_expression_can_reference_the_op_it_edits(self):
        b = _plate()
        self.assertTrue(b.apply(SetParam(target=2, param="distance",
                                         value="2 * op2.distance")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               PLATE_W * PLATE_H * 16.0, places=6)

    def test_malformed_expression_is_blocked(self):
        b = _plate()
        res = b.apply(SetParam(target=2, param="distance", value="2 * ("))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")

    def test_unknown_reference_is_blocked(self):
        b = _plate()
        res = b.apply(SetParam(target=2, param="distance", value="op9.bogus + 1"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestDocumentAndCatalog(unittest.TestCase):
    """The two orphaned FreeCAD modules, now load-bearing."""

    def test_document_query_is_the_real_feature_tree(self):
        b = _plate()
        b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                     diameter=HOLE_D, through=True))
        doc = b.query("document")
        self.assertEqual(doc["issues"], [])  # freecad_document.validate_context
        self.assertTrue(doc["objects"])
        self.assertIn("Part::", doc["objects"][0]["type"])
        types = [f["type_id"] for f in doc["features"]]
        self.assertEqual(types, ["PartDesign::Pad", "PartDesign::Pocket"])

    def test_catalog_query_maps_ops_onto_freecads_53_operations(self):
        view = _plate().query("catalog")
        self.assertEqual(view["operations"], 53)
        self.assertEqual(view["op_to_freecad"]["extrude"], "pad_sketch")
        self.assertEqual(view["op_to_freecad"]["revolve"], "revolve_sketch")
        self.assertIn("draft", view["unsupported"])


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestFrepParity(unittest.TestCase):
    """The harness must drive freecad and frep through the SAME surface."""

    def test_query_surface_matches_frep(self):
        from harnesscad.io.backends.frep import FRepBackend

        def build(cls):
            b = cls()
            b.apply(NewSketch(plane="XY"))
            b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
            b.apply(Extrude(sketch="sk1", distance=PLATE_T))
            return b

        fc, fr = build(FreeCADBackend), build(FRepBackend)
        self.assertEqual(fc.query("summary"), fr.query("summary"))
        self.assertEqual(fc.query("sketch_dof"), fr.query("sketch_dof"))
        for key in ("volume", "bbox"):
            self.assertIn(key, fc.query("measure"))
            self.assertIn(key, fr.query("measure"))
        for key in ("manifold", "watertight", "is_valid", "solid_present"):
            self.assertIn(key, fc.query("validity"))
            self.assertIn(key, fr.query("validity"))

    def test_freecad_is_the_exact_one(self):
        """Same op stream, both kernels: frep is grid-limited, freecad is exact."""
        from harnesscad.io.backends.frep import FRepBackend

        def build(cls):
            b = cls()
            b.apply(NewSketch(plane="XY"))
            b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
            b.apply(Extrude(sketch="sk1", distance=PLATE_T))
            b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                         diameter=HOLE_D, through=True))
            b.apply(Hole(face_or_sketch="solid", x=30.0, y=12.0,
                         diameter=HOLE_D, through=True))
            return b

        fc_err = abs(build(FreeCADBackend).query("measure")["volume"] - ANALYTIC_2HOLE)
        fr_err = abs(build(FRepBackend).query("measure")["volume"] - ANALYTIC_2HOLE)
        self.assertLess(fc_err, 1e-6)      # B-rep: exact
        self.assertLess(fc_err, fr_err)    # and strictly better than the SDF grid


if __name__ == "__main__":
    unittest.main()
