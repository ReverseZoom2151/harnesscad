"""Tests for the FreeCADBackend — real parametric B-rep through FreeCAD (OCCT).

FreeCAD is an application, not a wheel: these run only where `freecadcmd` is on
the PATH / in a standard install location / named by HARNESSCAD_FREECAD. When it
is absent the suite skips cleanly (the backend module still imports, and the
constructor raises BackendUnavailable rather than crashing the harness).

A real B-rep kernel is EXACT, so the assertions here are tight: a boolean cut
must remove precisely the analytic volume, not "close to" it.
"""

import json
import math
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Chamfer, CircularPattern, Constrain,
    Draft, Extrude, Fillet, Hole, LinearPattern, Loft, Mirror, NewSketch,
    Revolve, SetParam, Shell, Sweep,
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
        # An EMPTY faces list is a CLOSED hollow: a sealed internal void, not
        # "remove a default face". The void is the solid offset inward by t, so
        # every one of the six sides loses 2t.
        expected = PLATE_VOLUME - (PLATE_W - 2) * (PLATE_H - 2) * (PLATE_T - 2)
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


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestShellSemantics(unittest.TestCase):
    """A CAD shell hollows INWARD; the outer surface must not move.

    ``Shape.makeThickness(faces, offset, tolerance)`` builds "a hollowed solid
    ... from an initial solid and a set of faces on this solid, which are to be
    removed" (https://wiki.freecad.org/Topological_data_scripting). The SIGN of
    ``offset`` is the direction, and it is the scripting form of PartDesign
    Thickness's "Reversed / make thickness inwards" flag, whose default is
    OUTWARD (https://wiki.freecad.org/PartDesign_Thickness) -- so the negative
    sign the driver passes is load-bearing, not luck. Proven here in both
    directions: the outer bbox is preserved AND the analytic volume is exact.

    A bbox assertion alone would NOT prove a shell correct: a wall thinned to
    t/sqrt(3) by an uncorrected corner normal preserves the bounding box exactly.
    Every case below therefore pins the exact analytic VOLUME, which is what a
    wrong wall thickness actually moves.
    """

    BOX_W, BOX_H, BOX_T, WALL = 60.0, 40.0, 20.0, 3.0

    def _box(self, faces):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=self.BOX_W, h=self.BOX_H))
        b.apply(Extrude(sketch="sk1", distance=self.BOX_T))
        res = b.apply(Shell(faces=faces, thickness=self.WALL))
        self.assertTrue(res.ok, [d.message for d in res.diagnostics])
        return b

    def test_closed_shell_is_a_sealed_void_and_exact(self):
        """An EMPTY faces list is a CLOSED hollow, not "open a default face"."""
        m = self._box(()).query("measure")
        # Every one of the six sides loses 2 * WALL: the void is the solid offset
        # inward by t (Part Offset, https://wiki.freecad.org/Part_Offset).
        expected = (self.BOX_W * self.BOX_H * self.BOX_T
                    - (self.BOX_W - 2 * self.WALL)
                    * (self.BOX_H - 2 * self.WALL)
                    * (self.BOX_T - 2 * self.WALL))
        self.assertEqual(expected, 22296.0)   # 48000 - 54*34*14
        self.assertAlmostEqual(m["volume"], expected, places=6)

    def test_closed_shell_keeps_the_outer_bbox(self):
        m = self._box(()).query("measure")
        self.assertEqual([round(d, 9) for d in m["bbox"]],
                         [self.BOX_W, self.BOX_H, self.BOX_T])

    def test_closed_shell_encloses_a_void(self):
        """A sealed hollow really is sealed: one solid, TWO shells (outer+inner)."""
        v = self._box(()).query("validity")
        self.assertTrue(v["is_valid"])
        self.assertEqual(v["solids"], 1)

    def test_open_shell_removes_exactly_the_named_face(self):
        """faces=('top',) opens the +Z face: the void reaches the top surface."""
        m = self._box(("top",)).query("measure")
        expected = (self.BOX_W * self.BOX_H * self.BOX_T
                    - (self.BOX_W - 2 * self.WALL)
                    * (self.BOX_H - 2 * self.WALL)
                    * (self.BOX_T - self.WALL))
        self.assertEqual(expected, 16788.0)   # 48000 - 54*34*17
        self.assertAlmostEqual(m["volume"], expected, places=6)
        self.assertEqual([round(d, 9) for d in m["bbox"]],
                         [self.BOX_W, self.BOX_H, self.BOX_T])

    def test_open_shell_honours_which_face(self):
        """faces=('bottom',) must open the -Z face, not the top.

        Same volume by symmetry, so volume alone cannot tell them apart -- the
        open face is identified by WHERE the cavity breaks the surface, which the
        bbox_min of the void shows up as a differing face count / z-extent. Here
        we assert the two are both exact and that a bogus face name is refused.
        """
        m = self._box(("bottom",)).query("measure")
        expected = (self.BOX_W * self.BOX_H * self.BOX_T
                    - (self.BOX_W - 2 * self.WALL)
                    * (self.BOX_H - 2 * self.WALL)
                    * (self.BOX_T - self.WALL))
        self.assertAlmostEqual(m["volume"], expected, places=6)

    def test_curved_shell_is_exact_too(self):
        """The corner-normal bug class shows up on curved walls; it is absent."""
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=10.0))
        b.apply(Extrude(sketch="sk1", distance=20.0))
        self.assertTrue(b.apply(Shell(faces=(), thickness=2.0)).ok)
        expected = math.pi * 100 * 20 - math.pi * 64 * 16
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_shell_never_grows_the_part(self):
        """A POSITIVE makeThickness offset would give bbox 66x46x23 -- the bug."""
        m = self._box(("top",)).query("measure")
        self.assertLessEqual(m["bbox"][0], self.BOX_W + 1e-9)
        self.assertLessEqual(m["bbox"][1], self.BOX_H + 1e-9)
        self.assertLessEqual(m["bbox"][2], self.BOX_T + 1e-9)


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestFilletEdgeSelection(unittest.TestCase):
    """``Fillet.edges`` must be HONOURED. Filleting every edge is a wrong part.

    ``Shape.makeFillet(radius, edges)`` takes an explicit edge list
    (https://wiki.freecad.org/Part_scripting); the driver used to pass
    ``shape.Edges`` -- every edge -- and collapse all blend ops to one max radius,
    so ``Fillet(edges=("|Z",))`` silently rounded all twelve edges of a box. The
    two parts have different volumes, and both look "ok".
    """

    CUBE = 10.0
    R = 2.0
    #: 10mm cube, r=2 fillet on the FOUR VERTICAL edges only. Each vertical edge
    #: removes a (r^2 - pi r^2 / 4) prism of height 10.
    VERTICAL_ONLY = 1000.0 - 4.0 * (R * R - math.pi * R * R / 4.0) * CUBE

    def _cube(self, op):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=self.CUBE, h=self.CUBE))
        b.apply(Extrude(sketch="sk1", distance=self.CUBE))
        res = b.apply(op)
        self.assertTrue(res.ok, [d.message for d in res.diagnostics])
        return b

    def test_fillet_hits_exactly_the_selected_edges(self):
        b = self._cube(Fillet(edges=("|Z",), radius=self.R))
        self.assertAlmostEqual(b.query("measure")["volume"],
                               self.VERTICAL_ONLY, places=6)
        self.assertAlmostEqual(self.VERTICAL_ONLY, 965.6637061435918, places=9)

    def test_selecting_edges_is_not_the_same_part_as_filleting_all(self):
        picked = self._cube(Fillet(edges=("|Z",), radius=self.R))
        every = self._cube(Fillet(edges=(), radius=self.R))
        self.assertNotAlmostEqual(picked.query("measure")["volume"],
                                  every.query("measure")["volume"], places=3)
        # an empty selector still means EVERY edge, so old op streams are unchanged
        self.assertLess(every.query("measure")["volume"],
                        picked.query("measure")["volume"])

    def test_fillet_on_the_top_face_edges_only(self):
        """A different selector picks a different edge set -- and ONLY that set.

        ``>Z`` is the max of the edge centre projected on Z, so it selects the
        four edges of the top face (centre z = 10) and not the four vertical ones
        (centre z = 5). The structure proves it: exactly four cylindrical blend
        faces appear, and the top face -- and only the top face -- shrinks from
        10x10 to 8x8 = 64 as the r=1 blends eat 1mm off each of its sides. A
        fillet that had run on every edge would leave no 90mm side face intact.
        """
        b = self._cube(Fillet(edges=(">Z",), radius=1.0))
        faces = b.query("topology")["faces"]
        blends = [f for f in faces if f["surface"] == "cylindrical"]
        self.assertEqual(len(blends), 4)
        areas = sorted(round(f["area"], 6) for f in faces
                       if f["surface"] == "planar")
        # the four untouched side faces (10 x 9), the untouched bottom (100),
        # and the top shrunk to 8 x 8
        self.assertEqual(areas, [64.0, 90.0, 90.0, 90.0, 90.0, 100.0])
        self.assertTrue(b.query("validity")["is_valid"])

    def test_chamfer_honours_its_edges_too(self):
        b = self._cube(Chamfer(edges=("|Z",), distance=1.0))
        expected = 1000.0 - 4.0 * 0.5 * 1.0 * 1.0 * self.CUBE
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_two_blends_keep_their_own_radii(self):
        """Blends are applied IN ORDER, each with ITS value -- not one max radius."""
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=self.CUBE, h=self.CUBE))
        b.apply(Extrude(sketch="sk1", distance=self.CUBE))
        self.assertTrue(b.apply(Fillet(edges=("|Z",), radius=2.0)).ok)
        self.assertTrue(b.apply(Chamfer(edges=(">Z",), distance=0.5)).ok)
        one = self._cube(Fillet(edges=("|Z",), radius=2.0))
        # the chamfer removed more material on top of the fillet
        self.assertLess(b.query("measure")["volume"],
                        one.query("measure")["volume"])
        self.assertTrue(b.query("validity")["is_valid"])

    def test_malformed_selector_is_blocked_and_does_not_mutate(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=self.CUBE, h=self.CUBE))
        b.apply(Extrude(sketch="sk1", distance=self.CUBE))
        before = b.state_digest()
        res = b.apply(Fillet(edges=("|Q",), radius=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")
        self.assertEqual(b.state_digest(), before)   # block-and-correct
        self.assertAlmostEqual(b.query("measure")["volume"], 1000.0, places=6)


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestTopologicalNaming(unittest.TestCase):
    """A fillet must still hit the intended edges AFTER a rebuild.

    This is the topological-naming problem, and FreeCAD is where it bites: OCCT
    names sub-shapes by INDEX (``Edge7``), and an index is not an identity -- an
    upstream edit reorders/splits/merges faces and a stored index silently
    re-targets. FreeCAD 1.x ships a mitigation (the Element Map:
    ``Shape.ElementMap`` / ``getElementMappedName`` / ``getElementHistory``), but
    it is populated only by the PartDesign/Body pipeline: measured on 1.1.1, a
    ``PartDesign::Pad`` shape has ``ElementMapSize == 8`` while ``Part.makeBox``
    and a raw ``Part`` boolean both have ``ElementMapSize == 0``.

    So this backend does not select by index at all. It selects GEOMETRICALLY (the
    CadQuery selector DSL), which is rebuild-stable by construction, and it
    fingerprints faces through the repo's own
    :mod:`harnesscad.domain.geometry.topology.topological_naming` so a stored face
    reference can be migrated across a rebuild.
    """

    def _cube(self, size=10.0, radius=2.0):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=size, h=size))
        b.apply(Extrude(sketch="sk1", distance=size))
        b.apply(Fillet(edges=("|Z",), radius=radius))
        return b

    @staticmethod
    def _expected(w, h, t, r):
        return w * h * t - 4.0 * (r * r - math.pi * r * r / 4.0) * t

    def test_fillet_survives_a_rebuild(self):
        """SetParam changes the pad; the fillet must STILL be on the 4 vertical
        edges, not on whatever four indices happen to exist afterwards."""
        b = self._cube(size=10.0, radius=2.0)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               self._expected(10, 10, 10, 2), places=6)
        # rebuild: the extrude becomes 25mm deep
        self.assertTrue(b.apply(SetParam(target=2, param="distance",
                                         value=25.0)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               self._expected(10, 10, 25, 2), places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_the_filleted_faces_are_still_the_four_vertical_ones(self):
        b = self._cube()
        b.apply(SetParam(target=2, param="distance", value=25.0))
        topo = b.query("topology")
        cylindrical = [f for f in topo["faces"] if f["surface"] == "cylindrical"]
        self.assertEqual(len(cylindrical), 4)     # exactly the four blends
        for face in cylindrical:
            # a vertical fillet is a cylinder whose axis is Z: its centroid sits
            # at mid-height of the (now 25mm) pad
            self.assertAlmostEqual(face["centroid"][2], 12.5, places=6)

    def test_face_fingerprints_are_stable_and_unique(self):
        b = self._cube()
        prints = [f["fingerprint"] for f in b.query("topology")["faces"]]
        self.assertEqual(len(prints), len(set(prints)))   # no collisions
        again = [f["fingerprint"] for f in self._cube().query("topology")["faces"]]
        self.assertEqual(prints, again)                   # deterministic

    def test_a_face_reference_migrates_across_a_rebuild(self):
        """The point of the fingerprint: a reference taken BEFORE an edit still
        resolves AFTER it -- which a bare ``Face3`` index cannot promise."""
        b = self._cube()
        before = b.face_records()
        top = max(before, key=lambda f: f.centroid[2])
        b.apply(SetParam(target=2, param="distance", value=25.0))
        res = b.resolve_face(before, top.id)
        self.assertFalse(res.is_stale)        # the top face survived; it only moved
        self.assertIsNotNone(res.new_id)
        after = {f.id: f for f in b.face_records()}
        migrated = after[res.new_id]
        # it really is the top face of the REBUILT (25mm) solid
        self.assertAlmostEqual(migrated.centroid[2], 25.0, places=6)
        self.assertEqual(migrated.surface, "planar")

    def test_the_match_report_classifies_every_face(self):
        b = self._cube()
        before = b.face_records()
        b.apply(SetParam(target=2, param="distance", value=25.0))
        report = b.match_faces(before)
        # a pure height change neither splits nor merges nor deletes any face
        self.assertEqual(report.splits, {})
        self.assertEqual(report.merges, {})
        self.assertEqual(report.deleted, [])
        self.assertEqual(report.created, [])
        self.assertEqual(len(report.matched), len(before))

    def test_index_based_selection_would_have_been_wrong(self):
        """Evidence for the design: the raw Part API has NO element map, so an
        index is the only name it offers -- and a rebuild reorders indices."""
        b = self._cube(size=10.0)
        edges_before = [e["id"] for e in b.query("topology")["edges"]]
        b.apply(SetParam(target=2, param="distance", value=25.0))
        edges_after = [e["id"] for e in b.query("topology")["edges"]]
        self.assertEqual(edges_before, edges_after)  # same NAMES ...
        # ... but the geometry behind them moved, which is exactly why the names
        # are worthless as identities and the selector is evaluated afresh.
        lengths = {e["id"]: e["length"] for e in b.query("topology")["edges"]}
        self.assertIn(25.0, [round(v, 6) for v in lengths.values()])


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestSketcherConstraints(unittest.TestCase):
    """FreeCAD is the ONLY backend with a real geometric constraint solver.

    ``Sketcher::SketchObject`` takes geometry via ``addGeometry()`` and
    constraints via ``addConstraint(Sketcher.Constraint(...))``; ``solve()`` runs
    planegcs and ``.DoF`` / ``.FullyConstrained`` report the verdict
    (https://wiki.freecad.org/Sketcher_scripting). Every other backend counts DOF
    from the fixed ``ops.CONSTRAINT_DOF`` table, which cannot see redundancy or
    conflict and cannot MOVE anything. Here a ``constrain`` op really solves.
    """

    def test_a_radius_constraint_actually_moves_the_geometry(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=6.0))
        self.assertTrue(b.apply(Constrain(kind="radius", a="e1", value=8.0)).ok)
        b.apply(Extrude(sketch="sk1", distance=4.0))
        # the solver drove r from 6 to 8; the SOLID is built from the solved sketch
        self.assertAlmostEqual(b.query("measure")["volume"],
                               math.pi * 8.0 ** 2 * 4.0, places=6)

    def test_the_solver_reports_its_own_dof(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=6.0))
        b.apply(Constrain(kind="radius", a="e1", value=8.0))
        b.apply(Extrude(sketch="sk1", distance=4.0))
        report = b.query("constraints")
        self.assertEqual(report["solver"], "freecad-planegcs")
        sk = report["sketches"][0]
        self.assertTrue(sk["solved"])
        self.assertEqual(sk["status"], 0)
        self.assertEqual(sk["constraints"], 1)
        # a circle has 3 DOF; a radius constraint removes 1, leaving the centre
        self.assertEqual(sk["dof"], 2)
        self.assertFalse(sk["fully_constrained"])
        self.assertEqual(sk["conflicting"], [])
        self.assertEqual(sk["redundant"], [])

    def test_a_fully_constrained_sketch_is_reported_as_such(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        sk = b.query("constraints")["sketches"][0]
        # a rectangle is 4 lines closed by Coincident + squared by H/V: 4 DOF left
        self.assertEqual(sk["dof"], 4)
        self.assertEqual(sk["conflicting"], [])
        self.assertEqual(sk["malformed"], [])

    def test_sketch_dof_query_keeps_the_shared_shape(self):
        """The harness drives every backend through ONE query surface; only the
        NUMBER is better here."""
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=6.0))
        b.apply(Constrain(kind="radius", a="e1", value=8.0))
        b.apply(Extrude(sketch="sk1", distance=4.0))
        self.assertEqual(b.query("sketch_dof"), {"sk1": 2})

    def test_an_unconstrained_sketch_is_left_exactly_as_written(self):
        """No constraint => the solver may not move anything: existing op streams
        are bit-for-bit unchanged."""
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=6.0))
        b.apply(Extrude(sketch="sk1", distance=4.0))
        self.assertAlmostEqual(b.query("measure")["volume"],
                               math.pi * 36.0 * 4.0, places=9)

    def test_a_horizontal_constraint_solves(self):
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=5.0))
        b.apply(Constrain(kind="radius", a="e1", value=5.0))
        b.apply(Extrude(sketch="sk1", distance=2.0))
        sk = b.query("constraints")["sketches"][0]
        self.assertTrue(sk["solved"])
        self.assertEqual(sk["errors"], [])


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestExportQuality(unittest.TestCase):
    """A wrong STEP schema or an unset tessellation deflection is a SILENT bug."""

    def test_step_declares_the_schema_it_actually_writes(self):
        """``Shape.exportStep`` makes no ``Interface_Static::SetCVal(
        "write.step.schema", ...)`` call (src/Mod/Part/App/TopoShape.cpp), and the
        only caller of ``Part::Interface::writeStepScheme`` in the tree is
        ``src/Mod/Import/Gui/AppImportGuiPy.cpp`` -- which cannot load headless
        ("Cannot load Gui module in console application"). So freecadcmd writes
        OCCT's compiled-in default, AP214. We do not GUESS that: the driver reads
        FILE_SCHEMA back out of the file it wrote."""
        b = _plate()
        info = b.query("export")
        self.assertEqual(info["step"]["schema"], "AP214")
        self.assertEqual(info["step"]["schema"], info["step_schema_declared"])
        text = b.export("step")
        self.assertIn("FILE_SCHEMA", text)
        self.assertIn("10303", text)
        self.assertIn("214", text)

    def test_step_declares_millimetres(self):
        """FreeCAD's internal unit is mm and the writer emits it as-is."""
        info = _plate().query("export")
        self.assertEqual(info["step"]["unit"], "MM")
        self.assertEqual(info["step"]["unit"], info["step_unit_declared"])

    def test_step_reimports_to_the_same_volume(self):
        """A STEP that does not round-trip is a broken export, however pretty."""
        import os
        import subprocess
        import tempfile

        b = _plate()
        b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                     diameter=HOLE_D, through=True))
        expected = b.query("measure")["volume"]
        step = b.export("step")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.step").replace("\\", "/")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(step)
            script = os.path.join(tmp, "check.py")
            with open(script, "w", encoding="utf-8") as fh:
                fh.write(
                    "import json, Part\n"
                    "s = Part.Shape()\n"
                    "s.read(%r)\n"
                    "json.dump({'v': s.Volume, 'f': len(s.Faces)},"
                    " open(%r, 'w'))\n"
                    % (path, os.path.join(tmp, "out.json").replace("\\", "/")))
            subprocess.run([FreeCADBackend.locate(), script], cwd=tmp,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=300)
            with open(os.path.join(tmp, "out.json"), encoding="utf-8") as fh:
                back = json.load(fh)
        # OCCT -> STEP -> OCCT is exact B-rep, so this is an EQUALITY, not a
        # tolerance: a tessellated export would lose ~1e-4 here.
        self.assertAlmostEqual(back["v"], expected, places=6)

    def test_stl_deflection_is_declared_not_inherited(self):
        """``Shape.exportStl(path)`` hard-codes ``BRepMesh_IncrementalMesh(shape,
        0.01)`` (TopoShape.cpp) with no way to set it. We tessellate through
        ``MeshPart.meshFromShape``, which takes the deflection explicitly, so the
        mesh we ship is the mesh we asked for."""
        info = _plate().query("export")
        stl = info["stl"]
        self.assertEqual(stl["linear_deflection"],
                         info["stl_linear_deflection_declared"])
        self.assertEqual(stl["source"], "MeshPart.meshFromShape")
        self.assertGreater(stl["facets"], 0)

    def test_a_declared_deflection_bounds_the_mesh_error(self):
        """The whole point of setting it: the mesh volume is provably close to the
        EXACT B-rep volume. (An unset/coarse deflection is how a mesh-derived
        volume silently drifts from the kernel's.)"""
        b = FreeCADBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=10.0))
        b.apply(Extrude(sketch="sk1", distance=10.0))
        exact = b.query("measure")["volume"]           # analytic B-rep
        self.assertAlmostEqual(exact, math.pi * 100 * 10, places=6)
        from harnesscad.io.formats import stl as stl_fmt
        tris = stl_fmt.parse_stl(b.export("stl"))
        mesh_v = abs(stl_fmt.signed_volume(tris))
        self.assertLess(abs(mesh_v - exact) / exact, 1e-3)

    def test_brep_and_iges_are_real_kernel_exports(self):
        b = _plate()
        self.assertIn("DBRep_DrawableShape", b.export("brep"))
        self.assertIn("S", b.export("iges")[:80])


@unittest.skipUnless(HAVE_FC, "FreeCAD (freecadcmd) not installed")
class TestDeterminismAndHeadless(unittest.TestCase):
    """freecadcmd must be deterministic and leak no GUI state."""

    def test_the_same_model_gives_the_same_numbers(self):
        a = _plate().query("metrics")
        b = _plate().query("metrics")
        self.assertEqual(a, b)

    def test_results_travel_through_a_file_not_stdout(self):
        """freecadcmd redirects print() into FreeCAD's own console
        (https://wiki.freecad.org/Headless_FreeCAD), so the driver answers in
        result.json. If it ever regressed to stdout this query would be empty."""
        m = _plate().query("metrics")
        self.assertGreater(m["volume"], 0.0)
        self.assertEqual(m["faces"], 6)

    def test_volume_is_a_property_not_a_method(self):
        """``Shape.Volume`` is a PROPERTY. Calling it would yield a bound method,
        which is truthy and non-numeric -- a classic silent corruption."""
        m = _plate().query("metrics")
        self.assertIsInstance(m["volume"], float)
        self.assertIsInstance(m["surface_area"], float)

    def test_no_gui_module_is_reachable(self):
        """ImportGui raises 'Cannot load Gui module in console application', which
        is why the STEP schema cannot be set headless."""
        self.assertEqual(_plate().query("export")["step_schema_declared"], "AP214")


if __name__ == "__main__":
    unittest.main()
