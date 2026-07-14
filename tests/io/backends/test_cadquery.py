"""Tests for the real-geometry CadQueryBackend (OCCT).

These build actual B-rep solids, so they require cadquery / cadquery-ocp. When
cadquery is not importable the whole suite is skipped (the backend module itself
still imports fine — cadquery is imported lazily inside its methods).
"""

import math
import unittest

from harnesscad.io.backends.cadquery import CadQueryBackend
from harnesscad.eval.verifiers.geometry import BRepValidityCheck
from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle, AddCircle, AddLine, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, LinearPattern, CircularPattern, Mirror,
    Draft, Loft, Sweep, SetParam,
)
from harnesscad.eval.verifiers.verify import Severity


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_CQ = _cadquery_available()

# The harness's reference part: a 40 x 24 x 8 plate with two 6mm through-holes.
PLATE_W, PLATE_H, PLATE_T = 40.0, 24.0, 8.0
HOLE_D = 6.0
PLATE_VOLUME = PLATE_W * PLATE_H * PLATE_T
ANALYTIC_2HOLE = PLATE_VOLUME - 2.0 * math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T


def _ref_plate() -> CadQueryBackend:
    b = CadQueryBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H)).ok
    assert b.apply(Extrude(sketch="sk1", distance=PLATE_T)).ok
    return b


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
        from harnesscad.core.cisp.ops import Constrain
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


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestExactBoolean(unittest.TestCase):
    """A real B-rep kernel removes EXACTLY the analytic volume (no grid error)."""

    def test_two_through_holes_remove_exactly_the_analytic_volume(self):
        b = _ref_plate()
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                                     diameter=HOLE_D, through=True)).ok)
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=30.0, y=12.0,
                                     diameter=HOLE_D, through=True)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], ANALYTIC_2HOLE, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_boolean_cut_removes_exactly_the_cylinder(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
        b.apply(Extrude(sketch="sk1", distance=PLATE_T))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=10.0, cy=12.0, r=HOLE_D / 2.0))
        b.apply(Extrude(sketch="sk2", distance=PLATE_T))
        self.assertTrue(b.apply(Boolean(kind="cut")).ok)
        expected = PLATE_VOLUME - math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_boolean_honours_named_target_and_tool(self):
        """REGRESSION: op.target/op.tool were ignored -- a Boolean that named its
        operands silently cut the last two solids instead, with no diagnostic."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
        b.apply(Extrude(sketch="sk1", distance=PLATE_T))          # f1
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=10.0, cy=12.0, r=HOLE_D / 2.0))
        b.apply(Extrude(sketch="sk2", distance=PLATE_T))          # f2
        self.assertTrue(b.apply(Boolean(kind="cut", target="f1", tool="f2")).ok)
        expected = PLATE_VOLUME - math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_boolean_with_unknown_tool_is_blocked(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk2", x=0.0, y=0.0, w=4.0, h=4.0))
        b.apply(Extrude(sketch="sk2", distance=5.0))
        before = b.state_digest()
        res = b.apply(Boolean(kind="cut", target="f1", tool="f99"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-ref")
        self.assertEqual(b.state_digest(), before)

    def test_revolve_is_exact(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddRectangle(sketch="sk1", x=10.0, y=0.0, w=5.0, h=20.0))
        self.assertTrue(b.apply(Revolve(sketch="sk1", angle=360.0,
                                        axis=(0, 0, 0, 0, 1, 0))).ok)
        expected = math.pi * (15.0 ** 2 - 10.0 ** 2) * 20.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_shell_is_exact(self):
        """SETTLED SPEC: an EMPTY faces tuple is a CLOSED HOLLOW -- a sealed
        internal void, no face opened. CadQuery's free `hollow` documents it: "if
        no faces provided a watertight solid will be constructed". So the cavity
        is inset by t on ALL SIX sides (this used to assert the top-open result,
        because the backend silently defaulted the open face to '>Z')."""
        b = _ref_plate()
        self.assertTrue(b.apply(Shell(faces=(), thickness=1.0)).ok)
        expected = (PLATE_VOLUME
                    - (PLATE_W - 2) * (PLATE_H - 2) * (PLATE_T - 2))
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestDegenerateGuards(unittest.TestCase):
    def test_zero_volume_revolve_is_rejected(self):
        """REGRESSION: revolving about an axis normal to the sketch plane yields a
        topologically non-empty solid of ZERO volume. The old guard only counted
        solids, so this committed a phantom body and reported ok."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddRectangle(sketch="sk1", x=10.0, y=0.0, w=5.0, h=20.0))
        before = b.state_digest()
        res = b.apply(Revolve(sketch="sk1", angle=360.0, axis=(0, 0, 0, 0, 0, 1)))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "degenerate")
        self.assertFalse(b.query("summary")["solid_present"])
        self.assertEqual(b.state_digest(), before)

    def test_unknown_sketch_plane_is_rejected_at_sketch_time(self):
        """REGRESSION: an unknown plane used to sail through NewSketch and only
        explode inside _extrude, reported as a kernel-error on the wrong op."""
        res = CadQueryBackend().apply(NewSketch(plane="BOGUS"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestEdgeBlends(unittest.TestCase):
    def test_fillet_rounds_every_edge(self):
        base = _ref_plate().query("metrics")
        b = _ref_plate()
        self.assertTrue(b.apply(Fillet(edges=(), radius=1.0)).ok)
        filleted = b.query("metrics")
        self.assertGreater(filleted["faces"], base["faces"])
        self.assertLess(filleted["volume"], base["volume"])
        self.assertTrue(b.query("validity")["is_valid"])

    def test_chamfer_removes_more_than_the_equal_fillet(self):
        b = _ref_plate()
        self.assertTrue(b.apply(Chamfer(edges=(), distance=1.0)).ok)
        f = _ref_plate()
        f.apply(Fillet(edges=(), radius=1.0))
        self.assertLess(b.query("measure")["volume"], f.query("measure")["volume"])


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestPatterns(unittest.TestCase):
    def test_linear_pattern_is_exact(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(LinearPattern(count=3, spacing=20.0,
                                              direction=(1, 0, 0))).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 1500.0, places=6)

    def test_linear_pattern_normalises_its_direction(self):
        """REGRESSION: a non-unit direction scaled the spacing (direction=(2,0,0),
        spacing=20 stepped 40mm here but 20mm on frep)."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(LinearPattern(count=2, spacing=20.0,
                                              direction=(2, 0, 0))).ok)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 1000.0, places=6)
        # a 20mm step from a 10mm-wide part spans 30mm, NOT 50mm
        self.assertAlmostEqual(m["bbox"][0], 30.0, places=6)

    def test_circular_pattern_and_mirror_are_exact(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=4, angle=360.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 2000.0, places=6)

        m = CadQueryBackend()
        m.apply(NewSketch(plane="XY"))
        m.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        m.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(m.apply(Mirror(plane="YZ")).ok)
        self.assertAlmostEqual(m.query("measure")["volume"], 1000.0, places=6)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestExport(unittest.TestCase):
    def test_step_round_trips_through_the_kernel(self):
        """The headline claim: this is the only backend with real STEP B-rep."""
        import os
        import tempfile
        import cadquery as cq

        b = _ref_plate()
        b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                     diameter=HOLE_D, through=True))
        b.apply(Hole(face_or_sketch="solid", x=30.0, y=12.0,
                     diameter=HOLE_D, through=True))
        step = b.export("step")
        self.assertIn("ISO-10303", step)
        self.assertIn("MANIFOLD_SOLID_BREP", step)

        fd, path = tempfile.mkstemp(suffix=".step")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(step)
            reimported = cq.importers.importStep(path)
            volume = reimported.val().Volume()
        finally:
            os.remove(path)
        # The re-imported solid is the SAME solid: exact B-rep in, exact B-rep out.
        self.assertAlmostEqual(volume, ANALYTIC_2HOLE, places=6)
        self.assertAlmostEqual(volume, b.query("measure")["volume"], places=9)

    def test_stl_export_is_parseable_ascii(self):
        """REGRESSION: export('stl') wrote BINARY and read it back as UTF-8."""
        stl = _ref_plate().export("stl")
        self.assertTrue(stl.lstrip().startswith("solid"))
        self.assertIn("facet normal", stl)
        self.assertIn("endsolid", stl)

    def test_brep_export_is_the_native_kernel_format(self):
        self.assertIn("CASCADE", _ref_plate().export("brep"))

    def test_unsupported_format_is_a_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            _ref_plate().export("dwg")
        self.assertIn("cannot export", str(ctx.exception))

    def test_export_with_no_solid_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            CadQueryBackend().export("step")


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestUnsupportedOpsAreTyped(unittest.TestCase):
    def test_bad_refs_in_loft_and_sweep_never_silently_no_op(self):
        """loft/sweep are REAL now (see TestLoftSweepDraft), but a bad sketch ref
        must still block-and-correct rather than fabricate a body."""
        for op in (Loft(sketches=("sk1", "nope")),
                   Sweep(sketch="sk1", path="nope")):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok, "%s must not silently no-op" % type(op).__name__)
            self.assertTrue(res.diagnostics)
            self.assertEqual(res.diagnostics[0].severity, Severity.ERROR)
            self.assertEqual(res.diagnostics[0].code, "bad-ref")
            self.assertEqual(b.state_digest(), before)

    def test_unknown_hole_kind_is_rejected(self):
        b = _ref_plate()
        res = b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                           diameter=HOLE_D, through=True, kind="tapped"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestSetParam(unittest.TestCase):
    def test_set_param_replays_the_op_stream(self):
        b = _ref_plate()
        self.assertTrue(b.apply(SetParam(target=2, param="distance", value=16.0)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               PLATE_W * PLATE_H * 16.0, places=6)

    def test_failed_set_param_does_not_mutate(self):
        b = _ref_plate()
        before = b.state_digest()
        res = b.apply(SetParam(target=99, param="distance", value=16.0))
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestFrepParity(unittest.TestCase):
    """The harness must drive cadquery and frep through the SAME surface."""

    def test_query_surface_matches_frep(self):
        from harnesscad.io.backends.frep import FRepBackend

        def build(cls):
            b = cls()
            b.apply(NewSketch(plane="XY"))
            b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
            b.apply(Extrude(sketch="sk1", distance=PLATE_T))
            return b

        cq_b, fr_b = build(CadQueryBackend), build(FRepBackend)
        self.assertEqual(cq_b.query("summary"), fr_b.query("summary"))
        self.assertEqual(cq_b.query("sketch_dof"), fr_b.query("sketch_dof"))
        for key in ("volume", "bbox"):
            self.assertIn(key, cq_b.query("measure"))
            self.assertIn(key, fr_b.query("measure"))
        for key in ("manifold", "watertight", "is_valid", "solid_present"):
            self.assertIn(key, cq_b.query("validity"))
            self.assertIn(key, fr_b.query("validity"))

    def test_cadquery_is_exact_where_frep_is_grid_limited(self):
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

        cq_err = abs(build(CadQueryBackend).query("measure")["volume"] - ANALYTIC_2HOLE)
        fr_err = abs(build(FRepBackend).query("measure")["volume"] - ANALYTIC_2HOLE)
        self.assertLess(cq_err, 1e-6)     # B-rep: exact to machine precision
        self.assertLess(cq_err, fr_err)   # and strictly better than the SDF grid


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestShellSemantics(unittest.TestCase):
    """Workplane.shell(thickness, kind='arc') -- classreference.html:
    'Negative values shell inwards, positive values shell outwards.'

    A CAD shell HOLLOWS the part, so we must pass -thickness and the outer
    bounding box must not move. This is the bug frep/blender have."""

    def _box(self, w, h, t):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h))
        b.apply(Extrude(sketch="sk1", distance=t))
        return b

    def test_shell_does_not_grow_the_part(self):
        """The headline invariant: a 60x40x20 box shelled 3mm stays 60x40x20.
        A POSITIVE thickness would give 66x46x23 (examples.html: a positive
        thickness "wraps an object ... the original object will be the 'hollowed
        out' portion")."""
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(), thickness=3.0)).ok)
        bbox = b.query("measure")["bbox"]
        self.assertEqual([round(d, 6) for d in bbox], [60.0, 40.0, 20.0])

    def test_empty_faces_is_a_sealed_closed_hollow_of_the_exact_volume(self):
        """SETTLED SPEC + the wall-thickness trap.

        A bbox check CANNOT prove a shell is right: an inward shell with a wall
        42% too thin preserves the bbox exactly. Assert the ANALYTIC volume.

        Empty faces = closed hollow (free `hollow`: "if no faces provided a
        watertight solid will be constructed"), so the void is inset by t on all
        six sides: 60x40x20 at t=3 -> 48000 - 54*34*14 = 22296 exactly."""
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(), thickness=3.0)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 22296.0, places=6)
        self.assertEqual([round(d, 6) for d in b.query("measure")["bbox"]],
                         [60.0, 40.0, 20.0])
        self.assertTrue(b.query("validity")["is_valid"])
        # a sealed void is a solid with TWO shells (outer + inner), no opening
        self.assertEqual(b.query("metrics")["solids"], 1)

    def test_wall_thickness_is_exactly_t_on_every_axis(self):
        """Independent of volume: the void's bbox must be inset by exactly t on
        each side. This is the check that catches a too-thin wall."""
        t = 3.0
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(), thickness=t)).ok)
        solid = b._solids[-1].val()
        inner = [s for s in solid.Shells()]
        self.assertEqual(len(inner), 2)          # outer + the sealed void
        voids = sorted(inner, key=lambda s: s.BoundingBox().xlen)
        vb = voids[0].BoundingBox()              # the smaller shell = the cavity
        self.assertAlmostEqual(vb.xlen, 60.0 - 2 * t, places=6)
        self.assertAlmostEqual(vb.ylen, 40.0 - 2 * t, places=6)
        self.assertAlmostEqual(vb.zlen, 20.0 - 2 * t, places=6)

    def test_open_shell_volume_is_the_analytic_hollow(self):
        """Naming >Z REMOVES the top face, so the cavity opens upward: the void
        is inset t on the four walls and the floor but runs clear through the top.
        60x40x20 at t=3 -> 48000 - 54*34*17."""
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(">Z",), thickness=3.0)).ok)
        expected = 60.0 * 40.0 * 20.0 - 54.0 * 34.0 * 17.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_sealed_hollow_holds_more_material_than_an_opened_one(self):
        """The two spellings are DIFFERENT geometry -- which is exactly why
        defaulting the empty tuple to '>Z' was a silent correctness bug."""
        sealed = self._box(60.0, 40.0, 20.0)
        sealed.apply(Shell(faces=(), thickness=3.0))
        opened = self._box(60.0, 40.0, 20.0)
        opened.apply(Shell(faces=(">Z",), thickness=3.0))
        self.assertGreater(sealed.query("measure")["volume"],
                           opened.query("measure")["volume"])
        self.assertNotAlmostEqual(sealed.query("measure")["volume"],
                                  opened.query("measure")["volume"], places=3)

    def test_shell_honours_the_named_open_face(self):
        """REGRESSION: op.faces was IGNORED -- the TOP face was always removed, so
        Shell(faces=('<Z',)) silently opened the wrong side. Opening the bottom
        must leave the top face solid: the two shells are different solids, and
        the one opened at <Z has its cavity at the bottom."""
        top = self._box(60.0, 40.0, 20.0)
        top.apply(Shell(faces=(">Z",), thickness=3.0))
        bot = self._box(60.0, 40.0, 20.0)
        self.assertTrue(bot.apply(Shell(faces=("<Z",), thickness=3.0)).ok)
        # same volume by symmetry, but MIRRORED in Z -> different centre of mass.
        # Removing >Z leaves the floor + walls, so the material (and the COM) sits
        # LOW; removing <Z leaves the ceiling + walls, so the COM sits HIGH.
        self.assertAlmostEqual(top.query("measure")["volume"],
                               bot.query("measure")["volume"], places=6)
        top_z = top.query("metrics")["center_of_mass"][2]
        bot_z = bot.query("metrics")["center_of_mass"][2]
        self.assertLess(top_z, 10.0)      # below the box's mid-height
        self.assertGreater(bot_z, 10.0)   # above it
        self.assertLess(top_z, bot_z)
        # and the outer box is still untouched in both
        for b in (top, bot):
            self.assertEqual([round(d, 6) for d in b.query("measure")["bbox"]],
                             [60.0, 40.0, 20.0])

    def test_shell_can_open_several_faces_at_once(self):
        """classreference.html: 'You can also select multiple faces at once' --
        Workplane().box(10,10,10).faces('>Z or >X or <Y').shell(1). A CISP tuple
        of selectors is their union, so more open faces means less material."""
        one = self._box(60.0, 40.0, 20.0)
        one.apply(Shell(faces=(">Z",), thickness=3.0))
        three = self._box(60.0, 40.0, 20.0)
        self.assertTrue(three.apply(Shell(faces=(">Z", ">X", "<Y"),
                                          thickness=3.0)).ok)
        self.assertLess(three.query("measure")["volume"],
                        one.query("measure")["volume"])
        self.assertEqual([round(d, 6) for d in three.query("measure")["bbox"]],
                         [60.0, 40.0, 20.0])   # still does not grow

    def test_shell_kind_is_validated(self):
        b = self._box(20.0, 20.0, 20.0)
        before = b.state_digest()
        res = b.apply(Shell(faces=(), thickness=1.0, kind="bogus"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")
        self.assertEqual(b.state_digest(), before)

    def test_too_thick_shell_blocks_without_mutating(self):
        b = self._box(20.0, 20.0, 20.0)
        before = b.state_digest()
        res = b.apply(Shell(faces=(), thickness=50.0))
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestEdgeSelection(unittest.TestCase):
    """REGRESSION: Fillet.edges / Chamfer.edges were IGNORED and EVERY edge was
    blended. A fillet on the wrong edge set is a silent correctness bug.

    op.edges is now a tuple of CadQuery selector strings (selectors.html): '|Z'
    = edges parallel to Z, '>Z' = the top face's edges, etc."""

    def _box(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        return b

    def test_fillet_hits_only_the_selected_edges(self):
        """A 20x10x5 box has 12 edges, 4 of them vertical ('|Z'). Rounding only
        those gives 10 faces / V=995.708; rounding all 12 gives 26 faces /
        V=971.295. Two different parts -- so the selector must be honoured."""
        v = self._box()
        self.assertTrue(v.apply(Fillet(edges=("|Z",), radius=1.0)).ok)
        vm = v.query("metrics")
        self.assertEqual(vm["faces"], 10)
        self.assertAlmostEqual(vm["volume"], 995.7079632679, places=6)

        a = self._box()
        self.assertTrue(a.apply(Fillet(edges=(), radius=1.0)).ok)
        am = a.query("metrics")
        self.assertEqual(am["faces"], 26)          # every edge + every corner
        self.assertLess(am["volume"], vm["volume"])
        self.assertTrue(v.query("validity")["is_valid"])

    def test_fillet_top_face_edges_only(self):
        """'>Z' on edges picks the 4 edges of the top face (selectors.html)."""
        b = self._box()
        self.assertTrue(b.apply(Fillet(edges=(">Z",), radius=1.0)).ok)
        m = b.query("metrics")
        self.assertEqual(m["faces"], 10)   # 6 original + 4 rounded top edges
        # only the top edges moved: the bottom face is still an exact 20x10 rect
        self.assertEqual([round(d, 6) for d in m["bbox"]], [20.0, 10.0, 5.0])

    def test_selector_tuple_is_the_union_of_its_members(self):
        """selectors.html combines selectors with 'or'; a CISP tuple is that
        union, so ('|Z', '>Z') rounds strictly more than either alone."""
        both = self._box()
        self.assertTrue(both.apply(Fillet(edges=("|Z", ">Z"), radius=1.0)).ok)
        only_v = self._box()
        only_v.apply(Fillet(edges=("|Z",), radius=1.0))
        self.assertLess(both.query("measure")["volume"],
                        only_v.query("measure")["volume"])

    def test_chamfer_hits_only_the_selected_edges(self):
        b = self._box()
        self.assertTrue(b.apply(Chamfer(edges=("|Z",), distance=1.0)).ok)
        self.assertEqual(b.query("metrics")["faces"], 10)
        # a 45-deg chamfer removes half of what the same-radius fillet leaves
        expected = 20.0 * 10.0 * 5.0 - 4.0 * 0.5 * 1.0 * 1.0 * 5.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_asymmetric_chamfer_uses_length2(self):
        """Workplane.chamfer(length, length2=None) (classreference.html)."""
        b = self._box()
        self.assertTrue(b.apply(Chamfer(edges=("|Z",), distance=1.0,
                                        distance2=2.0)).ok)
        expected = 20.0 * 10.0 * 5.0 - 4.0 * 0.5 * 1.0 * 2.0 * 5.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_malformed_selector_is_a_typed_bad_value_not_a_kernel_error(self):
        b = self._box()
        before = b.state_digest()
        res = b.apply(Fillet(edges=("|Q",), radius=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")
        self.assertEqual(b.state_digest(), before)

    def test_selector_that_matches_nothing_blocks(self):
        b = self._box()
        before = b.state_digest()
        res = b.apply(Fillet(edges=("%BSPLINE",), radius=1.0))  # a box has none
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)

    def test_our_selector_dsl_agrees_with_cadquerys_own_selectors(self):
        """SELECTOR CORRECTNESS: our DSL is only safe to validate CadQuery
        selectors with if it means the same thing CadQuery means. Differentially
        compare, on a box, our evaluator against cq's real selector engine."""
        import cadquery as cq
        from harnesscad.domain.geometry.topology.selector_dsl import Entity, select

        box = cq.Workplane("XY").box(20.0, 10.0, 6.0)

        def entities(shapes):
            out = []
            for s in shapes:
                c = s.Center()
                if s.ShapeType() == "Face":
                    a = s.normalAt()
                elif s.ShapeType() == "Edge":
                    a = s.tangentAt()
                else:
                    a = None
                axis = (a.x, a.y, a.z) if a is not None else (0.0, 0.0, 0.0)
                out.append(Entity((c.x, c.y, c.z), axis, s.geomType()))
            return out

        for sel in (">Z", "<Z", "|Z", "#Z", "+Z", "-Z", ">X", "<Y",
                    "|Z and >Y", "not(<X or >X)", ">Z[0]", ">Z[-1]"):
            for kind in ("edges", "faces"):
                theirs = len(getattr(box, kind)(sel).vals())
                ours = len(select(sel, entities(getattr(box, kind)().vals())))
                self.assertEqual(ours, theirs,
                                 "selector %r on %s: our DSL says %d, CadQuery "
                                 "says %d" % (sel, kind, ours, theirs))


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestSteppedHoles(unittest.TestCase):
    """CAPABILITY: counterbore/countersink were refused ('not-yet-supported').
    CadQuery has both: Workplane.cboreHole(diameter, cboreDiameter, cboreDepth,
    depth=None) and Workplane.cskHole(diameter, cskDiameter, cskAngle,
    depth=None) (classreference.html); depth=None drills through-all."""

    def test_counterbore_cuts_the_exact_stepped_profile(self):
        b = _ref_plate()
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                                     diameter=6.0, through=True,
                                     kind="counterbore",
                                     cbore_diameter=12.0, cbore_depth=3.0)).ok)
        # the d=6 bore runs the full 8mm; the d=12 counterbore adds an ANNULUS
        # (12^2-6^2) over its 3mm depth -- it does not re-remove the bore.
        expected = (PLATE_VOLUME
                    - math.pi * 3.0 ** 2 * PLATE_T
                    - math.pi * (6.0 ** 2 - 3.0 ** 2) * 3.0)
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_counterbore_removes_more_than_the_plain_hole(self):
        plain = _ref_plate()
        plain.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                         diameter=6.0, through=True))
        cb = _ref_plate()
        cb.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0, diameter=6.0,
                      through=True, kind="counterbore",
                      cbore_diameter=12.0, cbore_depth=3.0))
        self.assertLess(cb.query("measure")["volume"],
                        plain.query("measure")["volume"])

    def test_countersink_is_a_real_cone_not_a_cylinder(self):
        """A countersink removes a truncated CONE, so it takes out strictly less
        than a counterbore of the same top diameter and equivalent depth."""
        ck = _ref_plate()
        self.assertTrue(ck.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                                      diameter=6.0, through=True,
                                      kind="countersink",
                                      csk_diameter=12.0, csk_angle=82.0)).ok)
        plain = _ref_plate()
        plain.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                         diameter=6.0, through=True))
        self.assertLess(ck.query("measure")["volume"],
                        plain.query("measure")["volume"])
        self.assertTrue(ck.query("validity")["is_valid"])
        # the cone flank is a real extra face, not a re-used cylinder wall
        self.assertGreater(ck.query("metrics")["faces"],
                           plain.query("metrics")["faces"])

    def test_degenerate_stepped_dimensions_are_rejected(self):
        for op in (Hole(face_or_sketch="solid", x=20.0, y=12.0, diameter=6.0,
                        through=True, kind="counterbore",
                        cbore_diameter=4.0, cbore_depth=3.0),      # cbore < bore
                   Hole(face_or_sketch="solid", x=20.0, y=12.0, diameter=6.0,
                        through=True, kind="countersink",
                        csk_diameter=12.0, csk_angle=200.0)):      # angle >= 180
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok)
            self.assertEqual(res.diagnostics[0].code, "bad-value")
            self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestLoftSweepDraft(unittest.TestCase):
    """CAPABILITY: loft / sweep / draft were all refused. CadQuery does all three."""

    def test_loft_between_two_offset_profiles(self):
        """Workplane.loft(ruled=False) / Solid.makeLoft (classreference.html).
        Loft.offsets moves each profile along its sketch-plane normal, which is
        what Workplane.workplane(offset=...) does."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=-10.0, y=-10.0, w=20.0, h=20.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=0.0, cy=0.0, r=5.0))
        self.assertTrue(b.apply(Loft(sketches=("sk1", "sk2"),
                                     offsets=(0.0, 10.0))).ok)
        m = b.query("measure")
        self.assertEqual([round(d, 6) for d in m["bbox"]], [20.0, 20.0, 10.0])
        # strictly between the two prisms it interpolates
        self.assertLess(m["volume"], 20.0 * 20.0 * 10.0)
        self.assertGreater(m["volume"], math.pi * 25.0 * 10.0)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_ruled_loft_differs_from_a_smooth_one(self):
        def build(ruled):
            b = CadQueryBackend()
            b.apply(NewSketch(plane="XY"))
            b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=10.0))
            b.apply(NewSketch(plane="XY"))
            b.apply(AddCircle(sketch="sk2", cx=0.0, cy=0.0, r=4.0))
            b.apply(NewSketch(plane="XY"))
            b.apply(AddCircle(sketch="sk3", cx=0.0, cy=0.0, r=9.0))
            b.apply(Loft(sketches=("sk1", "sk2", "sk3"), ruled=ruled,
                         offsets=(0.0, 10.0, 20.0)))
            return b.query("measure")["volume"]
        # ruled=True is a straight-sided (conical) blend; ruled=False is a smooth
        # spline through the sections, so the two enclose different volumes.
        self.assertNotAlmostEqual(build(True), build(False), places=3)

    def test_coplanar_loft_is_degenerate_not_a_phantom_body(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=-10.0, y=-10.0, w=20.0, h=20.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=0.0, cy=0.0, r=5.0))
        before = b.state_digest()
        res = b.apply(Loft(sketches=("sk1", "sk2")))   # no offsets -> coincident
        self.assertFalse(res.ok)
        self.assertFalse(b.query("summary")["solid_present"])
        self.assertEqual(b.state_digest(), before)

    def test_sweep_along_a_line_path_is_exact(self):
        """Workplane.sweep(path) (classreference.html). A r=2 circle swept 30mm
        along X is EXACTLY a cylinder: pi * 2^2 * 30."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="YZ"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=2.0))
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddLine(sketch="sk2", x1=0.0, y1=0.0, x2=30.0, y2=0.0))
        self.assertTrue(b.apply(Sweep(sketch="sk1", path="sk2")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               math.pi * 4.0 * 30.0, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_swept_body_is_a_boolean_operand(self):
        """A sweep pushes a real body, so Boolean(target=...) must be able to
        name it -- i.e. _solid_index knows 'sweep' is body-producing."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=-20.0, y=-20.0, w=40.0, h=40.0))
        b.apply(Extrude(sketch="sk1", distance=10.0))               # f1
        b.apply(NewSketch(plane="YZ"))
        b.apply(AddCircle(sketch="sk2", cx=0.0, cy=5.0, r=2.0))
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddLine(sketch="sk3", x1=-30.0, y1=0.0, x2=30.0, y2=0.0))
        self.assertTrue(b.apply(Sweep(sketch="sk2", path="sk3")).ok)  # f2
        res = b.apply(Boolean(kind="cut", target="f1", tool="f2"))
        self.assertTrue(res.ok, [d.message for d in res.diagnostics])
        self.assertLess(b.query("measure")["volume"], 40.0 * 40.0 * 10.0)

    def test_draft_tapers_the_side_walls_without_growing_the_part(self):
        """OCCT BRepOffsetAPI_DraftAngle. The neutral face ('<Z', the bottom)
        stays put and the side walls lean in, so the part loses volume but its
        bbox (set by the untouched neutral face) does not grow."""
        b = _ref_plate()
        base = b.query("measure")
        self.assertTrue(b.apply(Draft(faces=(), angle=5.0,
                                      neutral_plane="<Z")).ok)
        m = b.query("measure")
        self.assertLess(m["volume"], base["volume"])
        for got, want in zip(m["bbox"], base["bbox"]):
            self.assertLessEqual(round(got, 6), round(want, 6) + 1e-6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_draft_angle_is_monotonic(self):
        def vol(angle):
            b = _ref_plate()
            self.assertTrue(b.apply(Draft(faces=(), angle=angle,
                                          neutral_plane="<Z")).ok)
            return b.query("measure")["volume"]
        self.assertLess(vol(10.0), vol(5.0))    # more taper -> less material

    def test_bad_draft_angle_blocks(self):
        for angle in (0.0, 95.0):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(Draft(faces=(), angle=angle, neutral_plane="<Z"))
            self.assertFalse(res.ok)
            self.assertEqual(res.diagnostics[0].code, "bad-value")
            self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestPatternPitch(unittest.TestCase):
    """Patterns must place the right COUNT at the right PITCH."""

    def _unit(self):
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        return b

    def test_linear_pattern_count_and_pitch(self):
        """5 copies of a 10mm cube at a 25mm pitch: volume = 5 * 500 (they do not
        overlap) and the span is 4 * 25 + 10 = 110mm."""
        b = self._unit()
        self.assertTrue(b.apply(LinearPattern(count=5, spacing=25.0,
                                              direction=(1, 0, 0))).ok)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 5 * 10.0 * 10.0 * 5.0, places=6)
        self.assertAlmostEqual(m["bbox"][0], 4 * 25.0 + 10.0, places=6)
        self.assertEqual(b.query("metrics")["solids"], 5)

    def test_linear_pattern_pitch_is_independent_of_direction_magnitude(self):
        """REGRESSION: the direction was used RAW, so (2,0,0) doubled the pitch."""
        spans = []
        for d in ((1, 0, 0), (2, 0, 0), (7, 0, 0)):
            b = self._unit()
            self.assertTrue(b.apply(LinearPattern(count=3, spacing=25.0,
                                                  direction=d)).ok)
            spans.append(round(b.query("measure")["bbox"][0], 6))
        self.assertEqual(spans, [2 * 25.0 + 10.0] * 3)

    def test_circular_pattern_count_and_pitch(self):
        """count instances spanning `angle`, stepping angle/count -- the same
        formula the frep backend uses, so the two kernels agree.

        The block is held well off the axis (x in 40..50) so that 6 copies 60 deg
        apart are genuinely disjoint and the count is observable."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=40.0, y=-5.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=6, angle=360.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertEqual(b.query("metrics")["solids"], 6)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               6 * 10.0 * 10.0 * 5.0, places=6)
        # 60 deg pitch about Z: with copies at 0 and 180 deg the ring spans the
        # full 2 * 50mm in X, and a full-circle pattern puts its centroid back on
        # the axis. Both fail if the step is not angle/count.
        self.assertAlmostEqual(b.query("measure")["bbox"][0], 100.0, places=6)
        com = b.query("metrics")["center_of_mass"]
        self.assertAlmostEqual(com[0], 0.0, places=6)
        self.assertAlmostEqual(com[1], 0.0, places=6)

    def test_partial_arc_pattern_spans_the_arc_inclusively(self):
        """REGRESSION: we stepped angle/count unconditionally, so a 180-degree
        4-up pattern spanned only 135 degrees.

        CadQuery's Workplane.polarArray(fill=True) (cq.py, classreference.html)
        steps angle/count ONLY when the angle is a multiple of 360; otherwise it
        steps angle/(count-1) so the arc is spanned INCLUSIVELY, start and end.
        180 deg with count=4 therefore steps 60 deg: copies at 0, 60, 120, 180."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=40.0, y=-5.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=4, angle=180.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertEqual(b.query("metrics")["solids"], 4)
        # the last copy is at a FULL 180 deg, i.e. mirrored onto -X: the pattern
        # spans the whole half-turn, so X runs from -50 to +50. With the old
        # angle/count step (45 deg) the last copy sat at 135 deg and the -X
        # extent would only have reached -cos(45)*50 ~= -35.
        bbox = b.query("measure")["bbox"]
        self.assertAlmostEqual(bbox[0], 100.0, places=6)
        # and the arc is symmetric about +Y, so the centroid sits on the Y axis
        self.assertAlmostEqual(b.query("metrics")["center_of_mass"][0], 0.0,
                               places=6)

    def test_full_circle_pattern_still_divides_by_count(self):
        """The other half of the rule: a multiple of 360 divides by count, not
        count-1 (otherwise the last copy would land on top of the first)."""
        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=40.0, y=-5.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=4, angle=360.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        # 4 copies 90 deg apart, all disjoint. With angle/(count-1) = 120 deg the
        # 4th copy would coincide with the 1st and we would see only 3 solids.
        self.assertEqual(b.query("metrics")["solids"], 4)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               4 * 10.0 * 10.0 * 5.0, places=6)


@unittest.skipUnless(HAVE_CQ, "cadquery/OCCT not installed")
class TestTessellationAndSchema(unittest.TestCase):
    """importexport.html: exporters.export(w, fname, exportType, tolerance=0.1,
    angularTolerance=0.1, opt=None). `tolerance` is the LINEAR deflection in
    model units, `angularTolerance` is in RADIANS. We were passing NEITHER, so
    every mesh check silently ran at CadQuery's coarse 0.1mm/0.1rad default."""

    def _facets(self, stl: str) -> int:
        return stl.count("facet normal")

    def _drilled(self):
        b = _ref_plate()
        b.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                     diameter=HOLE_D, through=True))
        return b

    def test_linear_deflection_controls_mesh_density(self):
        """The linear deflection only BINDS when the angular one is loose (both
        are upper bounds and OCCT honours the tighter). Hold angular at 2.0 rad
        and the linear tolerance drives the facet count monotonically."""
        b = self._drilled()
        counts = [self._facets(b.export("stl", tolerance=t, angular_tolerance=2.0))
                  for t in (0.5, 0.05, 0.005, 0.0005)]
        self.assertEqual(counts, sorted(counts))
        self.assertLess(counts[0], counts[-1])
        self.assertEqual(len(set(counts)), len(counts))  # every step really moved

    def test_repeated_export_is_not_served_from_the_cached_mesh(self):
        """REGRESSION: OCCT caches the triangulation on the TopoDS_Shape, so the
        SECOND export of a shape silently reused the FIRST one's mesh and ignored
        the new tolerance (every tolerance from 0.0005 to 0.5 returned the same
        992 facets). The exporter must BRepTools.Clean_s first."""
        b = self._drilled()
        first_fine = self._facets(b.export("stl", tolerance=0.0005,
                                           angular_tolerance=2.0))
        then_coarse = self._facets(b.export("stl", tolerance=0.5,
                                            angular_tolerance=2.0))
        self.assertLess(then_coarse, first_fine)
        # and going back to fine reproduces the first result exactly
        again = self._facets(b.export("stl", tolerance=0.0005,
                                      angular_tolerance=2.0))
        self.assertEqual(again, first_fine)

    def test_default_deflection_is_pinned_not_cadquerys_coarse_default(self):
        """exporters.export defaults tolerance=0.1; Shape.exportStl (which it
        calls) defaults 1e-3. We pin 1e-3, and we ALSO tighten the angular bound,
        because on curved faces the angular one is what binds: at 0.1 rad the mesh
        is identical whether the linear tolerance is 0.1 or 0.01."""
        self.assertEqual(CadQueryBackend.LINEAR_DEFLECTION, 1e-3)
        self.assertLess(CadQueryBackend.LINEAR_DEFLECTION, 0.1)   # cq export()'s
        self.assertLess(CadQueryBackend.ANGULAR_DEFLECTION, 0.1)  # cq's
        b = self._drilled()
        # the pinned defaults must match an explicit request for the same values
        self.assertEqual(
            self._facets(b.export("stl")),
            self._facets(b.export("stl", tolerance=1e-3, angular_tolerance=0.05)))
        # and they must be strictly finer than what cq's export() would have given
        self.assertGreater(
            self._facets(b.export("stl")),
            self._facets(b.export("stl", tolerance=0.1, angular_tolerance=0.1)))

    def test_angular_deflection_also_controls_density(self):
        b = self._drilled()
        coarse = self._facets(b.export("stl", tolerance=0.5,
                                       angular_tolerance=1.0))
        fine = self._facets(b.export("stl", tolerance=0.5,
                                     angular_tolerance=0.05))
        self.assertLess(coarse, fine)

    def test_bad_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            _ref_plate().export("stl", tolerance=0.0)

    @staticmethod
    def _mesh_volume(stl: str) -> float:
        """Signed volume of an ASCII STL via the divergence theorem (sum of
        tetrahedra from the origin over each triangle)."""
        verts = []
        for line in stl.splitlines():
            line = line.strip()
            if line.startswith("vertex "):
                verts.append([float(v) for v in line.split()[1:4]])
        total = 0.0
        for i in range(0, len(verts), 3):
            a, b, c = verts[i], verts[i + 1], verts[i + 2]
            total += (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            ) / 6.0
        return abs(total)

    def test_mesh_volume_converges_to_the_analytic_volume(self):
        """THE POINT of fixing the deflection. The tessellated STL is what every
        mesh-based verifier and the differential oracle actually measure, so its
        volume must converge on the analytic truth as the deflection tightens.

        A 40x24x8 plate with one 6mm through-hole: the mesh CHORDS the cylinder,
        so it always over-reports volume (the facets cut the bore corners off);
        the error must shrink monotonically toward 0."""
        b = _ref_plate()
        b.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                     diameter=HOLE_D, through=True))
        analytic = PLATE_VOLUME - math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T
        self.assertAlmostEqual(b.query("measure")["volume"], analytic, places=6)

        errors = []
        for tol in (0.5, 0.05, 0.005):
            v = self._mesh_volume(b.export("stl", tolerance=tol,
                                           angular_tolerance=2.0))
            errors.append(abs(v - analytic))
        self.assertEqual(errors, sorted(errors, reverse=True))  # monotone
        self.assertLess(errors[-1], errors[0] / 10.0)

        # at our PINNED default the mesh is already within 0.05% of analytic --
        # at CadQuery's export() default (0.1mm / 0.1rad) it was ~10x worse.
        pinned = abs(self._mesh_volume(b.export("stl")) - analytic)
        coarse = abs(self._mesh_volume(
            b.export("stl", tolerance=0.1, angular_tolerance=0.1)) - analytic)
        self.assertLess(pinned / analytic, 5e-4)
        self.assertLess(pinned, coarse)

    def test_step_declares_the_configured_schema(self):
        """The STEP application protocol is a GLOBAL OCCT Interface_Static
        setting, so we set it on every export instead of inheriting whatever the
        process is in. AP214 must appear in the FILE_SCHEMA header."""
        step = _ref_plate().export("step")
        self.assertIn("ISO-10303", step)
        self.assertIn("FILE_SCHEMA", step)
        self.assertIn("AUTOMOTIVE_DESIGN", step)   # the AP214 schema name
        self.assertEqual(CadQueryBackend.STEP_SCHEMA, "AP214IS")

    def test_step_round_trip_is_lossless_after_a_shell(self):
        """A shelled part is the hard case (inner + outer surfaces): the STEP we
        write must re-import with the identical volume AND the identical bbox."""
        import os
        import tempfile
        import cadquery as cq

        b = CadQueryBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0))
        b.apply(Extrude(sketch="sk1", distance=20.0))
        self.assertTrue(b.apply(Shell(faces=(), thickness=3.0)).ok)
        step = b.export("step")

        fd, path = tempfile.mkstemp(suffix=".step")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(step)
            shape = cq.importers.importStep(path).val()
        finally:
            os.remove(path)
        bb = shape.BoundingBox()
        self.assertAlmostEqual(shape.Volume(), b.query("measure")["volume"],
                               places=6)
        self.assertEqual([round(bb.xlen, 6), round(bb.ylen, 6), round(bb.zlen, 6)],
                         [60.0, 40.0, 20.0])


if __name__ == "__main__":
    unittest.main()
