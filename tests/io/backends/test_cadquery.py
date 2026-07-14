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
    NewSketch, AddRectangle, AddCircle, Extrude, Fillet, Boolean,
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
        b = _ref_plate()
        self.assertTrue(b.apply(Shell(faces=(), thickness=1.0)).ok)
        expected = PLATE_VOLUME - (PLATE_W - 2) * (PLATE_H - 2) * (PLATE_T - 1)
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
    def test_unsupported_ops_never_silently_no_op(self):
        for op in (Draft(faces=(), angle=5.0),
                   Loft(sketches=("sk1", "sk2")),
                   Sweep(sketch="sk1", path="sk2")):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok, "%s must not silently no-op" % type(op).__name__)
            self.assertTrue(res.diagnostics)
            self.assertEqual(res.diagnostics[0].severity, Severity.ERROR)
            self.assertEqual(b.state_digest(), before)

    def test_counterbore_hole_is_refused_not_faked(self):
        b = _ref_plate()
        res = b.apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                           diameter=HOLE_D, through=True, kind="counterbore"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "not-yet-supported")


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


if __name__ == "__main__":
    unittest.main()
