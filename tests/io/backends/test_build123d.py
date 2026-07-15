"""Tests for the real-geometry Build123dBackend (OCCT via build123d/OCP).

These build actual B-rep solids, so they require build123d. When build123d is
not importable the whole suite is skipped (the backend module itself still
imports fine — build123d is imported lazily inside its methods).

The headline claims, asserted here deterministically:

* a box has its analytic volume;
* a fillet on the named VERTICAL edges gives a DIFFERENT volume (and face count)
  than a fillet on ALL edges — the field-liveness bug that hid in every backend
  until op.edges was honoured;
* build123d AGREES with the CadQuery backend to ~1e-9 on every shared op (both
  are OCCT B-rep) — the cross-check that the two front-ends mean the same thing.
"""

import math
import unittest

from harnesscad.io.backends.build123d import Build123dBackend
from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle, AddCircle, AddLine, Extrude, Fillet, Boolean,
    Revolve, Chamfer, Hole, Shell, LinearPattern, CircularPattern, Mirror,
    Draft, Loft, Sweep, SetParam, Constrain,
)
from harnesscad.eval.verifiers.verify import Severity


def _build123d_available() -> bool:
    try:
        import build123d  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _cadquery_available() -> bool:
    try:
        import cadquery  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


HAVE_BD = _build123d_available()
HAVE_CQ = _cadquery_available()

PLATE_W, PLATE_H, PLATE_T = 40.0, 24.0, 8.0
HOLE_D = 6.0
PLATE_VOLUME = PLATE_W * PLATE_H * PLATE_T
ANALYTIC_2HOLE = PLATE_VOLUME - 2.0 * math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T


def _ref_plate() -> Build123dBackend:
    b = Build123dBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H)).ok
    assert b.apply(Extrude(sketch="sk1", distance=PLATE_T)).ok
    return b


def _build_plate(w=20.0, h=10.0, t=5.0) -> Build123dBackend:
    b = Build123dBackend()
    assert b.apply(NewSketch(plane="XY")).ok
    assert b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h)).ok
    assert b.apply(Extrude(sketch="sk1", distance=t)).ok
    return b


# The backend module must import even without build123d installed.
class TestModuleImportsWithoutKernel(unittest.TestCase):
    def test_backend_constructs_without_kernel(self):
        b = Build123dBackend()
        self.assertEqual(b.query("summary")["feature_count"], 0)
        self.assertFalse(b.query("validity")["solid_present"])


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestRealPlate(unittest.TestCase):
    def test_plate_is_valid_solid(self):
        b = _build_plate()
        summary = b.query("summary")
        self.assertTrue(summary["solid_present"])
        self.assertEqual(summary["feature_count"], 1)
        v = b.query("validity")
        self.assertTrue(v["manifold"])
        self.assertTrue(v["watertight"])
        self.assertTrue(v["is_valid"])

    def test_measure_matches_nominal(self):
        b = _build_plate(w=20.0, h=10.0, t=5.0)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 20.0 * 10.0 * 5.0, places=6)
        self.assertEqual([round(d, 3) for d in m["bbox"]], [20.0, 10.0, 5.0])

    def test_export_step_is_real(self):
        step = _build_plate().export("step")
        self.assertTrue(step)
        self.assertIn("ISO-10303", step)

    def test_deterministic_digest(self):
        self.assertEqual(_build_plate().state_digest(), _build_plate().state_digest())

    def test_different_geometry_differs_in_digest(self):
        self.assertNotEqual(_build_plate(w=20.0).state_digest(),
                            _build_plate(w=30.0).state_digest())


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestBlockAndCorrect(unittest.TestCase):
    def test_bad_reference_does_not_mutate(self):
        b = Build123dBackend()
        before = b.state_digest()
        res = b.apply(Extrude(sketch="nope", distance=5.0))
        self.assertFalse(res.ok)
        self.assertTrue(res.diagnostics)
        self.assertEqual(res.diagnostics[0].severity, Severity.ERROR)
        self.assertEqual(b.state_digest(), before)

    def test_oversized_fillet_blocks_without_mutating(self):
        b = _build_plate()
        before = b.state_digest()
        res = b.apply(Fillet(edges=(), radius=999.0))
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)

    def test_bad_circle_radius_blocks(self):
        b = Build123dBackend()
        b.apply(NewSketch())
        self.assertFalse(b.apply(AddCircle(sketch="sk1", r=-1.0)).ok)

    def test_boolean_requires_two_solids(self):
        self.assertFalse(_build_plate().apply(Boolean(kind="cut")).ok)

    def test_unknown_sketch_plane_is_rejected_at_sketch_time(self):
        res = Build123dBackend().apply(NewSketch(plane="BOGUS"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestExactGeometry(unittest.TestCase):
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
        b = Build123dBackend()
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
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=PLATE_W, h=PLATE_H))
        b.apply(Extrude(sketch="sk1", distance=PLATE_T))              # f1
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=10.0, cy=12.0, r=HOLE_D / 2.0))
        b.apply(Extrude(sketch="sk2", distance=PLATE_T))              # f2
        self.assertTrue(b.apply(Boolean(kind="cut", target="f1", tool="f2")).ok)
        expected = PLATE_VOLUME - math.pi * (HOLE_D / 2.0) ** 2 * PLATE_T
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_boolean_with_unknown_tool_is_blocked(self):
        b = Build123dBackend()
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
        b = Build123dBackend()
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddRectangle(sketch="sk1", x=10.0, y=0.0, w=5.0, h=20.0))
        self.assertTrue(b.apply(Revolve(sketch="sk1", angle=360.0,
                                        axis=(0, 0, 0, 0, 1, 0))).ok)
        expected = math.pi * (15.0 ** 2 - 10.0 ** 2) * 20.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_zero_volume_revolve_is_rejected(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddRectangle(sketch="sk1", x=10.0, y=0.0, w=5.0, h=20.0))
        before = b.state_digest()
        res = b.apply(Revolve(sketch="sk1", angle=360.0, axis=(0, 0, 0, 0, 0, 1)))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "degenerate")
        self.assertFalse(b.query("summary")["solid_present"])
        self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestEdgeSelection(unittest.TestCase):
    """op.edges is a tuple of CadQuery selector strings — honoured, never dropped.
    A fillet on the wrong edge set is a silent correctness bug."""

    def _box(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        return b

    def test_fillet_hits_only_the_selected_edges(self):
        """A 20x10x5 box has 12 edges, 4 vertical ('|Z'). Rounding only those
        gives 10 faces / V=995.708; rounding all 12 gives 26 faces / V=971.295.
        Two different parts — so the selector MUST be honoured (the exact numbers
        the CadQuery backend asserts, proving the two kernels agree)."""
        v = self._box()
        self.assertTrue(v.apply(Fillet(edges=("|Z",), radius=1.0)).ok)
        vm = v.query("metrics")
        self.assertEqual(vm["faces"], 10)
        self.assertAlmostEqual(vm["volume"], 995.7079632679, places=6)

        a = self._box()
        self.assertTrue(a.apply(Fillet(edges=(), radius=1.0)).ok)
        am = a.query("metrics")
        self.assertEqual(am["faces"], 26)
        self.assertLess(am["volume"], vm["volume"])
        self.assertTrue(v.query("validity")["is_valid"])

    def test_fillet_on_named_edges_differs_from_all_edges(self):
        """The field-liveness headline: named vertical edges != all edges."""
        named = self._box()
        named.apply(Fillet(edges=("|Z",), radius=1.0))
        alle = self._box()
        alle.apply(Fillet(edges=(), radius=1.0))
        self.assertNotAlmostEqual(named.query("measure")["volume"],
                                  alle.query("measure")["volume"], places=3)

    def test_fillet_top_face_edges_only(self):
        b = self._box()
        self.assertTrue(b.apply(Fillet(edges=(">Z",), radius=1.0)).ok)
        m = b.query("metrics")
        self.assertEqual(m["faces"], 10)
        self.assertEqual([round(d, 6) for d in m["bbox"]], [20.0, 10.0, 5.0])

    def test_selector_tuple_is_the_union_of_its_members(self):
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
        expected = 20.0 * 10.0 * 5.0 - 4.0 * 0.5 * 1.0 * 1.0 * 5.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_asymmetric_chamfer_uses_length2(self):
        b = self._box()
        self.assertTrue(b.apply(Chamfer(edges=("|Z",), distance=1.0,
                                        distance2=2.0)).ok)
        expected = 20.0 * 10.0 * 5.0 - 4.0 * 0.5 * 1.0 * 2.0 * 5.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)

    def test_malformed_selector_is_a_typed_bad_value(self):
        b = self._box()
        before = b.state_digest()
        res = b.apply(Fillet(edges=("|Q",), radius=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")
        self.assertEqual(b.state_digest(), before)

    def test_selector_that_matches_nothing_blocks(self):
        b = self._box()
        before = b.state_digest()
        res = b.apply(Fillet(edges=("%BSPLINE",), radius=1.0))
        self.assertFalse(res.ok)
        self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestShellSemantics(unittest.TestCase):
    def _box(self, w, h, t):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=w, h=h))
        b.apply(Extrude(sketch="sk1", distance=t))
        return b

    def test_shell_does_not_grow_the_part(self):
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(), thickness=3.0)).ok)
        self.assertEqual([round(d, 6) for d in b.query("measure")["bbox"]],
                         [60.0, 40.0, 20.0])

    def test_empty_faces_is_a_sealed_closed_hollow_of_the_exact_volume(self):
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(), thickness=3.0)).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 22296.0, places=6)
        self.assertTrue(b.query("validity")["is_valid"])
        self.assertEqual(b.query("metrics")["solids"], 1)

    def test_open_shell_volume_is_the_analytic_hollow(self):
        b = self._box(60.0, 40.0, 20.0)
        self.assertTrue(b.apply(Shell(faces=(">Z",), thickness=3.0)).ok)
        expected = 60.0 * 40.0 * 20.0 - 54.0 * 34.0 * 17.0
        self.assertAlmostEqual(b.query("measure")["volume"], expected, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_sealed_holds_more_material_than_an_opened_one(self):
        sealed = self._box(60.0, 40.0, 20.0)
        sealed.apply(Shell(faces=(), thickness=3.0))
        opened = self._box(60.0, 40.0, 20.0)
        opened.apply(Shell(faces=(">Z",), thickness=3.0))
        self.assertGreater(sealed.query("measure")["volume"],
                           opened.query("measure")["volume"])

    def test_shell_honours_the_named_open_face(self):
        top = self._box(60.0, 40.0, 20.0)
        top.apply(Shell(faces=(">Z",), thickness=3.0))
        bot = self._box(60.0, 40.0, 20.0)
        self.assertTrue(bot.apply(Shell(faces=("<Z",), thickness=3.0)).ok)
        self.assertAlmostEqual(top.query("measure")["volume"],
                               bot.query("measure")["volume"], places=6)
        self.assertLess(top.query("metrics")["center_of_mass"][2], 10.0)
        self.assertGreater(bot.query("metrics")["center_of_mass"][2], 10.0)

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


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestSteppedHoles(unittest.TestCase):
    def test_counterbore_cuts_the_exact_stepped_profile(self):
        b = _ref_plate()
        self.assertTrue(b.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                                     diameter=6.0, through=True,
                                     kind="counterbore",
                                     cbore_diameter=12.0, cbore_depth=3.0)).ok)
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
        self.assertGreater(ck.query("metrics")["faces"],
                           plain.query("metrics")["faces"])

    def test_degenerate_stepped_dimensions_are_rejected(self):
        for op in (Hole(face_or_sketch="solid", x=20.0, y=12.0, diameter=6.0,
                        through=True, kind="counterbore",
                        cbore_diameter=4.0, cbore_depth=3.0),
                   Hole(face_or_sketch="solid", x=20.0, y=12.0, diameter=6.0,
                        through=True, kind="countersink",
                        csk_diameter=12.0, csk_angle=200.0)):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok)
            self.assertEqual(res.diagnostics[0].code, "bad-value")
            self.assertEqual(b.state_digest(), before)

    def test_unknown_hole_kind_is_rejected(self):
        res = _ref_plate().apply(Hole(face_or_sketch="solid", x=10.0, y=12.0,
                                      diameter=HOLE_D, through=True, kind="tapped"))
        self.assertFalse(res.ok)
        self.assertEqual(res.diagnostics[0].code, "bad-value")


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestLoftSweepDraft(unittest.TestCase):
    def test_loft_between_two_offset_profiles(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=-10.0, y=-10.0, w=20.0, h=20.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=0.0, cy=0.0, r=5.0))
        self.assertTrue(b.apply(Loft(sketches=("sk1", "sk2"),
                                     offsets=(0.0, 10.0))).ok)
        m = b.query("measure")
        self.assertEqual([round(d, 6) for d in m["bbox"]], [20.0, 20.0, 10.0])
        self.assertLess(m["volume"], 20.0 * 20.0 * 10.0)
        self.assertGreater(m["volume"], math.pi * 25.0 * 10.0)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_coplanar_loft_is_degenerate_not_a_phantom_body(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=-10.0, y=-10.0, w=20.0, h=20.0))
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk2", cx=0.0, cy=0.0, r=5.0))
        before = b.state_digest()
        res = b.apply(Loft(sketches=("sk1", "sk2")))
        self.assertFalse(res.ok)
        self.assertFalse(b.query("summary")["solid_present"])
        self.assertEqual(b.state_digest(), before)

    def test_sweep_along_a_line_path_is_exact(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="YZ"))
        b.apply(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=2.0))
        b.apply(NewSketch(plane="XZ"))
        b.apply(AddLine(sketch="sk2", x1=0.0, y1=0.0, x2=30.0, y2=0.0))
        self.assertTrue(b.apply(Sweep(sketch="sk1", path="sk2")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"],
                               math.pi * 4.0 * 30.0, places=6)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_bad_refs_in_loft_and_sweep_never_silently_no_op(self):
        for op in (Loft(sketches=("sk1", "nope")),
                   Sweep(sketch="sk1", path="nope")):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(op)
            self.assertFalse(res.ok)
            self.assertEqual(res.diagnostics[0].code, "bad-ref")
            self.assertEqual(b.state_digest(), before)

    def test_draft_tapers_side_walls_without_growing_the_part(self):
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
        self.assertLess(vol(10.0), vol(5.0))

    def test_bad_draft_angle_blocks(self):
        for angle in (0.0, 95.0):
            b = _ref_plate()
            before = b.state_digest()
            res = b.apply(Draft(faces=(), angle=angle, neutral_plane="<Z"))
            self.assertFalse(res.ok)
            self.assertEqual(res.diagnostics[0].code, "bad-value")
            self.assertEqual(b.state_digest(), before)


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestPatterns(unittest.TestCase):
    def _unit(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        return b

    def test_linear_pattern_count_and_pitch(self):
        b = self._unit()
        self.assertTrue(b.apply(LinearPattern(count=5, spacing=25.0,
                                              direction=(1, 0, 0))).ok)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 5 * 500.0, places=6)
        self.assertAlmostEqual(m["bbox"][0], 4 * 25.0 + 10.0, places=6)
        self.assertEqual(b.query("metrics")["solids"], 5)

    def test_linear_pattern_normalises_its_direction(self):
        b = self._unit()
        self.assertTrue(b.apply(LinearPattern(count=2, spacing=20.0,
                                              direction=(2, 0, 0))).ok)
        m = b.query("measure")
        self.assertAlmostEqual(m["volume"], 1000.0, places=6)
        self.assertAlmostEqual(m["bbox"][0], 30.0, places=6)

    def test_circular_pattern_count_and_pitch(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=40.0, y=-5.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=6, angle=360.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertEqual(b.query("metrics")["solids"], 6)
        self.assertAlmostEqual(b.query("measure")["volume"], 6 * 500.0, places=6)
        self.assertAlmostEqual(b.query("measure")["bbox"][0], 100.0, places=6)

    def test_partial_arc_pattern_spans_the_arc_inclusively(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=40.0, y=-5.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(CircularPattern(count=4, angle=180.0,
                                                axis=(0, 0, 0, 0, 0, 1))).ok)
        self.assertEqual(b.query("metrics")["solids"], 4)
        self.assertAlmostEqual(b.query("measure")["bbox"][0], 100.0, places=6)

    def test_mirror_is_exact(self):
        b = Build123dBackend()
        b.apply(NewSketch(plane="XY"))
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        b.apply(Extrude(sketch="sk1", distance=5.0))
        self.assertTrue(b.apply(Mirror(plane="YZ")).ok)
        self.assertAlmostEqual(b.query("measure")["volume"], 1000.0, places=6)


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestDofTracking(unittest.TestCase):
    def test_constraints_reduce_nominal_dof(self):
        b = Build123dBackend()
        b.apply(NewSketch())
        b.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=10.0, h=10.0))
        self.assertEqual(b.query("sketch_dof")["sk1"], 4)
        b.apply(Constrain(kind="distance", a="e1", value=10.0))
        self.assertEqual(b.query("sketch_dof")["sk1"], 3)


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
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


@unittest.skipUnless(HAVE_BD, "build123d/OCP not installed")
class TestExport(unittest.TestCase):
    def test_step_is_iso10303(self):
        b = _ref_plate()
        b.apply(Hole(face_or_sketch="solid", x=20.0, y=12.0,
                     diameter=HOLE_D, through=True))
        step = b.export("step")
        self.assertIn("ISO-10303", step)

    def test_stl_export_is_parseable_ascii(self):
        stl = _ref_plate().export("stl")
        self.assertTrue(stl.lstrip().startswith("solid"))
        self.assertIn("facet normal", stl)

    def test_brep_export_is_the_native_kernel_format(self):
        self.assertIn("CASCADE", _ref_plate().export("brep"))

    def test_unsupported_format_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            _ref_plate().export("dwg")

    def test_export_with_no_solid_is_a_clear_error(self):
        with self.assertRaises(ValueError):
            Build123dBackend().export("step")


@unittest.skipUnless(HAVE_BD and HAVE_CQ, "build123d and cadquery both required")
class TestAgreesWithCadQuery(unittest.TestCase):
    """Both backends are OCCT B-rep, so the SAME op stream must produce the SAME
    volume to ~1e-9. This is the cross-check the whole second front-end exists for.
    """

    def _cq(self):
        from harnesscad.io.backends.cadquery import CadQueryBackend
        return CadQueryBackend

    def _both(self, ops):
        from harnesscad.io.backends.cadquery import CadQueryBackend
        vols = []
        for cls in (Build123dBackend, CadQueryBackend):
            b = cls()
            for op in ops:
                res = b.apply(op)
                self.assertTrue(res.ok, "%s rejected %s: %s"
                                % (cls.__name__, type(op).__name__,
                                   [d.message for d in res.diagnostics]))
            vols.append(b.query("measure")["volume"])
        return vols

    def test_shared_ops_agree_to_machine_precision(self):
        streams = {
            "box": [NewSketch(plane="XY"),
                    AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
                    Extrude(sketch="sk1", distance=5)],
            "two_holes": [NewSketch(plane="XY"),
                          AddRectangle(sketch="sk1", x=0, y=0, w=40, h=24),
                          Extrude(sketch="sk1", distance=8),
                          Hole(face_or_sketch="solid", x=10, y=12,
                               diameter=6, through=True),
                          Hole(face_or_sketch="solid", x=30, y=12,
                               diameter=6, through=True)],
            "revolve": [NewSketch(plane="XZ"),
                        AddRectangle(sketch="sk1", x=10, y=0, w=5, h=20),
                        Revolve(sketch="sk1", angle=360, axis=(0, 0, 0, 0, 1, 0))],
            "fillet_vz": [NewSketch(plane="XY"),
                          AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
                          Extrude(sketch="sk1", distance=5),
                          Fillet(edges=("|Z",), radius=1.0)],
            "chamfer_vz": [NewSketch(plane="XY"),
                           AddRectangle(sketch="sk1", x=0, y=0, w=20, h=10),
                           Extrude(sketch="sk1", distance=5),
                           Chamfer(edges=("|Z",), distance=1.0)],
            "shell_sealed": [NewSketch(plane="XY"),
                             AddRectangle(sketch="sk1", x=0, y=0, w=60, h=40),
                             Extrude(sketch="sk1", distance=20),
                             Shell(faces=(), thickness=3.0)],
            "counterbore": [NewSketch(plane="XY"),
                            AddRectangle(sketch="sk1", x=0, y=0, w=40, h=24),
                            Extrude(sketch="sk1", distance=8),
                            Hole(face_or_sketch="solid", x=20, y=12, diameter=6,
                                 through=True, kind="counterbore",
                                 cbore_diameter=12, cbore_depth=3)],
            "countersink": [NewSketch(plane="XY"),
                            AddRectangle(sketch="sk1", x=0, y=0, w=40, h=24),
                            Extrude(sketch="sk1", distance=8),
                            Hole(face_or_sketch="solid", x=20, y=12, diameter=6,
                                 through=True, kind="countersink",
                                 csk_diameter=12, csk_angle=82)],
            "loft": [NewSketch(plane="XY"),
                     AddRectangle(sketch="sk1", x=-10, y=-10, w=20, h=20),
                     NewSketch(plane="XY"),
                     AddCircle(sketch="sk2", cx=0, cy=0, r=5),
                     Loft(sketches=("sk1", "sk2"), offsets=(0, 10))],
            "sweep": [NewSketch(plane="YZ"),
                      AddCircle(sketch="sk1", cx=0, cy=0, r=2),
                      NewSketch(plane="XZ"),
                      AddLine(sketch="sk2", x1=0, y1=0, x2=30, y2=0),
                      Sweep(sketch="sk1", path="sk2")],
            "linpat": [NewSketch(plane="XY"),
                       AddRectangle(sketch="sk1", x=0, y=0, w=10, h=10),
                       Extrude(sketch="sk1", distance=5),
                       LinearPattern(count=3, spacing=20, direction=(1, 0, 0))],
            "draft": [NewSketch(plane="XY"),
                      AddRectangle(sketch="sk1", x=0, y=0, w=40, h=24),
                      Extrude(sketch="sk1", distance=8),
                      Draft(faces=(), angle=5.0, neutral_plane="<Z")],
        }
        for name, ops in streams.items():
            vbd, vcq = self._both(ops)
            self.assertAlmostEqual(
                vbd, vcq, places=6,
                msg="%s: build123d=%.9f cadquery=%.9f differ by %.2e"
                    % (name, vbd, vcq, abs(vbd - vcq)))
            # tighter: relative agreement to ~1e-9 (both OCCT)
            self.assertLess(abs(vbd - vcq) / max(abs(vcq), 1e-9), 1e-9,
                            "%s not within 1e-9 relative" % name)


if __name__ == "__main__":
    unittest.main()
