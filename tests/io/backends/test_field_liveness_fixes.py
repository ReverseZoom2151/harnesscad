"""Regression tests for the 52 DEAD op fields.

A DEAD field is one an op DECLARES, a backend ACCEPTS, and the geometry never sees:
the part comes out well-formed, watertight, manifold -- and wrong, with no
diagnostic. :mod:`harnesscad.eval.selftest.field_liveness` is the oracle that
counts them; this file pins the individual fixes, so a regression names the field
it broke rather than moving a number in a matrix.

Each test here FAILS on the code as it was and PASSES now. Where a field is
genuinely inexpressible on an engine, the test asserts the typed REFUSAL instead --
a refusal is a feature, a wrong part is not. And where a refusal would turn away
input that is legal and in use, the test asserts it is ACCEPTED, because a
validator that rejects legal input is just a false positive with a typed error
message.

The one op-state model, four engines. ``freecad``, ``openscad`` and ``blender`` do
not re-implement the CISP semantics -- they COMPOSE an
:class:`~harnesscad.io.backends.frep.FRepBackend` and lower its CSG tree
(:mod:`harnesscad.io.backends.external`). That is precisely why the same fields
died on four engines at once, why the differential oracle could never see it (all
four agreed, and all four were wrong), and why most of these tests drive frep: a
field the tree drops is a field four engines drop.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.core.cisp.ops import (AddCircle, AddInstance, AddRectangle,
                                      Chamfer, CircularPattern, Constrain,
                                      Extrude, Fillet, Hole, LinearPattern,
                                      Mate, Mirror, NewSketch, Shell, Thicken,
                                      thicken_delta)
from harnesscad.io.backends.frep import (FRepBackend, MIN_WALL_CELLS,
                                         countersink_depth)


def build(ops, backend=None, **kw):
    """Apply ops to a backend. Returns (backend, None) or (backend, [codes])."""
    b = backend if backend is not None else FRepBackend(**kw)
    for op in ops:
        result = b.apply(op)
        if not result.ok:
            return b, [d.code for d in result.diagnostics]
    return b, None


BOX = [NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0),
       Extrude("sk1", 20.0)]

#: Two features on ONE body: the pad (f1) and a fillet on it (f2). A pattern that
#: names f1 must replicate the SHARP pad; one that names f2 must replicate the
#: ROUNDED one. Until now both replicated "the last solid", whatever was named.
TWO_FEATURES = [NewSketch("XY"), AddRectangle("sk1", 100.0, 0.0, 20.0, 20.0),
                Extrude("sk1", 10.0), Fillet((), 2.0)]


def volume(b):
    return b.query("metrics")["volume"]


class HoleKindTest(unittest.TestCase):
    """A counterbore, a countersink and a plain hole were ONE CYLINDER.

    Four of six engines validated ``kind`` / ``cbore_*`` / ``csk_*``, recorded them
    on the feature, and then cut the same bare cylinder for all three. Three
    distinct manufacturing intents, one geometry, no diagnostic.
    """

    def _hole(self, **kw):
        params = dict(face_or_sketch="", x=0.0, y=0.0, diameter=8.0, depth=None,
                      through=True, kind="simple")
        params.update(kw)
        b, err = build(BOX + [Hole(**params)])
        self.assertIsNone(err, "hole was refused: %s" % err)
        return volume(b)

    def test_the_three_kinds_are_three_different_solids(self):
        simple = self._hole(kind="simple")
        cbore = self._hole(kind="counterbore", cbore_diameter=16.0, cbore_depth=5.0)
        csk = self._hole(kind="countersink", csk_diameter=16.0, csk_angle=82.0)

        # each stepped hole removes MORE material than the plain bore it contains
        self.assertLess(cbore, simple)
        self.assertLess(csk, simple)
        # ...and they are not each other
        self.assertNotAlmostEqual(cbore, csk, delta=1.0)

    def test_cbore_diameter_and_depth_reach_the_geometry(self):
        base = self._hole(kind="counterbore", cbore_diameter=16.0, cbore_depth=5.0)
        wider = self._hole(kind="counterbore", cbore_diameter=24.0, cbore_depth=5.0)
        deeper = self._hole(kind="counterbore", cbore_diameter=16.0, cbore_depth=10.0)
        self.assertLess(wider, base, "cbore_diameter did not reach the kernel")
        self.assertLess(deeper, base, "cbore_depth did not reach the kernel")

    def test_csk_diameter_and_angle_reach_the_geometry(self):
        base = self._hole(kind="countersink", csk_diameter=16.0, csk_angle=82.0)
        wider = self._hole(kind="countersink", csk_diameter=24.0, csk_angle=82.0)
        blunter = self._hole(kind="countersink", csk_diameter=16.0, csk_angle=120.0)
        self.assertLess(wider, base, "csk_diameter did not reach the kernel")
        # a blunter included angle is a SHALLOWER cone: less material removed
        self.assertGreater(blunter, base, "csk_angle did not reach the kernel")

    def test_countersink_depth_is_the_fastener_convention(self):
        """csk_angle is the FULL included angle (CadQuery's cskHole convention)."""
        # opening 8 -> 16 through a 90 deg included angle: half-angle 45, so the
        # cone is exactly as deep as the radial step (4mm).
        self.assertAlmostEqual(countersink_depth(8.0, 16.0, 90.0), 4.0, places=9)

    def test_an_underspecified_stepped_hole_is_refused_not_faked(self):
        """No 'conventional ratio' invented behind the caller's back.

        Substituting one is how a counterbore silently became a plain cylinder.
        """
        _, err = build(BOX + [Hole("", 0.0, 0.0, 8.0, None, True, "counterbore")])
        self.assertEqual(err, ["bad-value"])
        _, err = build(BOX + [Hole("", 0.0, 0.0, 8.0, None, True, "countersink")])
        self.assertEqual(err, ["bad-value"])


class HoleFaceTest(unittest.TestCase):
    """``face_or_sketch`` named the face to drill THROUGH, and was ignored."""

    def _blind(self, ref):
        b, err = build(BOX + [Hole(ref, 0.0, 0.0, 8.0, 10.0, False, "simple")])
        self.assertIsNone(err, "hole(%r) refused: %s" % (ref, err))
        return b.query("metrics")["center_of_mass"][2]

    def test_drilling_from_the_bottom_is_not_drilling_from_the_top(self):
        """A blind hole from '<Z' removes material low down; from '>Z', high up.

        So the centre of mass moves the OTHER way. Volume alone cannot see this,
        which is why the field could stay dead for so long.
        """
        top = self._blind(">Z")
        bottom = self._blind("<Z")
        self.assertGreater(bottom, top,
                           "face_or_sketch did not reach the kernel: a hole from "
                           "the bottom and one from the top gave the same solid")

    def test_the_incumbent_body_tokens_are_not_selectors(self):
        """'solid' / 'body' / 'last' / a feature id mean THE BODY, not a datum.

        Every reference op stream in the pressure corpus writes
        ``Hole(face_or_sketch="solid")``, and the output gate's property corpus
        writes ``"f1"``. Parsing those as selectors -- and bad-valuing them when
        they failed to parse -- broke every hole in the corpus. They are legal,
        in-use input and they must build.
        """
        for ref in ("", "solid", "body", "last", "f1"):
            _, err = build(BOX + [Hole(ref, 0.0, 0.0, 8.0, None, True, "simple")])
            self.assertIsNone(err, "hole(face_or_sketch=%r) was refused: %s"
                              % (ref, err))


class ShellSelectorTest(unittest.TestCase):
    """The schema said ``(">Z",)``; frep's vocabulary said ``("top",)``.

    So a shell written exactly as ``core/cisp/ops.py`` documents it got a
    ``bad-value`` from our own backend. The schema wins: CadQuery selector strings
    are canonical, and the old words survive as aliases.
    """

    def test_a_shell_written_as_the_schema_documents_it_builds(self):
        b, err = build(BOX + [Shell((">Z",), 6.0)])
        self.assertIsNone(err, "the canonical selector was refused: %s" % err)
        self.assertLess(volume(b), 60.0 * 40.0 * 20.0, "the shell removed nothing")

    def test_the_named_vocabulary_still_works_as_an_alias(self):
        b, err = build(BOX + [Shell(("top",), 6.0)])
        self.assertIsNone(err, "the legacy alias was refused: %s" % err)

    def test_opening_the_top_is_not_opening_the_bottom(self):
        top, _ = build(BOX + [Shell((">Z",), 6.0)])
        bottom, _ = build(BOX + [Shell(("<Z",), 6.0)])
        self.assertNotAlmostEqual(
            top.query("metrics")["center_of_mass"][2],
            bottom.query("metrics")["center_of_mass"][2], delta=0.1,
            msg="shell.faces did not reach the kernel")

    def test_a_malformed_selector_is_a_typed_error(self):
        _, err = build(BOX + [Shell(("bogus[",), 6.0)])
        self.assertEqual(err, ["bad-value"])


class ShellWallResolutionTest(unittest.TestCase):
    """frep must not build a wall its sampling grid cannot represent.

    An 80x30x5 plate shelled to t=1 at resolution 48 came back 78.13 x 28.22 x
    3.52 -- 75% under volume, the OUTER surface pulled in by 2mm on every side --
    watertight, 2-manifold, is_valid True, and ZERO diagnostics. The field is
    exact; the cell is 1.67mm and the wall is 1mm, so the wall is simply not in
    the sampled data.
    """

    PLATE = [NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 80.0, 30.0),
             Extrude("sk1", 5.0)]

    def test_a_sub_cell_wall_is_refused_on_every_mesher(self):
        for mesher in ("marching_cubes", "surface_nets", "dual_contouring"):
            b = FRepBackend(resolution=48, mesher=mesher)
            _, err = build(self.PLATE + [Shell((), 1.0)], backend=b)
            self.assertEqual(err, ["unsupported-op"],
                             "a 1mm wall at a 1.67mm cell was built anyway "
                             "(mesher=%s)" % mesher)

    def test_the_refusal_names_a_resolution_that_would_work(self):
        b = FRepBackend(resolution=48)
        result = None
        for op in self.PLATE + [Shell((), 1.0)]:
            result = b.apply(op)
        self.assertFalse(result.ok)
        self.assertIn("resolution=", result.diagnostics[0].message)

    def test_a_wall_the_grid_CAN_see_is_not_refused(self):
        """The check must not become a false positive.

        A 60x40x20 box at t=3 is a part frep builds CORRECTLY (analytic hollow
        volume 22296). An earlier threshold of 3 cells refused it and broke the
        output gate's own 200-stream property corpus. A refusal of a part the
        engine can build is not a safety improvement.
        """
        analytic = 60.0 * 40.0 * 20.0 - 54.0 * 34.0 * 14.0
        for res in (24, 32, 40, 48):
            b = FRepBackend(resolution=res)
            _, err = build(BOX + [Shell((), 3.0)], backend=b)
            self.assertIsNone(err, "a buildable 3mm wall was refused at res=%d: %s"
                              % (res, err))
            self.assertAlmostEqual(volume(b), analytic, delta=0.05 * analytic)

    def test_dual_contouring_holds_a_thinner_wall_than_marching_cubes(self):
        """The floor is a property of the EXTRACTION, not a taste in margins.

        Marching cubes can only put a crossing on a cell EDGE; dual contouring
        places a vertex INSIDE the cell by a QEF, so it holds a sub-cell wall
        further down. Encoding one number for both would be wrong for one of them.
        """
        self.assertLess(MIN_WALL_CELLS["dual_contouring"],
                        MIN_WALL_CELLS["marching_cubes"])


class ShellJoinTest(unittest.TestCase):

    def test_frep_refuses_the_intersection_join_rather_than_faking_it(self):
        """An SDF's inward offset IS the arc join (erosion by a ball).

        A miter join is an algebraic intersection of half-spaces the scalar field
        does not carry. Building the arc join and calling it 'intersection' is the
        bug; refusing is the honest answer.
        """
        _, err = build(BOX + [Shell((">Z",), 6.0, "intersection")])
        self.assertEqual(err, ["unsupported-op"])

    def test_the_arc_join_is_built(self):
        _, err = build(BOX + [Shell((">Z",), 6.0, "arc")])
        self.assertIsNone(err)


class BlendEdgeSelectorTest(unittest.TestCase):
    """``Fillet.edges`` was recorded on the feature and then thrown away.

    frep's ``blend_tree`` rounds EVERY convex edge of the body, so a fillet of the
    four vertical edges rounded all twelve and returned a valid, watertight, WRONG
    solid. A 20x10x5 box filleted r=1 on ``|Z`` has volume 995.6; on all twelve,
    970.6. Two different parts, one reported "ok".
    """

    SMALL = [NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 20.0, 10.0),
             Extrude("sk1", 5.0)]

    def test_filleting_four_edges_is_not_filleting_twelve(self):
        vertical, err = build(self.SMALL + [Fillet(("|Z",), 1.0)])
        self.assertIsNone(err, "the canonical selector was refused: %s" % err)
        every, err = build(self.SMALL + [Fillet((), 1.0)])
        self.assertIsNone(err)

        # the analytic |Z volume: 4 corners, each losing (r^2 - pi r^2/4) over the
        # full 5mm height.
        expected = 20.0 * 10.0 * 5.0 - 4.0 * (1.0 - math.pi / 4.0) * 5.0
        self.assertAlmostEqual(volume(vertical), expected, delta=0.5)
        self.assertLess(volume(every), volume(vertical) - 5.0,
                        "edges= was ignored: naming 4 edges rounded all 12")

    def test_different_selectors_are_different_parts(self):
        a, _ = build(self.SMALL + [Fillet(("|Z",), 1.0)])
        b, _ = build(self.SMALL + [Fillet((">Z",), 1.0)])
        self.assertNotAlmostEqual(volume(a), volume(b), delta=1.0)

    def test_chamfer_distance2_makes_an_asymmetric_chamfer(self):
        sym, err = build(self.SMALL + [Chamfer(("|Z",), 1.0, None)])
        self.assertIsNone(err)
        asym, err = build(self.SMALL + [Chamfer(("|Z",), 1.0, 2.0)])
        self.assertIsNone(err)
        self.assertLess(volume(asym), volume(sym),
                        "distance2 did not reach the kernel")

    def test_a_malformed_edge_selector_is_a_typed_error(self):
        _, err = build(self.SMALL + [Fillet(("bogus[",), 1.0)])
        self.assertEqual(err, ["bad-value"])


class FeatureReferenceTest(unittest.TestCase):
    """Every pattern and every mirror operated on the LAST SOLID.

    ``feature`` was validated (an unknown id was refused) and then never used --
    the most misleading shape this bug can take, because the reference is real, so
    it looks wired. Patterning the pad and patterning the pad-plus-fillet are
    different parts.
    """

    def test_linear_pattern_replicates_the_feature_it_names(self):
        pad, err = build(TWO_FEATURES + [LinearPattern("f1", (1.0, 0.0, 0.0), 3, 40.0)])
        self.assertIsNone(err)
        rounded, err = build(TWO_FEATURES + [LinearPattern("f2", (1.0, 0.0, 0.0), 3, 40.0)])
        self.assertIsNone(err)
        self.assertNotAlmostEqual(volume(pad), volume(rounded), delta=1.0,
                                  msg="linear_pattern.feature was ignored")
        # f1 is the SHARP pad, so patterning it leaves MORE material than f2
        self.assertGreater(volume(pad), volume(rounded))

    def test_circular_pattern_replicates_the_feature_it_names(self):
        pad, _ = build(TWO_FEATURES + [CircularPattern("f1", (0, 0, 0, 0, 0, 1), 4, 360.0)])
        rounded, _ = build(TWO_FEATURES + [CircularPattern("f2", (0, 0, 0, 0, 0, 1), 4, 360.0)])
        self.assertNotAlmostEqual(volume(pad), volume(rounded), delta=1.0,
                                  msg="circular_pattern.feature was ignored")

    def test_mirror_mirrors_the_feature_it_names(self):
        pad, _ = build(TWO_FEATURES + [Mirror("f1", "XZ")])
        rounded, _ = build(TWO_FEATURES + [Mirror("f2", "XZ")])
        self.assertNotAlmostEqual(volume(pad), volume(rounded), delta=1.0,
                                  msg="mirror.feature_or_body was ignored")

    def test_an_unknown_feature_is_still_refused(self):
        _, err = build(TWO_FEATURES + [Mirror("f99", "XZ")])
        self.assertEqual(err, ["bad-ref"])

    def test_an_empty_reference_still_means_the_last_body(self):
        named, _ = build(TWO_FEATURES + [LinearPattern("f2", (1.0, 0.0, 0.0), 3, 40.0)])
        default, _ = build(TWO_FEATURES + [LinearPattern("", (1.0, 0.0, 0.0), 3, 40.0)])
        self.assertAlmostEqual(volume(named), volume(default), delta=1e-6)


class CircularPatternAxisTest(unittest.TestCase):
    """The rotation axis was read, normalised, and thrown away: every circular
    pattern spun about Z."""

    def test_the_axis_decides_the_plane_the_pattern_sweeps(self):
        about_z, err = build(TWO_FEATURES + [CircularPattern("f1", (0, 0, 0, 0, 0, 1), 4, 360.0)])
        self.assertIsNone(err)
        about_x, err = build(TWO_FEATURES + [CircularPattern("f1", (0, 0, 0, 1, 0, 0), 4, 360.0)])
        self.assertIsNone(err)

        z_bbox = about_z.query("metrics")["bbox"]
        x_bbox = about_x.query("metrics")["bbox"]
        # a body 100mm out on +X, spun about Z, sweeps a wide disc in XY
        self.assertGreater(z_bbox[0], 200.0)
        self.assertGreater(z_bbox[1], 200.0)
        # spun about X it cannot move in X at all -- it sweeps in YZ instead
        self.assertLess(x_bbox[0], 30.0,
                        "circular_pattern.axis was ignored: the pattern spun "
                        "about Z when it was told to spin about X")
        self.assertGreater(x_bbox[2], 30.0)

    def test_a_degenerate_axis_is_refused(self):
        _, err = build(TWO_FEATURES + [CircularPattern("f1", (0, 0, 0, 0, 0, 0), 4, 360.0)])
        self.assertEqual(err, ["bad-value"])


class ConstraintTest(unittest.TestCase):
    """``value`` fed a DOF counter and never the geometry.

    ``Constrain(kind="radius", a="e1", value=8.0)`` on an r=6 circle left it at 6:
    the constraint was "applied", the sketch reported one fewer degree of freedom,
    and the part came out the wrong size.
    """

    CIRCLE = [NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 10.0)]

    def test_a_radius_constraint_drives_the_circle(self):
        for radius in (10.0, 20.0):
            b, err = build(self.CIRCLE
                           + [Constrain("radius", "e1", None, radius),
                              Extrude("sk1", 10.0)])
            self.assertIsNone(err)
            self.assertAlmostEqual(volume(b), math.pi * radius ** 2 * 10.0,
                                   delta=0.02 * math.pi * radius ** 2 * 10.0)

    def test_a_second_entity_on_a_unary_constraint_is_refused(self):
        """A radius is a property of ONE circle. 'b' cannot denote anything here."""
        _, err = build(self.CIRCLE + [AddCircle("sk1", 0.0, 0.0, 5.0),
                                      Constrain("radius", "e1", "e2", 10.0)])
        self.assertEqual(err, ["bad-value"])

    def test_the_in_use_constraint_forms_are_all_accepted(self):
        """The stock generators emit these. A validator that rejects them is a bug.

        ``data/datagen/generators.py`` constrains a RECTANGLE "horizontal" and
        emits ``distance`` with a single entity. Neither moves any geometry (there
        is no determined assignment), both are legal, and both must build. An
        earlier arity table refused them and broke every generator stream.
        """
        rect = [NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0)]
        for op in (Constrain("horizontal", "e1"),
                   Constrain("vertical", "e1"),
                   Constrain("distance", "e1", None, 60.0)):
            _, err = build(rect + [op, Extrude("sk1", 10.0)])
            self.assertIsNone(err, "%r was refused: %s" % (op, err))


class ThickenBothTest(unittest.TestCase):
    """``Thicken.both`` was read by NO backend -- frep, cadquery, build123d, stub.

    A symmetric thicken and a one-sided one accepted the same op and built the SAME
    solid, silently. It stayed invisible because the op was added to the protocol
    after the liveness oracle was written, so nothing ever probed it.
    """

    def test_a_symmetric_thicken_is_not_a_one_sided_one(self):
        one, err = build(BOX + [Thicken((), 4.0, False)])
        self.assertIsNone(err)
        both, err = build(BOX + [Thicken((), 4.0, True)])
        self.assertIsNone(err)
        self.assertNotAlmostEqual(volume(one), volume(both), places=3,
                                  msg="Thicken.both changed nothing -- the field "
                                      "is dead again")

    def test_both_moves_the_boundary_by_half_the_wall(self):
        """'symmetric about the surface' (ops.Thicken): half the wall lands inside,
        where the solid is already material -- so the boundary moves t/2."""
        both, err = build(BOX + [Thicken((), 4.0, True)])
        self.assertIsNone(err)
        half, err = build(BOX + [Thicken((), 2.0, False)])
        self.assertIsNone(err)
        self.assertAlmostEqual(volume(both), volume(half), delta=0.02 * volume(half))

    def test_thicken_delta_is_the_one_definition(self):
        self.assertEqual(thicken_delta(Thicken((), 4.0, False)), 4.0)
        self.assertEqual(thicken_delta(Thicken((), 4.0, True)), 2.0)


class MatePortTest(unittest.TestCase):
    """A port-typed Mate was GATED for admissibility and then recorded without its
    ports: ``{"kind","a","b","value"}`` in all four backends. So a mate joining
    'p1' and one joining 'p2' produced identical model state -- four DEAD cells."""

    ASM = BOX + [AddInstance("f1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                 AddInstance("f1", 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)]

    def _mates(self, op):
        b, err = build(self.ASM + [op])
        self.assertIsNone(err, "mate was refused: %s" % (err,))
        return b.query("assembly")["mates"]

    def test_the_ports_a_mate_names_are_recorded(self):
        m = self._mates(Mate("coaxial", "i1", "i2", None,
                             base_port="p1", incoming_port="q1",
                             base_port_type="bore", incoming_port_type="axis"))[0]
        self.assertEqual(m["base_port"], "p1")
        self.assertEqual(m["incoming_port"], "q1")
        self.assertEqual(m["base_port_type"], "bore")
        self.assertEqual(m["incoming_port_type"], "axis")

    def test_two_different_ports_are_two_different_assemblies(self):
        a = self._mates(Mate("coaxial", "i1", "i2", None, base_port="p1",
                             incoming_port="q1", base_port_type="bore",
                             incoming_port_type="axis"))
        b = self._mates(Mate("coaxial", "i1", "i2", None, base_port="p2",
                             incoming_port="q1", base_port_type="bore",
                             incoming_port_type="axis"))
        self.assertNotEqual(a, b, "the port name was dropped again")

    def test_a_plain_id_only_mate_records_exactly_what_it_always_did(self):
        """The port keys are only emitted for a port-typed mate, so the historical
        record is byte-identical and no downstream reader sees a new key."""
        m = self._mates(Mate("rigid", "i1", "i2", None))[0]
        self.assertEqual(m, {"kind": "rigid", "a": "i1", "b": "i2", "value": None})


class SchemaCoverageTest(unittest.TestCase):
    """The anti-rot latch that failed: three commits added twelve ops to the CISP
    protocol and none of them taught the liveness oracle, so 48 fields -- the ones
    most likely to be unwired, being the newest -- went unprobed."""

    def test_every_op_field_has_a_liveness_fixture(self):
        from harnesscad.eval.selftest import field_liveness

        self.assertEqual(field_liveness.unmapped(), [],
                         "the op schema grew and the liveness oracle did not")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
