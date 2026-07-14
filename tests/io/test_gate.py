"""The output gate: the harness must never SHIP a wrong part.

The claim under test is *soundness*, not completeness. The harness is allowed to
fail to produce a part. It is not allowed to produce a wrong one and write it to
disk. Formally, for every artifact the harness is asked to emit:

    outcome in { a written file that passes measurement,
                 a typed refusal, with no file }

and nothing else. The forbidden THIRD OUTCOME -- "wrote it anyway" -- is the bug
class the gate exists to eliminate, and
``PropertyTotalCorrectnessTest.test_the_third_outcome_never_occurs`` asserts it
never happens across a seeded corpus of 200 generated op streams.

These are unittest.TestCase classes on purpose: bare pytest-style functions
collect as ZERO tests under ``python -m unittest``, and a soundness suite that
silently runs nothing is worse than no suite at all.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Chamfer, Extrude, Fillet, Hole,
    LinearPattern, Mirror, NewSketch, Shell,
)
from harnesscad.domain.geometry.sdf import field_transforms as xf
from harnesscad.io import gate
from harnesscad.io import render as render_route
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.formats import registry as formats
from harnesscad.io.formats import stl as stl_codec

#: Every writable format a session can be exported to. An invalid part must be
#: refused on ALL of them: one ungated codec is a hole in the hull.
EXPORT_EXTENSIONS = ("stl", "obj", "glb", "amf", "step", "svg", "png")

#: Coarse enough that 200 marching-cubes builds finish, fine enough that the
#: errors the gate hunts (millimetres on a 60 mm part) are far outside the
#: discretisation noise.
CORPUS_RESOLUTION = 18
CORPUS_SIZE = 200
CORPUS_SEED = 20260714


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def build_box(w=60.0, h=40.0, d=20.0, resolution=40):
    """A plain w x h x d box on a live F-rep backend."""
    b = FRepBackend(resolution=resolution)
    b.reset()
    sid = b.apply(NewSketch(plane="XY")).created[0]
    b.apply(AddRectangle(sketch=sid, x=0.0, y=0.0, w=w, h=h))
    b.apply(Extrude(sketch=sid, distance=d))
    return b


def cube_mesh(size=10.0, invert=False):
    """A closed, correctly wound cube -- or an inside-out one."""
    s = float(size)
    v = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
         (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
         (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    if invert:
        f = [(a, c, b) for (a, b, c) in f]
    return [tuple(float(c) for c in p) for p in v], f


def triangles_of(verts, faces):
    return [stl_codec.Triangle(verts[a], verts[b], verts[c]) for a, b, c in faces]


class DegenerateBackend:
    """A backend double whose geometry is broken in a named way.

    Real backends are hard to coerce into emitting garbage -- which is the point
    of them -- so "every export path refuses a bad part" is tested with a backend
    that hands the registry exactly the bad geometry we name. It quacks like a
    GeometryBackend (``export`` + ``state_digest``), which is all the registry and
    the gate ever ask of one.
    """

    def __init__(self, flavour="zero_volume"):
        self.flavour = flavour

    def geometry(self):
        if self.flavour == "zero_volume":
            v = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
            return v, [(0, 1, 2), (0, 2, 3)]          # flat plate: open, no volume
        if self.flavour == "non_manifold":
            v, f = cube_mesh()                        # cube + a fin on one edge
            return list(v) + [(5.0, 5.0, 25.0)], list(f) + [(0, 1, 8)]
        if self.flavour == "inverted":
            return cube_mesh(invert=True)
        raise AssertionError("unknown flavour " + self.flavour)

    def mesh(self, tolerance=None):
        return self.geometry()

    def export(self, fmt, tolerance=None, angular_tolerance=None):
        v, f = self.geometry()
        if fmt == "step":
            # Structurally fine STEP text. The point is that the gate must judge
            # the GEOMETRY behind it, not the bytes it is handed.
            return ("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n"
                    "#1=PRODUCT('p','p','',());\nENDSEC;\nEND-ISO-10303-21;\n")
        if fmt == "glb":
            from harnesscad.io.formats import glb as glb_codec
            return glb_codec.write_glb(triangles_of(v, f), name="bad")
        return stl_codec.write_binary_stl(triangles_of(v, f))

    def state_digest(self):
        return "degenerate-" + self.flavour


class ScriptedBackend:
    """A backend whose op log and metrics are scripted, to drive the intent rules.

    The F-rep backend cannot be made to *increase* volume with a boolean cut --
    good. But the gate must hold the rule anyway, because another backend, another
    model, another config might. So the rule is tested against a backend that
    breaks it on purpose.
    """

    SCRIPT: list = []

    def __init__(self):
        self.reset()

    def reset(self):
        self._oplog = []
        self._bodies = [{"id": "f0"}]
        self._state = {"volume": 1000.0, "bbox": [10.0, 10.0, 10.0]}

    def apply(self, op):
        from harnesscad.io.backends.base import ApplyResult

        for scripted_op, state in self.SCRIPT:
            if scripted_op is op:
                self._state = dict(state)
                break
        self._oplog.append(op)
        return ApplyResult(True, ["f%d" % len(self._oplog)])

    def query(self, q):
        return dict(self._state) if q == "metrics" else {}

    def mesh(self, tolerance=None):
        return cube_mesh()

    def export(self, fmt, tolerance=None, angular_tolerance=None):
        v, f = cube_mesh()
        return stl_codec.write_binary_stl(triangles_of(v, f))

    def state_digest(self):
        return "scripted"


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="harnesscad-gate-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.tmp, name)


# ---------------------------------------------------------------------------
# 1. The shipped bug: a shell hollows inward; it must not dilate.
# ---------------------------------------------------------------------------

class ShellSemanticTest(unittest.TestCase):

    def test_shelled_box_keeps_its_bounding_box(self):
        """A 60x40x20 box shelled at t=3 comes out 60x40x20 -- hollow, not dilated.

        This is the exact part that shipped wrong. It went out 63x43x23: 3 mm
        oversize in every dimension, watertight, and nothing complained.
        """
        b = build_box()
        before = b.query("metrics")["bbox"]
        self.assertTrue(b.apply(Shell(faces=[], thickness=3.0)).ok)
        after = b.query("metrics")

        self.assertEqual([round(c, 2) for c in before], [60.0, 40.0, 20.0])
        self.assertEqual([round(c, 2) for c in after["bbox"]], [60.0, 40.0, 20.0],
                         "the shell grew the part: this is the bug that shipped")
        # ...and it is genuinely HOLLOW, not merely the right size: outer 48000,
        # cavity 54x34x14 = 25704, so the wall is ~22296 mm3.
        self.assertGreater(after["volume"], 21000.0)
        self.assertLess(after["volume"], 23500.0)
        self.assertTrue(b.query("validity")["is_valid"])

    def test_shell_field_is_the_cad_semantic_not_curvs(self):
        """``shell_inward`` keeps the boundary; Curv's ``shell`` moves it outward."""
        self.assertEqual(xf.shell_inward(-10.0, 3.0), 7.0)    # deep inside -> out
        self.assertEqual(xf.shell_inward(0.0, 3.0), 0.0)      # BOUNDARY UNMOVED
        self.assertEqual(xf.shell_inward(-1.5, 3.0), -1.5)    # in the wall
        self.assertEqual(xf.shell_inward(5.0, 3.0), 5.0)      # outside stays out
        self.assertEqual(xf.shell(0.0, 3.0), -1.5)            # Curv: boundary inside

    def test_the_wall_is_actually_the_declared_thickness(self):
        """The gate measures the WALL, not just the envelope."""
        b = build_box()
        b.apply(Shell(faces=[], thickness=3.0))
        report = gate.check(b, source=b)
        self.assertTrue(report.ok, [f.check for f in report.failures])
        wall = wall_record(report)
        self.assertEqual(wall["status"], "pass")
        self.assertAlmostEqual(wall["deepest_material"], 3.0,
                               delta=wall["tolerance"])


def wall_record(report):
    for c in report.declared:
        if c["check"] == "shell-wall-thickness":
            return c
    raise AssertionError("the gate did not probe the wall at all")


# ---------------------------------------------------------------------------
# 2. A bbox check cannot prove a shell is correct.
# ---------------------------------------------------------------------------

class ShellWallTest(TempDirTest):

    def test_a_perfect_bbox_does_not_save_a_wall_that_is_too_thin(self):
        """A wall 42% too thin, with the bounding box preserved EXACTLY.

        The outer surface of an inward shell is untouched by construction, so the
        one number the envelope check reads is the one number a broken shell is
        guaranteed to get right. If the gate stopped at the bbox it would ship
        this part. It must not.
        """
        original = xf.shell_inward
        xf.shell_inward = lambda d, t: max(d, -(d + t * 0.58))
        try:
            b = build_box()
            b.apply(Shell(faces=[], thickness=3.0))

            # The envelope is intact -- the shell did not grow the part by so much
            # as a micron, so the bbox check (and every check built on it) PASSES.
            # The part is still wrong.
            #
            # REBASELINED for the dual-contouring default (was ``+ 1e-6``). The
            # old bound was an artefact of marching cubes: MC pins every vertex to
            # a grid EDGE, and for an axis-aligned box those edges lie exactly on
            # the face planes, so MC's bbox is exact and its error is one-sided
            # (it only ever chamfers corners INWARD). A dual-contouring vertex is
            # a QEF minimiser and may sit a fraction of a micron OUTSIDE the true
            # face: the same 60x40x20 box measures [59.999977, 40.000179,
            # 19.999595]. That is 2e-4 mm of tessellation noise, four orders of
            # magnitude below the bug this test is about (a two-sided shell grows
            # the part by t/2 = 1.5 mm).
            bbox = b.query("metrics")["bbox"]
            for measured, declared in zip(bbox, (60.0, 40.0, 20.0)):
                self.assertLessEqual(measured, declared + 1e-3,
                                     "the envelope grew; this test is about a "
                                     "shell whose envelope is CORRECT")
                self.assertAlmostEqual(measured, declared, delta=0.5)
            envelope_only = gate.declared_failures(b)[1]
            self.assertEqual(
                next(c for c in envelope_only
                     if c["check"] == "shell-preserves-bbox")["status"], "pass",
                "the envelope check must PASS here -- that is the whole point")

            out = self.path("thin.stl")
            with self.assertRaises(gate.InvalidArtifact) as caught:
                formats.export_session(b, out)
            self.assertFalse(os.path.exists(out))

            codes = {f.check for f in caught.exception.failures}
            self.assertIn("shell-wrong-wall", codes)
            failure = next(f for f in caught.exception.failures
                           if f.check == "shell-wrong-wall")
            self.assertAlmostEqual(failure.measured, 1.74, delta=0.35)
            self.assertEqual(failure.expected, 3.0)
        finally:
            xf.shell_inward = original

    def test_a_wall_that_is_too_thick_is_caught_too(self):
        original = xf.shell_inward
        xf.shell_inward = lambda d, t: max(d, -(d + t * 2.0))
        try:
            b = build_box()
            b.apply(Shell(faces=[], thickness=3.0))
            report = gate.check(b, "fat.stl", source=b)
            self.assertFalse(report.ok)
            self.assertIn("shell-wrong-wall", {f.check for f in report.failures})
        finally:
            xf.shell_inward = original


# ---------------------------------------------------------------------------
# 3. A degenerate part is refused on EVERY export path, and no file appears.
# ---------------------------------------------------------------------------

class DegenerateRefusalTest(TempDirTest):

    FLAVOURS = ("zero_volume", "non_manifold", "inverted")

    def test_every_export_path_refuses_a_degenerate_part(self):
        for flavour in self.FLAVOURS:
            for ext in EXPORT_EXTENSIONS:
                with self.subTest(flavour=flavour, format=ext):
                    out = self.path("part_%s.%s" % (flavour, ext))
                    with self.assertRaises(gate.InvalidArtifact) as caught:
                        formats.export_session(DegenerateBackend(flavour), out)
                    self.assertFalse(
                        os.path.exists(out),
                        "%s: an INVALID artifact was written to disk" % ext)
                    self.assertTrue(caught.exception.failures,
                                    "a refusal must name what failed")

    def test_the_refusal_names_the_measurement_that_failed(self):
        """A refusal is not a shrug. It says which number was wrong."""
        expected = {
            "zero_volume": {"not-watertight", "degenerate-volume", "degenerate-bbox"},
            "non_manifold": {"not-2-manifold"},
            "inverted": {"inverted-normals"},
        }
        for flavour, wanted in expected.items():
            with self.subTest(flavour=flavour):
                with self.assertRaises(gate.InvalidArtifact) as caught:
                    formats.export_session(DegenerateBackend(flavour),
                                           self.path("p.stl"))
                codes = {f.check for f in caught.exception.failures}
                self.assertTrue(wanted & codes,
                                "expected one of %r, got %r" % (wanted, codes))

    def test_the_renderer_refuses_too(self):
        """A PNG is an artifact. A beautiful render of a broken part is how it ships."""
        out = self.path("shot.png")
        with self.assertRaises(gate.InvalidArtifact):
            render_route.render(DegenerateBackend("inverted"), out)
        self.assertFalse(os.path.exists(out))

    def test_the_backends_own_write_stl_is_gated(self):
        """``FRepBackend.write_stl`` bypasses the registry, so the gate stands there."""
        b = FRepBackend(resolution=20)
        b.reset()
        sid = b.apply(NewSketch(plane="XY")).created[0]
        b.apply(AddRectangle(sketch=sid, x=0.0, y=0.0, w=10.0, h=10.0))
        # No extrude: there is no solid, so there is no part.
        out = self.path("empty.stl")
        with self.assertRaises(gate.InvalidArtifact) as caught:
            b.write_stl(out)
        self.assertFalse(os.path.exists(out))
        self.assertIn("empty-geometry",
                      {f.check for f in caught.exception.failures})


# ---------------------------------------------------------------------------
# 4. The declared-intent rules.
# ---------------------------------------------------------------------------

class DeclaredIntentTest(TempDirTest):

    def test_a_shell_that_grows_the_part_is_refused(self):
        """Put the shipped bug back, and watch the gate refuse the artifact.

        This is the regression that matters most. Not "is the shell fixed" -- that
        is ShellSemanticTest -- but "if the shell breaks AGAIN, under any model or
        any config, does the harness still refuse to ship it". The gate must not
        depend on the op being correct.
        """
        original = xf.shell_inward
        xf.shell_inward = xf.shell                 # Curv's dilating shell
        try:
            b = build_box(resolution=32)
            b.apply(Shell(faces=[], thickness=3.0))
            grown = [round(c, 1) for c in b.query("metrics")["bbox"]]
            self.assertEqual(grown, [63.0, 43.0, 23.0], "the bug did not reproduce")

            out = self.path("enclosure.stl")
            with self.assertRaises(gate.InvalidArtifact) as caught:
                formats.export_session(b, out)
            self.assertFalse(os.path.exists(out), "the oversize part was SHIPPED")

            failure = next(f for f in caught.exception.failures
                           if f.check == "shell-grew-bbox")
            self.assertEqual(failure.family, "declared")
        finally:
            xf.shell_inward = original

    def test_a_cut_that_increases_volume_is_refused(self):
        cut = Boolean(kind="cut", target="f1", tool="f2")
        ScriptedBackend.SCRIPT = [(cut, {"volume": 5000.0,
                                         "bbox": [10.0, 10.0, 10.0]})]
        try:
            b = ScriptedBackend()
            b.apply(cut)                       # volume 1000 -> 5000, under a CUT
            failures, checks = gate.declared_failures(b)
            self.assertIn("cut-increased-volume", {f.check for f in failures})
            self.assertTrue(any(c["check"] == "cut-removes-material" for c in checks))
        finally:
            ScriptedBackend.SCRIPT = []

    def test_extrude_height_matches_the_declared_distance(self):
        b = build_box(d=12.5, resolution=32)
        failures, checks = gate.declared_failures(b)
        self.assertEqual(failures, [])
        height = next(c for c in checks if c["check"] == "extrude-height")
        self.assertEqual(height["status"], "pass")
        self.assertEqual(height["axis"], "Z")
        self.assertAlmostEqual(height["measured"], 12.5, delta=0.6)

    def test_a_clean_report_proves_the_gate_actually_looked(self):
        """A pass is worth nothing unless the gate can show what it checked."""
        b = build_box(resolution=32)
        b.apply(Shell(faces=[], thickness=3.0))
        report = gate.check(b, "part.stl", source=b)
        self.assertTrue(report.ok)
        checked = {c["check"] for c in report.declared}
        self.assertIn("shell-preserves-bbox", checked)
        self.assertIn("shell-wall-thickness", checked)
        self.assertIn("extrude-height", checked)
        self.assertEqual(report.measurement["declared_intent"], "checked")


# ---------------------------------------------------------------------------
# 5. The measurement must be of our GEOMETRY, not of our TESSELLATION.
# ---------------------------------------------------------------------------

class MeasurementProvenanceTest(unittest.TestCase):

    def test_the_gate_names_the_tessellation_its_numbers_came_off(self):
        """CadQuery's export default is a 0.1 mm deflection -- the same order as
        the errors the gate hunts -- and the harness passed no tolerance at all.

        The gate does not re-tessellate at a private setting (that would certify a
        mesh nobody exports). It measures the mesh the harness ships, and it
        NAMES the tessellation that mesh came off, on every report.
        """
        b = build_box(resolution=32)
        report = gate.check(b, source=b)
        tess = report.measurement["tessellation"]
        self.assertTrue(tess["controlled"],
                        "the gate measured a mesh whose tessellation it cannot name")
        self.assertEqual(tess["kind"], "grid")       # F-rep: an exact field on a grid
        self.assertEqual(tess["resolution"], 32)
        self.assertEqual(tess["route"], "backend.mesh()")

    def test_a_backend_that_cannot_name_its_tessellation_is_flagged(self):
        """Silence about the mesh is not a pass. It is a caveat, and it is printed."""
        tess = gate._tessellation_of(ScriptedBackend())   # declares no control
        self.assertFalse(tess["controlled"])
        self.assertIn("unknown discretisation error", tess["warning"])

    def test_a_deflection_backend_reports_its_pinned_tolerance(self):
        class Pinned(ScriptedBackend):
            LINEAR_DEFLECTION = 0.01
            ANGULAR_DEFLECTION = 0.1

        tess = gate._tessellation_of(Pinned())
        self.assertTrue(tess["controlled"])
        self.assertEqual(tess["kind"], "deflection")
        self.assertEqual(tess["linear_deflection"], 0.01)
        self.assertNotIn("warning", tess)

    def test_a_coarse_backend_is_warned_about(self):
        """The 0.1 mm default that silently corrupted every measurement."""
        class Coarse(ScriptedBackend):
            LINEAR_DEFLECTION = 0.1       # CadQuery's export default

        tess = gate._tessellation_of(Coarse())
        self.assertTrue(tess["controlled"])
        self.assertIn("coarser than the gate's reference", tess["warning"])

    def test_every_report_carries_its_provenance(self):
        b = build_box(resolution=24)
        report = gate.check(b, source=b)
        self.assertIn("tessellation", report.measurement)
        self.assertEqual(report.measurement["geometry_source"], "backend")


class HonestyTest(unittest.TestCase):
    """A gate that claims more than it verifies is worse than no gate."""

    def test_the_gate_states_what_it_does_not_prove(self):
        claims = gate.claims()
        self.assertTrue(claims["proves"])
        self.assertTrue(claims["does_not_prove"])
        blob = " ".join(claims["does_not_prove"]).lower()
        # THE ORACLE IS MANY-TO-ONE: volume + bbox + genus do not pin down a part.
        self.assertIn("brief", blob)
        self.assertIn("place", blob)

    def test_every_report_carries_the_disclaimer(self):
        b = build_box(resolution=24)
        report = gate.check(b, source=b)
        self.assertTrue(report.measurement["does_not_prove"])
        self.assertTrue(report.measurement["proves"])

    def test_a_hole_in_the_wrong_place_is_not_caught(self):
        """The gate is honest about this, so the suite PINS it as known.

        Two parts differing only in where a hole was bored are both valid solids.
        The gate passes both. That is not a bug in the gate -- it is the boundary
        of what a measured oracle can establish, and it is exactly why
        DOES_NOT_PROVE says so out loud. If this test ever fails, the gate got
        STRONGER and the disclaimer must be narrowed to match.
        """
        def bored(x):
            b = build_box(resolution=28)
            b.apply(Hole(face_or_sketch="f1", x=x, y=20.0, diameter=8.0,
                         depth=5.0, through=True, kind="simple"))
            return b

        right = gate.check(bored(20.0), source=bored(20.0))
        wrong = gate.check(bored(40.0), source=bored(40.0))
        self.assertTrue(right.ok)
        self.assertTrue(wrong.ok)


# ---------------------------------------------------------------------------
# 6. --force: allowed, but never silent.
# ---------------------------------------------------------------------------

class ForceTest(TempDirTest):

    def test_force_writes_the_artifact_and_indicts_it(self):
        out = self.path("bad.stl")
        formats.export_session(DegenerateBackend("inverted"), out, force=True)
        self.assertTrue(os.path.exists(out), "--force must actually write the file")

        sidecar = self.path("bad.INVALID.json")
        self.assertTrue(os.path.exists(sidecar),
                        "a forced artifact is NEVER written silently")
        with open(sidecar, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.assertFalse(payload["ok"])
        self.assertTrue(payload["forced"])
        self.assertIn("inverted-normals", {f["check"] for f in payload["failures"]})
        self.assertIn("not a valid part", payload["warning"])

    def test_a_valid_artifact_gets_no_sidecar(self):
        out = self.path("good.stl")
        formats.export_session(build_box(resolution=20), out, force=True)
        self.assertTrue(os.path.exists(out))
        self.assertFalse(os.path.exists(self.path("good.INVALID.json")))


# ---------------------------------------------------------------------------
# 7. THE TOTAL-CORRECTNESS CLAIM.
# ---------------------------------------------------------------------------

def random_stream(rng):
    """A randomly-parameterised but structurally-plausible op stream.

    Deliberately includes parameters that CANNOT work -- a wall thicker than the
    part, a hole wider than the stock -- because those must be REFUSED, not
    shipped, and a corpus of only-valid parts would prove nothing.
    """
    ops = [NewSketch(plane=rng.choice(["XY", "YZ", "XZ"]))]
    if rng.random() < 0.3:
        ops.append(AddCircle(sketch="sk1", cx=0.0, cy=0.0, r=rng.uniform(4.0, 40.0)))
    else:
        ops.append(AddRectangle(sketch="sk1", x=0.0, y=0.0,
                                w=rng.uniform(8.0, 80.0), h=rng.uniform(8.0, 80.0)))
    ops.append(Extrude(sketch="sk1", distance=rng.uniform(1.0, 40.0)))

    for _ in range(rng.randint(0, 2)):
        pick = rng.random()
        if pick < 0.30:
            ops.append(Shell(faces=[], thickness=rng.uniform(0.5, 12.0)))
        elif pick < 0.55:
            ops.append(Hole(face_or_sketch="f1", x=0.0, y=0.0,
                            diameter=rng.uniform(2.0, 50.0),
                            depth=rng.uniform(1.0, 30.0),
                            through=rng.random() < 0.5, kind="simple"))
        elif pick < 0.70:
            ops.append(Fillet(edges=["f1"], radius=rng.uniform(0.5, 10.0)))
        elif pick < 0.82:
            ops.append(Chamfer(edges=["f1"], distance=rng.uniform(0.5, 8.0)))
        elif pick < 0.92:
            ops.append(Mirror(feature_or_body="",
                              plane=rng.choice(["XY", "YZ", "XZ"])))
        else:
            ops.append(LinearPattern(
                feature="f1",
                direction=rng.choice([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]),
                count=rng.randint(2, 4), spacing=rng.uniform(5.0, 40.0)))
    return ops


class PropertyTotalCorrectnessTest(TempDirTest):
    """The soundness claim, as a test rather than a promise."""

    def test_the_third_outcome_never_occurs(self):
        """Over the corpus, no file is ever written that fails measurement.

        For every stream the harness is asked to export, EXACTLY ONE of:

            (a) a file is written -- and re-measuring that file, read back off the
                disk, finds nothing wrong with it; or
            (b) a typed refusal is raised, and NO file exists.

        The forbidden third outcome -- a file on disk that fails the measured
        checks -- is asserted never to occur.
        """
        rng = random.Random(CORPUS_SEED)
        shipped = refused = plan_rejected = 0
        third_outcome = []
        refusal_codes = {}

        for i in range(CORPUS_SIZE):
            backend = FRepBackend(resolution=CORPUS_RESOLUTION)
            backend.reset()

            if not all(backend.apply(op).ok for op in random_stream(rng)):
                plan_rejected += 1        # the backend refused the plan: never
                continue                  # reached the gate, so nothing was emitted

            out = self.path("part_%03d.stl" % i)
            try:
                formats.export_session(backend, out)
            except gate.InvalidArtifact as exc:
                refused += 1
                self.assertFalse(
                    os.path.exists(out),
                    "stream %d: REFUSED, but the file was written anyway" % i)
                self.assertTrue(exc.failures, "a refusal must be typed")
                for f in exc.failures:
                    refusal_codes[f.check] = refusal_codes.get(f.check, 0) + 1
                continue
            except formats.FormatError:
                refused += 1              # also a refusal: typed, nothing written
                self.assertFalse(os.path.exists(out))
                continue

            # (a) A file was written. It must survive an INDEPENDENT re-measurement
            # read back off the disk. We do not trust the writer.
            self.assertTrue(os.path.exists(out))
            with open(out, "rb") as fh:
                verts, faces = gate._weld(stl_codec.parse_stl(fh.read()))
            residual = gate.measured_failures(gate.measure(verts, faces))
            if residual:
                third_outcome.append((i, [f.check for f in residual]))
            shipped += 1

        self.assertEqual(
            third_outcome, [],
            "THE THIRD OUTCOME OCCURRED: %d artifact(s) were WRITTEN to disk that "
            "fail the measured checks: %r" % (len(third_outcome), third_outcome[:5]))

        # The corpus must actually exercise both outcomes, or the claim is vacuous.
        self.assertGreaterEqual(shipped, 20, "too few parts shipped; corpus too easy")
        self.assertGreaterEqual(
            refused, 1,
            "nothing was refused: the corpus never stressed the gate (codes: %r)"
            % (refusal_codes,))
        self.assertGreaterEqual(shipped + refused, 100,
                                "too few streams reached the gate")
        print("\n  corpus: %d streams | %d shipped-valid | %d refused | "
              "%d rejected pre-gate | THIRD OUTCOME: %d\n  refusal codes: %r"
              % (CORPUS_SIZE, shipped, refused, plan_rejected, len(third_outcome),
                 refusal_codes))


if __name__ == "__main__":
    unittest.main()
