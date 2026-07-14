"""End-to-end tests for the OpenSCAD geometry backend.

These drive a real CISP op stream through HarnessSession -> OpenScadBackend ->
the openscad binary, and assert on the geometry that comes back: a watertight
2-manifold mesh, the analytic volume of the extruded plate, and a boolean cut
that removes EXACTLY the right amount of material.

The exactness claim is the point of this backend. OpenSCAD is a CGAL CSG kernel,
so a cut is not sampled -- and because the circular tool is faceted by OpenSCAD's
own $fn law (which this repo already ports, in domain.geometry.parametric.facets),
the volume of the removed material is known in CLOSED FORM: the volume of a
regular n-gon prism. The test asserts against that number, not against a
tolerance band. The SDF backend, by contrast, can only be within a fraction of a
percent of it -- its mesh is limited by the marching-cubes grid -- and
:meth:`OpenScadVsFRepTest.test_openscad_is_exact_where_frep_is_grid_limited`
pins that difference down.

Skips cleanly (unittest.skipUnless) when openscad is not installed.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.domain.geometry.parametric import facets
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.external import DEFAULT_SEGMENTS
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.backends.openscad import (
    OpenScadBackend, OpenScadError, countersink_depth,
)
from harnesscad.io.formats import stl as stl_fmt
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

HAVE_OPENSCAD = OpenScadBackend.available()
REASON = "openscad is not installed on this machine"

# A 20 x 10 rectangle extruded 5 -> analytic volume 1000.
PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]

# The same plate, with an r=3 cylinder extruded through it and cut away.
HOLE_RADIUS = 3.0
CUT_OPS = PLATE_OPS + [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk2", "cx": 10.0, "cy": 5.0, "r": HOLE_RADIUS},
    {"op": "extrude", "sketch": "sk2", "distance": 5.0},
    {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"},
]

PLATE_VOLUME = 20.0 * 10.0 * 5.0


def ngon_prism_volume(r: float, height: float, segments: int) -> float:
    """The EXACT volume of the prism OpenSCAD builds for a circle of radius r.

    OpenSCAD does not extrude a circle; it extrudes the inscribed regular n-gon
    it resolves the circle into. Its area is ``n/2 * r^2 * sin(2*pi/n)``. This is
    the ground truth a real CSG kernel must hit, and it is strictly less than the
    true cylinder ``pi*r^2*h``.
    """
    n = facets.get_fragments_from_r(r, fn=float(segments))
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n) * height


HOLE_VOLUME = ngon_prism_volume(HOLE_RADIUS, 5.0, DEFAULT_SEGMENTS)
CUT_VOLUME = PLATE_VOLUME - HOLE_VOLUME

#: Everything comes back through a binary STL, whose vertices are float32. That
#: is the only error budget this backend has -- ~1e-7 relative on a 20 mm part.
STL_FLOAT32_TOLERANCE = 1e-4


def run_ops(backend, ops) -> None:
    session = HarnessSession(backend)
    result = session.apply_ops([parse_op(o) for o in ops])
    if not result.ok:
        raise AssertionError("op stream rejected: %s"
                             % [d.to_dict() for d in result.diagnostics])


class OpenScadAvailabilityTest(unittest.TestCase):
    """The graceful-absence contract holds whether or not openscad is here."""

    def test_registered_in_the_backend_table(self):
        self.assertIn("openscad", BACKENDS)

    def test_available_never_raises(self):
        self.assertIsInstance(OpenScadBackend.available(), bool)

    def test_backend_unavailable_is_typed_and_actionable(self):
        exc = BackendUnavailable("openscad", "not here", ["PATH:openscad"])
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.tool, "openscad")
        self.assertEqual(exc.searched, ["PATH:openscad"])

    def test_server_never_crashes_on_a_missing_tool(self):
        """Present -> the real backend; absent -> the stub, WITH a note. Never a
        traceback: the CLI must stay usable on a machine without the binary."""
        server = CISPServer(backend="openscad")
        if HAVE_OPENSCAD:
            self.assertEqual(server.backend_name, "openscad")
            self.assertIsNone(server.backend_note)
        else:
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("openscad", server.backend_note)


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class OpenScadBackendTest(unittest.TestCase):

    def test_satisfies_the_geometry_backend_protocol(self):
        self.assertIsInstance(OpenScadBackend(), GeometryBackend)

    def test_plate_builds_and_has_the_analytic_volume(self):
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        self.assertTrue(backend.query("summary")["solid_present"])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], PLATE_VOLUME, delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 6) for v in measure["bbox"]], [20.0, 10.0, 5.0])

    def test_plate_mesh_is_a_watertight_2_manifold(self):
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        verts, faces = backend.mesh()
        self.assertTrue(faces)
        he = HalfedgeMesh(verts, faces)
        self.assertTrue(he.is_2manifold()[0])
        self.assertTrue(he.is_closed())
        self.assertEqual(backend.query("validity")["genus"], 0)

    def test_boolean_cut_removes_material(self):
        plate = OpenScadBackend()
        run_ops(plate, PLATE_OPS)
        cut = OpenScadBackend()
        run_ops(cut, CUT_OPS)
        self.assertLess(cut.query("measure")["volume"],
                        plate.query("measure")["volume"])

    def test_boolean_cut_removes_EXACTLY_the_right_material(self):
        """A real CSG kernel is not 'close': the cut volume is the closed-form
        volume of the n-gon prism OpenSCAD's own $fn law resolves the circle to,
        to float32 STL precision and no worse."""
        backend = OpenScadBackend()
        run_ops(backend, CUT_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, CUT_VOLUME, delta=STL_FLOAT32_TOLERANCE)
        removed = PLATE_VOLUME - volume
        self.assertAlmostEqual(removed, HOLE_VOLUME, delta=STL_FLOAT32_TOLERANCE)

    def test_cut_mesh_is_a_watertight_genus_1_solid(self):
        backend = OpenScadBackend()
        run_ops(backend, CUT_OPS)
        validity = backend.query("validity")
        self.assertTrue(validity["is_valid"])
        self.assertTrue(validity["watertight"])
        self.assertEqual(validity["genus"], 1)  # a plate with a through-hole

    def test_exported_stl_round_trips_through_io_formats_stl(self):
        backend = OpenScadBackend()
        run_ops(backend, CUT_OPS)
        data = backend.export("stl")
        triangles = stl_fmt.parse_stl(data)
        self.assertTrue(triangles)
        self.assertAlmostEqual(abs(stl_fmt.signed_volume(triangles)), CUT_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        # binary -> ascii -> binary is lossless in triangle count and volume
        ascii_text = backend.export("stl-ascii")
        reparsed = stl_fmt.parse_ascii_stl(ascii_text)
        self.assertEqual(len(reparsed), len(triangles))
        self.assertAlmostEqual(abs(stl_fmt.signed_volume(reparsed)), CUT_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.stl")
            self.assertEqual(backend.write_stl(path), len(triangles))
            with open(path, "rb") as fh:
                self.assertEqual(len(stl_fmt.parse_stl(fh.read())), len(triangles))

    def test_scad_source_is_emitted_and_statically_valid(self):
        from harnesscad.domain.programs.validate import openscad_check

        backend = OpenScadBackend()
        run_ops(backend, CUT_OPS)
        source = backend.export("scad")
        self.assertIn("difference()", source)
        self.assertIn("linear_extrude", source)
        self.assertTrue(openscad_check.is_valid(source))

    def test_deterministic_replay(self):
        a, b = OpenScadBackend(), OpenScadBackend()
        run_ops(a, CUT_OPS)
        run_ops(b, CUT_OPS)
        self.assertEqual(a.state_digest(), b.state_digest())
        self.assertEqual(a.program(), b.program())
        self.assertEqual(a.export("stl"), b.export("stl"))

    def test_ops_openscad_cannot_honour_are_refused_not_faked(self):
        """OpenSCAD has no 3D erosion. Rather than dilate the part with a
        minkowski() and call it a fillet, the op is refused with a typed
        diagnostic and NOTHING mutates (block-and-correct)."""
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        for op in ({"op": "fillet", "edges": [], "radius": 1.0},
                   {"op": "chamfer", "edges": [], "distance": 1.0},
                   {"op": "draft", "faces": [], "angle": 5.0, "neutral_plane": "XY"},
                   {"op": "loft", "sketches": ["sk1"], "ruled": False},
                   {"op": "sweep", "sketch": "sk1", "path": "sk1"}):
            result = backend.apply(parse_op(op))
            self.assertFalse(result.ok, op["op"])
            self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertEqual(backend.state_digest(), before)

    def test_bad_reference_still_blocks_and_corrects(self):
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "extrude", "sketch": "sk99",
                                         "distance": 1.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-ref")
        self.assertEqual(backend.state_digest(), before)


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class OpenScadVsFRepTest(unittest.TestCase):
    """The two backends must be the same MODEL, and differ only in KERNEL."""

    def test_same_query_surface_as_frep(self):
        scad, sdf = OpenScadBackend(), FRepBackend()
        run_ops(scad, CUT_OPS)
        run_ops(sdf, CUT_OPS)
        # the op-level projections are identical, value for value
        self.assertEqual(scad.query("summary"), sdf.query("summary"))
        self.assertEqual(scad.query("sketch_dof"), sdf.query("sketch_dof"))
        self.assertEqual(scad.query("assembly"), sdf.query("assembly"))
        # the geometric projections carry identical KEYS (the values are the two
        # kernels' answers, and are allowed -- required -- to differ)
        for what in ("measure", "metrics", "validity", "mesh", "mass_properties"):
            self.assertEqual(sorted(scad.query(what)), sorted(sdf.query(what)), what)
        for what in ("validity", "measure"):
            self.assertTrue(scad.query(what))
        self.assertEqual(scad.query("validity")["is_valid"],
                         sdf.query("validity")["is_valid"])
        self.assertEqual(scad.query("nonsense"), sdf.query("nonsense"))

    def test_openscad_is_exact_where_frep_is_grid_limited(self):
        """The headline comparison.

        Both backends cut the same hole out of the same plate. OpenSCAD's answer
        is the closed-form volume of the CSG result. frep's answer is the volume
        of a marching-cubes mesh of a sampled field, so it carries a grid error of
        a fraction of a percent -- real, bounded, and strictly larger than
        OpenSCAD's.
        """
        scad, sdf = OpenScadBackend(), FRepBackend()
        run_ops(scad, CUT_OPS)
        run_ops(sdf, CUT_OPS)
        v_scad = scad.query("measure")["volume"]
        v_frep = sdf.query("measure")["volume"]

        scad_error = abs(v_scad - CUT_VOLUME) / CUT_VOLUME
        frep_error = abs(v_frep - CUT_VOLUME) / CUT_VOLUME
        self.assertLess(scad_error, 1e-6, "openscad must be exact, not merely close")
        self.assertGreater(frep_error, scad_error * 100.0,
                           "frep's grid error must dominate openscad's float32 error")
        self.assertLess(frep_error, 0.02, "frep should still be within 2% of truth")


# --------------------------------------------------------------------------
# The field census.
#
# The bug this suite exists to prevent: an op declares a field, the backend
# accepts the op, ignores the field, and returns a valid-looking solid that is a
# DIFFERENT PART. Four sibling backends all shipped it (cadquery's fillet ignored
# `edges` and rounded all 12; freecad's hole ignored `kind` and cut the same
# cylinder for a plain hole, a counterbore and a countersink). The differential
# oracle could not see any of it, because they all dropped the SAME fields and so
# agreed with each other perfectly while all being wrong.
#
# So every field of every op gets its own test, and each one asserts ONE of:
#   HONOURED -- changing the field changes the geometry, by a number we can name;
#   REFUSED  -- the op is rejected with a typed diagnostic and nothing mutates.
# Silence -- accepting the field and dropping it -- is what fails.
# --------------------------------------------------------------------------

# A 40 x 40 x 10 plate: room for a 6 mm hole with a 12 mm counterbore.
BLOCK_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
    {"op": "extrude", "sketch": "sk1", "distance": 10.0},
]
BLOCK_VOLUME = 40.0 * 40.0 * 10.0

# The brief's shell case: 60 x 40 x 20, t = 3, closed.
SHELL_BOX_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 60.0, "h": 40.0},
    {"op": "extrude", "sketch": "sk1", "distance": 20.0},
]
BOX_VOLUME = 60.0 * 40.0 * 20.0
#: Closed shell: the box minus its cavity, inset 3 on all six faces.
SHELL_CLOSED_VOLUME = BOX_VOLUME - 54.0 * 34.0 * 14.0            # == 22296
#: Open-top shell: the cavity runs out through +Z, so it is 3 mm deeper.
SHELL_OPEN_TOP_VOLUME = BOX_VOLUME - 54.0 * 34.0 * 17.0          # == 16788

BASE_HOLE = {"op": "hole", "face_or_sketch": "", "x": 20.0, "y": 20.0,
             "diameter": 6.0, "through": True, "kind": "simple"}


def ngon_area(r: float, segments: int = DEFAULT_SEGMENTS) -> float:
    n = facets.get_fragments_from_r(r, fn=float(segments))
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n)


def apply_all(backend, ops):
    """Apply ops one at a time, returning the LAST ApplyResult (ok or not)."""
    result = None
    for o in ops:
        result = backend.apply(parse_op(o))
        if not result.ok:
            return result
    return result


def volume_of(ops) -> float:
    backend = OpenScadBackend()
    run_ops(backend, ops)
    return backend.query("measure")["volume"]


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class HoleFieldTest(unittest.TestCase):
    """Every field of Hole is HONOURED -- including the three that FreeCAD dropped.

    FreeCAD's blend/hole path collapsed 'simple', 'counterbore' and 'countersink'
    into the same cylinder. OpenSCAD can express all three EXACTLY (a counterbore
    is a stacked cylinder; a countersink is cylinder(r1=, r2=), a real cone), so
    there is no excuse to drop them, and these tests assert the closed-form volume
    of the material each one removes -- not merely that they differ.
    """

    def test_kind_changes_the_part_all_three_are_distinct(self):
        simple = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="simple")])
        cbore = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="counterbore",
                                            cbore_diameter=12.0, cbore_depth=4.0)])
        csk = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="countersink",
                                          csk_diameter=12.0, csk_angle=90.0)])
        self.assertNotAlmostEqual(simple, cbore, places=3)
        self.assertNotAlmostEqual(simple, csk, places=3)
        self.assertNotAlmostEqual(cbore, csk, places=3)
        # and each removes strictly MORE than the plain bore
        self.assertLess(cbore, simple)
        self.assertLess(csk, simple)

    def test_simple_hole_removes_exactly_the_ngon_prism(self):
        v = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="simple")])
        expected = BLOCK_VOLUME - ngon_area(3.0) * 10.0
        self.assertAlmostEqual(v, expected, delta=STL_FLOAT32_TOLERANCE)

    def test_counterbore_removes_exactly_the_stacked_bore(self):
        """cbore_diameter and cbore_depth are HONOURED, to closed form.

        The extra material is the annulus between the 12 mm bore and the 6 mm
        shaft, 4 mm deep -- as n-gon prisms, since that is what OpenSCAD builds.
        """
        v = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="counterbore",
                                        cbore_diameter=12.0, cbore_depth=4.0)])
        expected = (BLOCK_VOLUME - ngon_area(3.0) * 10.0
                    - (ngon_area(6.0) - ngon_area(3.0)) * 4.0)
        self.assertAlmostEqual(v, expected, delta=STL_FLOAT32_TOLERANCE)

    def test_cbore_diameter_and_cbore_depth_each_move_the_geometry(self):
        base = dict(BASE_HOLE, kind="counterbore", cbore_diameter=12.0, cbore_depth=4.0)
        wider = volume_of(BLOCK_OPS + [dict(base, cbore_diameter=16.0)])
        deeper = volume_of(BLOCK_OPS + [dict(base, cbore_depth=6.0)])
        nominal = volume_of(BLOCK_OPS + [base])
        self.assertLess(wider, nominal)     # a wider bore removes more
        self.assertLess(deeper, nominal)    # a deeper bore removes more

    def test_countersink_removes_exactly_the_cone(self):
        """csk_diameter and csk_angle are HONOURED, to closed form.

        csk_angle is the FULL included angle, so a 90 deg countersink opening from
        6 mm to 12 mm is exactly 3 mm deep. The removed solid is the n-gon frustum
        between r=3 and r=6, minus the shaft already bored through it.
        """
        v = volume_of(BLOCK_OPS + [dict(BASE_HOLE, kind="countersink",
                                        csk_diameter=12.0, csk_angle=90.0)])
        h = countersink_depth(6.0, 12.0, 90.0)
        self.assertAlmostEqual(h, 3.0, places=9)
        n = facets.get_fragments_from_r(6.0, fn=float(DEFAULT_SEGMENTS))
        k = 0.5 * n * math.sin(2.0 * math.pi / n)          # n-gon area coefficient
        frustum = k * h * (3.0 ** 2 + 3.0 * 6.0 + 6.0 ** 2) / 3.0
        expected = (BLOCK_VOLUME - ngon_area(3.0) * 10.0
                    - (frustum - ngon_area(3.0) * h))
        self.assertAlmostEqual(v, expected, delta=STL_FLOAT32_TOLERANCE)

    def test_csk_angle_is_honoured_a_shallower_cone_removes_less(self):
        base = dict(BASE_HOLE, kind="countersink", csk_diameter=12.0)
        v90 = volume_of(BLOCK_OPS + [dict(base, csk_angle=90.0)])
        v60 = volume_of(BLOCK_OPS + [dict(base, csk_angle=60.0)])
        # a 60 deg included angle is a NARROWER, hence DEEPER, cone: more material
        self.assertGreater(countersink_depth(6.0, 12.0, 60.0),
                           countersink_depth(6.0, 12.0, 90.0))
        self.assertLess(v60, v90)

    def test_underspecified_stepped_hole_is_REFUSED_not_silently_a_plain_bore(self):
        """The FreeCAD bug, refused at the door.

        A counterbore with no cbore_diameter is not a hole -- it is an
        underspecified request. Inventing a 'conventional ratio' would make this
        backend disagree with every other one for reasons nobody wrote down, and
        falling back to a plain cylinder is how three ops became one part.
        """
        for op in (dict(BASE_HOLE, kind="counterbore"),
                   dict(BASE_HOLE, kind="counterbore", cbore_diameter=12.0),
                   dict(BASE_HOLE, kind="counterbore", cbore_depth=4.0),
                   dict(BASE_HOLE, kind="countersink")):
            backend = OpenScadBackend()
            run_ops(backend, BLOCK_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op(op))
            self.assertFalse(result.ok, op)
            self.assertEqual(result.diagnostics[0].code, "bad-value")
            self.assertEqual(backend.state_digest(), before)  # nothing mutated

    def test_incoherent_stepped_hole_is_refused(self):
        cases = [
            dict(BASE_HOLE, kind="counterbore", cbore_diameter=4.0, cbore_depth=4.0),
            dict(BASE_HOLE, kind="counterbore", cbore_diameter=12.0, cbore_depth=-1.0),
            dict(BASE_HOLE, kind="countersink", csk_diameter=3.0),
            dict(BASE_HOLE, kind="countersink", csk_diameter=12.0, csk_angle=200.0),
        ]
        for op in cases:
            backend = OpenScadBackend()
            run_ops(backend, BLOCK_OPS)
            result = backend.apply(parse_op(op))
            self.assertFalse(result.ok, op)
            self.assertEqual(result.diagnostics[0].code, "bad-value")

    def test_diameter_x_y_depth_and_through_are_all_honoured(self):
        nominal = volume_of(BLOCK_OPS + [BASE_HOLE])
        bigger = volume_of(BLOCK_OPS + [dict(BASE_HOLE, diameter=10.0)])
        self.assertLess(bigger, nominal)                       # diameter

        blind = volume_of(BLOCK_OPS + [dict(BASE_HOLE, through=False, depth=4.0)])
        self.assertAlmostEqual(blind, BLOCK_VOLUME - ngon_area(3.0) * 4.0,
                               delta=STL_FLOAT32_TOLERANCE)    # through + depth
        self.assertGreater(blind, nominal)

        deeper = volume_of(BLOCK_OPS + [dict(BASE_HOLE, through=False, depth=8.0)])
        self.assertLess(deeper, blind)                         # depth

        # x / y: move the bore to a corner so it is only half in the material
        corner = volume_of(BLOCK_OPS + [dict(BASE_HOLE, x=0.0, y=20.0)])
        self.assertAlmostEqual(corner, BLOCK_VOLUME - ngon_area(3.0) * 10.0 / 2.0,
                               delta=1e-2)
        self.assertGreater(corner, nominal)

    def test_face_or_sketch_is_honoured_it_selects_the_bore_axis(self):
        """The datum a hole is cut from decides which way it points."""
        on_xy = volume_of(BLOCK_OPS + [dict(BASE_HOLE, face_or_sketch="")])
        yz_ops = BLOCK_OPS + [
            {"op": "new_sketch", "plane": "YZ"},
            dict(BASE_HOLE, face_or_sketch="sk2", x=20.0, y=5.0),
        ]
        on_yz = volume_of(yz_ops)
        # a bore through 10 mm of plate (Z) removes far less than one through
        # 40 mm of it (X) -- so the field cannot have been ignored
        self.assertAlmostEqual(on_xy, BLOCK_VOLUME - ngon_area(3.0) * 10.0,
                               delta=STL_FLOAT32_TOLERANCE)
        # a 40 mm bore is four times the cut surface, so it carries four times
        # the float32 STL noise -- still 1e-8 RELATIVE, which is the whole budget
        expected_yz = BLOCK_VOLUME - ngon_area(3.0) * 40.0
        self.assertLess(abs(on_yz - expected_yz) / expected_yz, 1e-7)
        self.assertLess(on_yz, on_xy)

    def test_a_face_selector_datum_picks_the_FACE_the_bore_enters(self):
        """'<Z' must bore from the BOTTOM. frep silently bores from the top.

        For a blind hole those are different parts. frep reads face_or_sketch only
        when it names a sketch and falls back to "the top, along Z" otherwise, so
        '<Z' was being dropped -- caught by the field-liveness oracle, and fixed
        here rather than explained away.
        """
        blind = dict(BASE_HOLE, through=False, depth=4.0, diameter=8.0)
        top = OpenScadBackend()
        run_ops(top, BLOCK_OPS + [dict(blind, face_or_sketch=">Z")])
        bottom = OpenScadBackend()
        run_ops(bottom, BLOCK_OPS + [dict(blind, face_or_sketch="<Z")])
        # equal volume by symmetry -- so VOLUME alone cannot tell them apart, and
        # the centre of mass is what proves the field was read
        self.assertAlmostEqual(top.query("measure")["volume"],
                               bottom.query("measure")["volume"],
                               delta=STL_FLOAT32_TOLERANCE)
        z_top = top.query("mass_properties")["center_of_mass"][2]
        z_bottom = bottom.query("mass_properties")["center_of_mass"][2]
        self.assertLess(z_top, 5.0)      # material removed from the top
        self.assertGreater(z_bottom, 5.0)  # material removed from the bottom
        self.assertNotAlmostEqual(z_top, z_bottom, places=3)
        # the default datum is the top
        default = OpenScadBackend()
        run_ops(default, BLOCK_OPS + [dict(blind, face_or_sketch="")])
        self.assertAlmostEqual(
            default.query("mass_properties")["center_of_mass"][2], z_top, places=6)

    def test_a_counterbore_sits_on_the_face_its_datum_names(self):
        cbore = dict(BASE_HOLE, kind="counterbore",
                     cbore_diameter=12.0, cbore_depth=4.0)
        top = OpenScadBackend()
        run_ops(top, BLOCK_OPS + [dict(cbore, face_or_sketch=">Z")])
        bottom = OpenScadBackend()
        run_ops(bottom, BLOCK_OPS + [dict(cbore, face_or_sketch="<Z")])
        self.assertLess(top.query("mass_properties")["center_of_mass"][2], 5.0)
        self.assertGreater(bottom.query("mass_properties")["center_of_mass"][2], 5.0)

    def test_a_datum_needing_real_topology_is_REFUSED(self):
        """A datum that does not name ONE axis-aligned face is refused, typed.

        OpenSCAD has no topological faces, so '%CYLINDER' names nothing it has.
        What must never happen is the historical behaviour: accept it, fall back to
        'the top, along Z', and bore through a face nobody asked for.
        """
        for datum in ("%CYLINDER", "|Z and >Y", "not-a-selector"):
            backend = OpenScadBackend()
            run_ops(backend, BLOCK_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op(dict(BASE_HOLE, face_or_sketch=datum)))
            self.assertFalse(result.ok, datum)
            self.assertIn(result.diagnostics[0].code,
                          ("unsupported-op", "bad-value"), datum)
            self.assertEqual(backend.state_digest(), before)

    def test_bad_hole_kind_is_refused(self):
        backend = OpenScadBackend()
        run_ops(backend, BLOCK_OPS)
        result = backend.apply(parse_op(dict(BASE_HOLE, kind="tapped")))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-value")

    def test_a_counterbored_plate_is_a_valid_watertight_solid(self):
        backend = OpenScadBackend()
        run_ops(backend, BLOCK_OPS + [dict(BASE_HOLE, kind="counterbore",
                                           cbore_diameter=12.0, cbore_depth=4.0)])
        validity = backend.query("validity")
        self.assertTrue(validity["is_valid"])
        self.assertTrue(validity["watertight"])
        self.assertEqual(validity["genus"], 1)


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class ShellFieldTest(unittest.TestCase):
    """Shell is IMPLEMENTED for a prism, exactly, and REFUSED for anything else.

    The erosion of an extruded profile is the 2D offset of that profile -- and
    offset() is a real, exact OpenSCAD operator. So there is no reason to refuse
    the prism case, and every reason to refuse the general one: OpenSCAD has no 3D
    erosion at all, and faking it with minkowski() would GROW the part.
    """

    def test_closed_shell_has_EXACTLY_the_analytic_volume(self):
        """The brief's number: 60x40x20 at t=3 is 48000 - 54*34*14 = 22296."""
        v = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [], "thickness": 3.0}])
        self.assertEqual(SHELL_CLOSED_VOLUME, 22296.0)
        self.assertAlmostEqual(v, 22296.0, delta=STL_FLOAT32_TOLERANCE)

    def test_a_closed_shell_is_a_hollow_solid_and_keeps_its_bounding_box(self):
        backend = OpenScadBackend()
        run_ops(backend, SHELL_BOX_OPS + [{"op": "shell", "faces": [],
                                           "thickness": 3.0}])
        measure = backend.query("measure")
        # a shell REMOVES material: the outer surface is untouched. (frep's bug
        # was iq's two-sided 'onion', which dilated the part.)
        self.assertEqual([round(v, 6) for v in measure["bbox"]], [60.0, 40.0, 20.0])
        validity = backend.query("validity")
        self.assertTrue(validity["watertight"])
        # two closed surfaces (the wall and the void) -> chi = 4, genus = -1.
        # A SOLID box would be genus 0: this asserts the cavity actually exists.
        self.assertEqual(validity["euler_characteristic"], 4)
        self.assertEqual(validity["genus"], -1)

    def test_faces_is_HONOURED_an_open_top_is_not_a_closed_box(self):
        closed = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [],
                                             "thickness": 3.0}])
        opened = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": ["top"],
                                             "thickness": 3.0}])
        self.assertAlmostEqual(closed, SHELL_CLOSED_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertAlmostEqual(opened, SHELL_OPEN_TOP_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        # the open box is exactly one 54 x 34 x 3 lid lighter
        self.assertAlmostEqual(closed - opened, 54.0 * 34.0 * 3.0,
                               delta=STL_FLOAT32_TOLERANCE)

    def test_opening_the_bottom_is_not_the_same_as_opening_the_top(self):
        top = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": ["top"],
                                          "thickness": 3.0}])
        bottom = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": ["bottom"],
                                             "thickness": 3.0}])
        both = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": ["top", "bottom"],
                                           "thickness": 3.0}])
        # same volume by symmetry, but they are different SOLIDS -- so assert on
        # the centroid, which is what tells them apart
        self.assertAlmostEqual(top, bottom, delta=STL_FLOAT32_TOLERANCE)
        self.assertLess(both, top)   # an open tube removes both lids
        a = OpenScadBackend()
        run_ops(a, SHELL_BOX_OPS + [{"op": "shell", "faces": ["top"],
                                     "thickness": 3.0}])
        b = OpenScadBackend()
        run_ops(b, SHELL_BOX_OPS + [{"op": "shell", "faces": ["bottom"],
                                     "thickness": 3.0}])
        za = a.query("mass_properties")["center_of_mass"][2]
        zb = b.query("mass_properties")["center_of_mass"][2]
        self.assertLess(za, zb)   # an open TOP puts the mass low

    def test_thickness_is_honoured(self):
        t3 = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [], "thickness": 3.0}])
        t5 = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [], "thickness": 5.0}])
        self.assertAlmostEqual(t5, BOX_VOLUME - 50.0 * 30.0 * 10.0,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertGreater(t5, t3)   # a thicker wall leaves more material

    def test_kind_is_HONOURED_arc_and_intersection_are_different_joins(self):
        """CadQuery's two shell joins map onto OpenSCAD's two offset modes.

        'intersection' extends the cavity's offset sides to their intersection --
        a sharp corner -- which is offset(delta=). 'arc' rolls a radius around the
        corner, which is offset(r=). On a convex profile they coincide, so this
        uses an L (a reflex corner), where they must not.
        """
        ell = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
             "w": 60.0, "h": 20.0},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
             "w": 20.0, "h": 60.0},
            {"op": "extrude", "sketch": "sk1", "distance": 20.0},
        ]
        arc = volume_of(ell + [{"op": "shell", "faces": [], "thickness": 3.0,
                                "kind": "arc"}])
        sharp = volume_of(ell + [{"op": "shell", "faces": [], "thickness": 3.0,
                                  "kind": "intersection"}])
        self.assertNotAlmostEqual(arc, sharp, places=3)
        # the sharp join is the exact straight-skeleton inset: the L's area is
        # 2000, its 3 mm inset 1316, and the cavity is 14 tall.
        self.assertAlmostEqual(sharp, 2000.0 * 20.0 - 1316.0 * 14.0,
                               delta=STL_FLOAT32_TOLERANCE)

    def test_the_CANONICAL_cadquery_selector_spelling_works(self):
        """``ops.Shell`` documents its faces as CadQuery selector strings.

        A shell written exactly as our own schema documents it (">Z") must build.
        The named vocabulary ("top") is an accepted ALIAS, not the only spelling --
        the two must be the same part.
        """
        selector = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [">Z"],
                                               "thickness": 3.0}])
        alias = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": ["top"],
                                            "thickness": 3.0}])
        self.assertAlmostEqual(selector, SHELL_OPEN_TOP_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertAlmostEqual(selector, alias, delta=STL_FLOAT32_TOLERANCE)
        # and '<Z' is the OTHER cap, not the same one
        both = volume_of(SHELL_BOX_OPS + [{"op": "shell", "faces": [">Z", "<Z"],
                                           "thickness": 3.0}])
        self.assertAlmostEqual(both, BOX_VOLUME - 54.0 * 34.0 * 20.0,
                               delta=STL_FLOAT32_TOLERANCE)

    def test_a_selector_needing_real_topology_is_REFUSED(self):
        """'|Z' names edges. OpenSCAD has none -- so it is refused, typed."""
        for sel in ("|Z and >Y", "%PLANE", "not-a-selector"):
            backend = OpenScadBackend()
            run_ops(backend, SHELL_BOX_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op({"op": "shell", "faces": [sel],
                                             "thickness": 3.0}))
            self.assertFalse(result.ok, sel)
            self.assertIn(result.diagnostics[0].code,
                          ("unsupported-op", "bad-value"), sel)
            self.assertEqual(backend.state_digest(), before)

    def test_shell_of_a_non_prism_is_REFUSED_not_approximated(self):
        """A revolve is not an extrusion, so its erosion is not a 2D offset.

        This is the honest limit of the implementation, and it is stated rather
        than papered over.
        """
        backend = OpenScadBackend()
        run_ops(backend, [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 10.0, "y": 0.0,
             "w": 4.0, "h": 6.0},
            {"op": "revolve", "sketch": "sk1", "axis": [0, 0, 0, 0, 1, 0],
             "angle": 360.0},
        ])
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "shell", "faces": [],
                                         "thickness": 1.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertIn("prism", result.diagnostics[0].message)
        self.assertEqual(backend.state_digest(), before)

    def test_opening_a_side_wall_is_REFUSED_rather_than_ignored(self):
        """'+x' is not a cap of a Z-extruded prism. Ignoring the name and handing
        back a CLOSED box would be the silent-drop bug."""
        backend = OpenScadBackend()
        run_ops(backend, SHELL_BOX_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "shell", "faces": ["right"],
                                         "thickness": 3.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertEqual(backend.state_digest(), before)

    def test_a_thickness_that_leaves_no_cavity_is_refused(self):
        """A 'shell' that silently returns the solid box is the worst outcome."""
        for thickness in (10.0, 25.0):
            backend = OpenScadBackend()
            run_ops(backend, SHELL_BOX_OPS)
            result = backend.apply(parse_op({"op": "shell", "faces": [],
                                             "thickness": thickness}))
            self.assertFalse(result.ok, thickness)
            self.assertEqual(result.diagnostics[0].code, "bad-value")

    def test_an_unknown_join_kind_is_refused(self):
        backend = OpenScadBackend()
        run_ops(backend, SHELL_BOX_OPS)
        result = backend.apply(parse_op({"op": "shell", "faces": [],
                                         "thickness": 3.0, "kind": "chamfer"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-value")


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class BlendFieldTest(unittest.TestCase):
    """Fillet and Chamfer are REFUSED -- and that is the correct answer.

    OpenSCAD is a CSG language with NO topological entities: there is no edge to
    select, so `edges=("|Z",)` names something that does not exist. And there is
    no 3D erosion (offset() is 2D-only; minkowski() is a dilation with no
    inverse), so even `edges=()` -- 'round everything' -- is not expressible:
    minkowski() with a sphere GROWS a 20x10x5 box to 22x12x7. Shipping that as a
    fillet is exactly the bug the other four backends shipped.
    """

    def test_fillet_is_refused_for_every_edge_selector_including_the_empty_one(self):
        for edges in ([], ["|Z"], [">Z"], ["|Z and >Y"]):
            backend = OpenScadBackend()
            run_ops(backend, BLOCK_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op({"op": "fillet", "edges": edges,
                                             "radius": 2.0}))
            self.assertFalse(result.ok, edges)
            self.assertEqual(result.diagnostics[0].code, "unsupported-op")
            self.assertEqual(backend.state_digest(), before)

    def test_the_refusal_says_WHY_naming_the_missing_capability(self):
        backend = OpenScadBackend()
        run_ops(backend, BLOCK_OPS)
        msg = backend.apply(parse_op({"op": "fillet", "edges": ["|Z"],
                                      "radius": 2.0})).diagnostics[0].message
        self.assertIn("edges", msg)
        self.assertIn("minkowski", msg)

    def test_chamfer_is_refused_for_every_selector_and_both_setbacks(self):
        for op in ({"op": "chamfer", "edges": [], "distance": 2.0},
                   {"op": "chamfer", "edges": ["|Z"], "distance": 2.0},
                   {"op": "chamfer", "edges": [], "distance": 2.0, "distance2": 4.0}):
            backend = OpenScadBackend()
            run_ops(backend, BLOCK_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op(op))
            self.assertFalse(result.ok, op)
            self.assertEqual(result.diagnostics[0].code, "unsupported-op")
            self.assertEqual(backend.state_digest(), before)


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class TessellationAndCacheKeyTest(unittest.TestCase):
    """$fn is PINNED, and it is IN THE CACHE KEY -- along with the tool version.

    An unpinned $fn means a cylinder's volume depends on whatever $fa/$fs the
    installed binary defaults to, and every downstream measurement inherits it. A
    cache key that omits $fn (or the tool version) re-serves the geometry a
    DIFFERENT tessellation -- or a different kernel build -- produced. Blender's
    and FreeCAD's keys both omitted the tool version and silently served stale
    meshes across an upgrade.
    """

    def test_every_curved_entity_carries_an_explicit_fn(self):
        """Nothing is left to OpenSCAD's $fa=12 / $fs=2 defaults."""
        backend = OpenScadBackend()
        run_ops(backend, CUT_OPS)
        source = backend.export("scad")
        for line in source.splitlines():
            stripped = line.strip()
            for curved in ("circle(", "cylinder(", "rotate_extrude(", "offset("):
                if stripped.startswith(curved):
                    self.assertIn("$fn", stripped,
                                  "unpinned tessellation: %s" % stripped)

    def test_fn_is_stamped_into_the_program_text(self):
        source = OpenScadBackend(segments=32)
        run_ops(source, PLATE_OPS)
        self.assertIn("$fn (32)", source.program())

    def test_fn_changes_the_geometry_and_the_digest_and_the_program(self):
        coarse, fine = OpenScadBackend(segments=8), OpenScadBackend(segments=64)
        run_ops(coarse, CUT_OPS)
        run_ops(fine, CUT_OPS)
        # the same op stream, two different solids -- because it IS two solids
        self.assertNotEqual(coarse.program(), fine.program())
        self.assertNotEqual(coarse.state_digest(), fine.state_digest())
        self.assertNotAlmostEqual(coarse.query("measure")["volume"],
                                  fine.query("measure")["volume"], places=2)
        # an 8-gon bore removes less than a 64-gon one, so the plate is heavier
        self.assertGreater(coarse.query("measure")["volume"],
                           fine.query("measure")["volume"])

    def test_the_openscad_version_is_in_the_program_and_the_digest(self):
        """The stale-geometry-across-upgrade bug, closed.

        The version is in the program TEXT, and the program text is what names
        the content-addressed cache directory -- so a new OpenSCAD build cannot
        hit the previous build's STL.
        """
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        version = backend.tool_version()
        self.assertTrue(version)
        self.assertNotEqual(version, "unknown")
        self.assertIn(version, backend.program())

        digest = backend.state_digest()
        original = type(backend)._VERSIONS.get(backend.executable)
        try:
            type(backend)._VERSIONS[backend.executable] = "OpenSCAD version 2099.99"
            self.assertNotEqual(backend.state_digest(), digest)
        finally:
            type(backend)._VERSIONS[backend.executable] = original

    def test_the_digest_separates_two_models_that_differ_only_in_a_dropped_field(self):
        """The regression guard for the whole class of bug.

        A plain hole and a counterbore must not share a digest. If they ever do,
        a field has been dropped again.
        """
        simple, cbore = OpenScadBackend(), OpenScadBackend()
        run_ops(simple, BLOCK_OPS + [dict(BASE_HOLE, kind="simple")])
        run_ops(cbore, BLOCK_OPS + [dict(BASE_HOLE, kind="counterbore",
                                         cbore_diameter=12.0, cbore_depth=4.0)])
        self.assertNotEqual(simple.state_digest(), cbore.state_digest())
        self.assertNotEqual(simple.program(), cbore.program())
        self.assertNotEqual(simple.export("stl"), cbore.export("stl"))


@unittest.skipUnless(HAVE_OPENSCAD, REASON)
class FailureAndStalenessTest(unittest.TestCase):
    """A failed run RAISES. It never re-serves, and never leaves, a stale file.

    Blender's traceback exited 0, and the caller only checked that the output file
    existed -- so a crashed build re-served the previous STL and reported success.
    """

    def test_a_run_that_produces_no_geometry_raises(self):
        backend = OpenScadBackend()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "model.stl")
            # a syntactically VALID script with an empty top-level object: this is
            # OpenSCAD's signature failure, and it exits ZERO
            with self.assertRaises(OpenScadError):
                backend._run("// no geometry at all\n", tmp, out)
            self.assertFalse(os.path.exists(out))

    def test_a_failed_run_deletes_the_stale_artefact_it_would_otherwise_re_serve(self):
        backend = OpenScadBackend()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "model.stl")
            with open(out, "wb") as fh:
                fh.write(b"a stale STL from a previous, different model")
            with self.assertRaises(OpenScadError):
                backend._run("// no geometry at all\n", tmp, out)
            # the cache is a file-exists check, so leaving this behind would mean
            # serving the OLD model forever and calling it a success
            self.assertFalse(os.path.exists(out))

    def test_a_syntactically_broken_program_raises(self):
        backend = OpenScadBackend()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "model.stl")
            with self.assertRaises(OpenScadError):
                backend._run("this is not openscad {{{\n", tmp, out)
            self.assertFalse(os.path.exists(out))

    def test_the_exported_mesh_is_the_mesh_we_measured(self):
        """Export and measure must not be two different geometries."""
        backend = OpenScadBackend()
        run_ops(backend, BLOCK_OPS + [dict(BASE_HOLE, kind="counterbore",
                                           cbore_diameter=12.0, cbore_depth=4.0)])
        measured = backend.query("measure")["volume"]
        exported = abs(stl_fmt.signed_volume(stl_fmt.parse_stl(backend.export("stl"))))
        self.assertAlmostEqual(measured, exported, places=9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
