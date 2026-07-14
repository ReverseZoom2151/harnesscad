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
from harnesscad.io.backends.openscad import OpenScadBackend
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
        """OpenSCAD has no 3D offset. Rather than dilate the part with a
        minkowski() and call it a fillet, the op is refused with a typed
        diagnostic and NOTHING mutates (block-and-correct)."""
        backend = OpenScadBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        for op in ({"op": "fillet", "edges": [], "radius": 1.0},
                   {"op": "chamfer", "edges": [], "distance": 1.0},
                   {"op": "shell", "faces": [], "thickness": 1.0}):
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
