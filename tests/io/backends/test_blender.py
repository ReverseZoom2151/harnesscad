"""End-to-end tests for the Blender geometry backend.

These drive a real CISP op stream through HarnessSession -> BlenderBackend ->
headless Blender (``blender --background --factory-startup --python ...``) and
assert on the mesh that comes back.

Blender's boolean modifier (``solver='EXACT'``) is a genuine mesh set operation:
a cut lands on the true intersection curve, not on a marching-cubes staircase.
So, exactly like the OpenSCAD backend, the volume removed by a cut is known in
CLOSED FORM -- the tool is a regular n-gon prism (both external backends facet
curves with OpenSCAD's $fn law, so they tessellate a circle identically) -- and
the test asserts against that number rather than a tolerance band.

Skips cleanly (unittest.skipUnless) when Blender is not installed.
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
from harnesscad.domain.programs.validate import bpy_script
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.blender import BlenderBackend
from harnesscad.io.backends.external import DEFAULT_SEGMENTS
from harnesscad.io.backends.frep import FRepBackend
from harnesscad.io.formats import stl as stl_fmt
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

HAVE_BLENDER = BlenderBackend.available()
REASON = "blender is not installed on this machine"

PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]

HOLE_RADIUS = 3.0
CUT_OPS = PLATE_OPS + [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk2", "cx": 10.0, "cy": 5.0, "r": HOLE_RADIUS},
    {"op": "extrude", "sketch": "sk2", "distance": 5.0},
    {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"},
]

PLATE_VOLUME = 20.0 * 10.0 * 5.0


def ngon_prism_volume(r: float, height: float, segments: int) -> float:
    """Exact volume of the n-gon prism a faceted circle actually sweeps."""
    n = facets.get_fragments_from_r(r, fn=float(segments))
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n) * height


HOLE_VOLUME = ngon_prism_volume(HOLE_RADIUS, 5.0, DEFAULT_SEGMENTS)
CUT_VOLUME = PLATE_VOLUME - HOLE_VOLUME

#: The mesh returns through a binary STL (float32 vertices) -- the whole error
#: budget of this backend on a 20 mm part.
STL_FLOAT32_TOLERANCE = 1e-4


def run_ops(backend, ops) -> None:
    session = HarnessSession(backend)
    result = session.apply_ops([parse_op(o) for o in ops])
    if not result.ok:
        raise AssertionError("op stream rejected: %s"
                             % [d.to_dict() for d in result.diagnostics])


class BlenderAvailabilityTest(unittest.TestCase):
    """The graceful-absence contract holds whether or not Blender is here."""

    def test_registered_in_the_backend_table(self):
        self.assertIn("blender", BACKENDS)

    def test_available_never_raises(self):
        self.assertIsInstance(BlenderBackend.available(), bool)

    def test_backend_unavailable_is_typed_and_actionable(self):
        exc = BackendUnavailable("blender", "not here", ["PATH:blender"])
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.tool, "blender")
        self.assertEqual(exc.searched, ["PATH:blender"])

    def test_the_generated_bpy_script_is_valid_python(self):
        """The script is generated, so its syntax is checked the way the repo
        already checks BlenderLLM's: statically, without running Blender. This
        holds even on a machine with no Blender at all."""
        check = bpy_script.check_syntax(_script())
        self.assertTrue(check.ok, check.error)
        calls = bpy_script.extract_calls(_script())
        self.assertTrue(any(c.op == "modifier_apply" for c in calls))
        # Every geometry call is in BlenderLLM's known vocabulary; the only other
        # bpy.ops call in the script is the glTF exporter, which is I/O, not
        # geometry, and so is (correctly) outside that vocabulary.
        geometry = [c for c in calls if c.group != "export_scene"]
        self.assertTrue(all(bpy_script.is_recognized_vocabulary(c) for c in geometry))

    def test_server_never_crashes_on_a_missing_tool(self):
        server = CISPServer(backend="blender")
        if HAVE_BLENDER:
            self.assertEqual(server.backend_name, "blender")
            self.assertIsNone(server.backend_note)
        else:
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("blender", server.backend_note)


def _script() -> str:
    from harnesscad.io.backends import blender as blender_mod

    return blender_mod.BUILD_SCRIPT


@unittest.skipUnless(HAVE_BLENDER, REASON)
class BlenderBackendTest(unittest.TestCase):

    def test_satisfies_the_geometry_backend_protocol(self):
        self.assertIsInstance(BlenderBackend(), GeometryBackend)

    def test_plate_builds_and_has_the_analytic_volume(self):
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        self.assertTrue(backend.query("summary")["solid_present"])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], PLATE_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 6) for v in measure["bbox"]], [20.0, 10.0, 5.0])

    def test_plate_mesh_is_a_watertight_2_manifold(self):
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        verts, faces = backend.mesh()
        self.assertTrue(faces)
        he = HalfedgeMesh(verts, faces)
        self.assertTrue(he.is_2manifold()[0])
        self.assertTrue(he.is_closed())
        self.assertEqual(backend.query("validity")["genus"], 0)

    def test_boolean_cut_removes_material(self):
        plate = BlenderBackend()
        run_ops(plate, PLATE_OPS)
        cut = BlenderBackend()
        run_ops(cut, CUT_OPS)
        self.assertLess(cut.query("measure")["volume"],
                        plate.query("measure")["volume"])

    def test_boolean_cut_removes_the_right_material(self):
        """Blender's EXACT solver is a real mesh boolean: the removed volume is
        the closed-form volume of the n-gon prism tool, not an approximation of
        it."""
        backend = BlenderBackend()
        run_ops(backend, CUT_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, CUT_VOLUME, delta=STL_FLOAT32_TOLERANCE)
        removed = PLATE_VOLUME - volume
        self.assertAlmostEqual(removed, HOLE_VOLUME, delta=STL_FLOAT32_TOLERANCE)

    def test_cut_mesh_is_a_watertight_genus_1_solid(self):
        backend = BlenderBackend()
        run_ops(backend, CUT_OPS)
        validity = backend.query("validity")
        self.assertTrue(validity["is_valid"])
        self.assertTrue(validity["watertight"])
        self.assertEqual(validity["genus"], 1)  # a plate with a through-hole

    def test_exported_stl_round_trips_through_io_formats_stl(self):
        backend = BlenderBackend()
        run_ops(backend, CUT_OPS)
        data = backend.export("stl")
        triangles = stl_fmt.parse_stl(data)
        self.assertTrue(triangles)
        self.assertAlmostEqual(abs(stl_fmt.signed_volume(triangles)), CUT_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        ascii_text = backend.export("stl-ascii")
        reparsed = stl_fmt.parse_ascii_stl(ascii_text)
        self.assertEqual(len(reparsed), len(triangles))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.stl")
            self.assertEqual(backend.write_stl(path), len(triangles))
            with open(path, "rb") as fh:
                self.assertEqual(len(stl_fmt.parse_stl(fh.read())), len(triangles))

    def test_glb_export_is_a_real_gltf_binary(self):
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        data = backend.export("glb")
        self.assertIsInstance(data, bytes)
        self.assertEqual(data[:4], b"glTF")

    def test_deterministic_replay(self):
        a, b = BlenderBackend(), BlenderBackend()
        run_ops(a, CUT_OPS)
        run_ops(b, CUT_OPS)
        self.assertEqual(a.state_digest(), b.state_digest())
        self.assertEqual(a.program(), b.program())
        self.assertEqual(a.export("stl"), b.export("stl"))

    def test_bad_reference_blocks_and_corrects(self):
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "extrude", "sketch": "sk99",
                                         "distance": 1.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-ref")
        self.assertEqual(backend.state_digest(), before)

    def test_shell_is_a_real_solidify(self):
        """Blender CAN honour shell (the solidify modifier), so it must -- and the
        result must be a hollow solid with less material than the plate."""
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS + [{"op": "shell", "faces": [], "thickness": 1.0}])
        volume = backend.query("measure")["volume"]
        self.assertGreater(volume, 0.0)
        self.assertLess(volume, PLATE_VOLUME)


@unittest.skipUnless(HAVE_BLENDER, REASON)
class BlenderVsFRepTest(unittest.TestCase):
    """The two backends must be the same MODEL, and differ only in KERNEL."""

    def test_same_query_surface_as_frep(self):
        blend, sdf = BlenderBackend(), FRepBackend()
        run_ops(blend, CUT_OPS)
        run_ops(sdf, CUT_OPS)
        self.assertEqual(blend.query("summary"), sdf.query("summary"))
        self.assertEqual(blend.query("sketch_dof"), sdf.query("sketch_dof"))
        self.assertEqual(blend.query("assembly"), sdf.query("assembly"))
        for what in ("measure", "metrics", "validity", "mesh", "mass_properties"):
            self.assertEqual(sorted(blend.query(what)), sorted(sdf.query(what)), what)
        self.assertEqual(blend.query("validity")["is_valid"],
                         sdf.query("validity")["is_valid"])
        self.assertEqual(blend.query("nonsense"), sdf.query("nonsense"))

    def test_blender_beats_freps_grid_on_the_cut_volume(self):
        blend, sdf = BlenderBackend(), FRepBackend()
        run_ops(blend, CUT_OPS)
        run_ops(sdf, CUT_OPS)
        blend_error = abs(blend.query("measure")["volume"] - CUT_VOLUME) / CUT_VOLUME
        frep_error = abs(sdf.query("measure")["volume"] - CUT_VOLUME) / CUT_VOLUME
        self.assertLess(blend_error, 1e-6)
        self.assertGreater(frep_error, blend_error * 100.0)
        self.assertLess(frep_error, 0.02)

    def test_blender_and_openscad_agree(self):
        """Two independent real kernels, same op stream, same faceting law: their
        volumes must agree to STL precision. That agreement is what makes either
        of them believable."""
        from harnesscad.io.backends.openscad import OpenScadBackend

        if not OpenScadBackend.available():
            self.skipTest("openscad is not installed on this machine")
        blend, scad = BlenderBackend(), OpenScadBackend()
        run_ops(blend, CUT_OPS)
        run_ops(scad, CUT_OPS)
        self.assertAlmostEqual(blend.query("measure")["volume"],
                               scad.query("measure")["volume"],
                               delta=STL_FLOAT32_TOLERANCE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
