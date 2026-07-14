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

import json
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

#: The shell case from the bug report: a 60x40x20 box hollowed to a 3 mm wall.
SHELL_BOX = (60.0, 40.0, 20.0)
SHELL_THICKNESS = 3.0
BOX_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
     "w": SHELL_BOX[0], "h": SHELL_BOX[1]},
    {"op": "extrude", "sketch": "sk1", "distance": SHELL_BOX[2]},
]
SHELL_BOX_OPS = BOX_OPS + [{"op": "shell", "faces": [],
                            "thickness": SHELL_THICKNESS}]
BOX_VOLUME = SHELL_BOX[0] * SHELL_BOX[1] * SHELL_BOX[2]
#: A closed hollow box: the outer box less the cavity inset by the wall on all six
#: sides. Exact, and only reachable when the solidify offset is -1 with even offsets.
SHELL_WALL_VOLUME = BOX_VOLUME - (
    (SHELL_BOX[0] - 2 * SHELL_THICKNESS)
    * (SHELL_BOX[1] - 2 * SHELL_THICKNESS)
    * (SHELL_BOX[2] - 2 * SHELL_THICKNESS))


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

    def test_shell_does_not_grow_the_part(self):
        """A CAD shell hollows INWARD: the outer surface does not move.

        Blender manual, Solidify > Offset: "A value between (-1 to 1) to locate the
        solidified output inside or outside the original mesh. The inside and
        outside is determined by the face normals. Set to 0.0, the solidified output
        will be centered on the original mesh."

        The backend used to leave the offset at 0.0 (centred), which pushed half the
        thickness OUTWARD along the averaged corner normals: this 60x40x20 box came
        out 61.732 x 41.732 x 21.732 at t=3 (each side gained (t/2)/sqrt(3) = 0.866).
        offset = -1.0 puts the whole thickness on the inside of the face normals.
        """
        backend = BlenderBackend()
        run_ops(backend, SHELL_BOX_OPS)
        bbox = backend.query("measure")["bbox"]
        for got, want in zip(bbox, (SHELL_BOX[0], SHELL_BOX[1], SHELL_BOX[2])):
            self.assertAlmostEqual(got, want, delta=1e-3)

    def test_shell_wall_is_exactly_the_requested_thickness(self):
        """use_even_offset ("Maintain thickness by adjusting for sharp corners") plus
        use_quality_normals ("Calculate normals which result in more even thickness").

        Without them the inner surface rides the *vertex* normal, so a box corner
        walks the diagonal and the wall comes out thin (13843.6 instead of 22296.0 on
        this part). With them the hollow-box wall volume is exact in closed form."""
        backend = BlenderBackend()
        run_ops(backend, SHELL_BOX_OPS)
        self.assertAlmostEqual(backend.query("measure")["volume"],
                               SHELL_WALL_VOLUME, delta=1e-2)

    def test_shell_is_hollow_and_watertight(self):
        backend = BlenderBackend()
        run_ops(backend, SHELL_BOX_OPS)
        validity = backend.query("validity")
        self.assertTrue(validity["watertight"])
        # An inner surface as well as an outer one: strictly more geometry than the
        # solid box, for strictly less material.
        solid = BlenderBackend()
        run_ops(solid, BOX_OPS)
        self.assertGreater(len(backend.mesh()[1]), len(solid.mesh()[1]))
        self.assertLess(backend.query("measure")["volume"],
                        solid.query("measure")["volume"])

    def test_fillet_only_touches_the_edges_the_op_names(self):
        """THE ONE THAT MATTERS. ``Fillet.edges`` must SELECT edges.

        BevelModifier.limit_method is a global heuristic, not a selection: 'ANGLE'
        only "bevels edges whose angle of adjacent face normals plus the defined
        Angle is less than 180 degrees", and 'NONE' bevels "the entire mesh by a
        constant amount". Neither can express "these four edges", so the modifier
        route silently rounded EVERY edge of the model -- boolean-cut pocket corners
        and the solidify rim included -- whatever the op asked for. bmesh.ops.bevel
        takes the edge list itself (``geom``: "Input edges and vertices"), so the op's
        selector picks exactly the BMEdges it names.

        Filleting 4 edges must therefore give a DIFFERENT part from filleting all 12.
        """
        four = BlenderBackend()
        run_ops(four, PLATE_OPS + [{"op": "fillet", "edges": ["|Z"], "radius": 1.0}])
        every = BlenderBackend()
        run_ops(every, PLATE_OPS + [{"op": "fillet", "edges": [], "radius": 1.0}])
        plain = BlenderBackend()
        run_ops(plain, PLATE_OPS)

        v_four = four.query("measure")["volume"]
        v_every = every.query("measure")["volume"]
        # Two different parts, not one. (The bug made these identical.)
        self.assertNotAlmostEqual(v_four, v_every, delta=1.0)
        # Only the 4 vertical edges lost material: r=1 on a 5 mm tall edge removes
        # (1 - pi/4) r^2 h = 0.2146 * 5 per edge, 4.292 mm^3 over four edges. The
        # mesh arc is a polygon inscribed in that quarter-circle, so it removes a
        # hair MORE than the analytic solid (4.393 at 8 segments) and never less.
        removed = PLATE_VOLUME - v_four
        analytic = 4.0 * (1.0 - math.pi / 4.0) * 5.0
        self.assertGreater(removed, analytic)
        self.assertAlmostEqual(removed, analytic, delta=0.03 * analytic)
        self.assertLess(v_every, v_four)          # all 12 edges removes strictly more
        self.assertLess(v_four, plain.query("measure")["volume"])
        # ...and the 4-edge fillet touches far less of the mesh than the 12-edge one.
        self.assertLess(len(four.mesh()[1]), len(every.mesh()[1]))

    def test_two_fillets_at_two_radii_compose(self):
        """A modifier stack with one global angle limit cannot carry two radii; a
        per-feature bmesh.ops.bevel can. Each Fillet op is its own bevel over its own
        edge set, applied in op order."""
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS + [
            {"op": "fillet", "edges": ["|Z"], "radius": 1.0},
            {"op": "fillet", "edges": [">Z"], "radius": 0.5},
        ])
        both = backend.query("measure")["volume"]
        one = BlenderBackend()
        run_ops(one, PLATE_OPS + [{"op": "fillet", "edges": ["|Z"], "radius": 1.0}])
        # The second fillet removed more material, and the part still fits its box.
        self.assertLess(both, one.query("measure")["volume"])
        for got, want in zip(backend.query("measure")["bbox"], (20.0, 10.0, 5.0)):
            self.assertAlmostEqual(got, want, delta=1e-3)

    def test_a_malformed_edge_selector_is_refused_not_ignored(self):
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "fillet", "edges": [">>Q"],
                                         "radius": 1.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-value")
        self.assertEqual(backend.state_digest(), before)

    def test_asymmetric_chamfer_is_refused_not_silently_symmetric(self):
        """Blender's bevel carries ONE width (bmesh.ops.bevel takes a single
        'offset'), so Chamfer.distance2 -- which OCCT's BRepFilletAPI_MakeChamfer
        does honour -- has no counterpart here. Refused with a typed diagnostic
        rather than quietly built symmetric."""
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "chamfer", "edges": ["|Z"],
                                         "distance": 1.0, "distance2": 2.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertEqual(backend.state_digest(), before)

    def test_chamfer_on_named_edges_is_exact(self):
        """A 1 mm chamfer on the 4 vertical edges of a 20x10x5 plate removes four
        right-triangular prisms: 4 * (d^2/2) * h = 4 * 0.5 * 5 = 10 mm^3, exactly."""
        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS
                + [{"op": "chamfer", "edges": ["|Z"], "distance": 1.0}])
        self.assertAlmostEqual(backend.query("measure")["volume"],
                               PLATE_VOLUME - 10.0, delta=1e-3)

    def test_the_build_fails_loudly_when_the_script_raises(self):
        """Blender's --python-exit-code "Set the exit-code in [0..255] to exit if a
        Python exception is raised... zero disables" -- and ZERO IS THE DEFAULT. A
        traceback in the build script used to exit 0, so a crashed build that left a
        stale file behind reported success. The runner now passes 1 and checks it."""
        from harnesscad.io.backends import blender as blender_mod

        backend = BlenderBackend()
        run_ops(backend, PLATE_OPS)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "model.stl")
            broken = blender_mod.BUILD_SCRIPT.replace(
                "def main():", "def main():\n    raise RuntimeError('boom')\n", 1)
            original = blender_mod.BUILD_SCRIPT
            blender_mod.BUILD_SCRIPT = broken
            try:
                with self.assertRaises(RuntimeError) as caught:
                    backend._run(backend.program(), tmp, out)
            finally:
                blender_mod.BUILD_SCRIPT = original
        self.assertIn("exit 1", str(caught.exception))

    def test_the_cache_key_tracks_the_blender_version(self):
        """BLENDER_PATTERNS globs for the NEWEST install, so an upgrade would swap the
        geometry kernel underneath an unchanged content-addressed cache key."""
        from harnesscad.io.backends import blender as blender_mod

        backend = BlenderBackend()
        run_ops(backend, BOX_OPS)
        program = json.loads(backend.program())
        self.assertIn("kernel", program)
        self.assertTrue(program["kernel"].lower().startswith("blender"))
        self.assertEqual(program["kernel"],
                         blender_mod.blender_version(backend.executable))

    def test_leaf_meshes_are_validated_before_the_exact_solver_sees_them(self):
        """The boolean manual: "Only Manifold meshes are guaranteed to give proper
        results", and Mesh.from_pydata does no validation at all."""
        script = _script()
        self.assertIn("bmesh.ops.remove_doubles", script)
        self.assertIn("bmesh.ops.dissolve_degenerate", script)
        self.assertIn("me.validate(", script)

    def test_boolean_uses_the_exact_solver(self):
        """BooleanModifier.solver: 'FLOAT' is the fast solver and is documented as
        "without support for overlapping geometry" -- which is precisely the case a
        CAD cut hits (a hole tool flush with a face). Only 'EXACT' ("the best results
        for coplanar faces") is CAD-correct, and self-intersecting operands (a mirror
        or a pattern whose copies touch) need use_self."""
        script = _script()
        self.assertIn('mod.solver = "EXACT"', script)
        self.assertIn("mod.use_self = True", script)
        self.assertNotIn('"FAST"', script)
        self.assertNotIn('"FLOAT"', script)

    def test_shell_uses_the_documented_inward_offset(self):
        script = _script()
        self.assertIn("mod.offset = -1.0", script)
        self.assertIn("mod.use_even_offset = True", script)
        self.assertIn("mod.use_quality_normals = True", script)

    def test_bevel_only_touches_the_edges_it_is_meant_to(self):
        """BevelModifier.limit_method='ANGLE' -- "Only bevel edges with sharp enough
        angles between faces". 'NONE' would "Bevel the entire mesh by a constant
        amount", rounding every seam of a faceted cylinder into mush.

        The plate's 12 box edges are 90 degrees and get filleted; the cylindrical
        wall of the hole is faceted at 360/n degrees per seam, far below the 30-degree
        limit, so it stays a clean cylinder. The volume a fillet removes from a box's
        edges is known in closed form: 12 edges lose (1 - pi/4) r^2 per unit length,
        and the 8 corners lose (1 - pi/6 - 3*(1 - pi/4)/... ) -- rather than re-derive
        the corner solid, this asserts the fillet (a) shrinks the part, (b) by less
        than the whole edge band, and (c) leaves the bbox alone (a fillet never grows
        a part), and cross-checks the number against CadQuery's OCCT fillet in
        BlenderVsCadQueryTest."""
        backend = BlenderBackend()
        run_ops(backend, BOX_OPS + [{"op": "fillet", "edges": [], "radius": 2.0}])
        measure = backend.query("measure")
        for got, want in zip(measure["bbox"], SHELL_BOX):
            self.assertAlmostEqual(got, want, delta=1e-3)
        self.assertLess(measure["volume"], BOX_VOLUME)
        self.assertGreater(measure["volume"], BOX_VOLUME * 0.98)

    def test_chamfer_is_a_one_segment_bevel(self):
        backend = BlenderBackend()
        run_ops(backend, BOX_OPS + [{"op": "chamfer", "edges": [], "distance": 2.0}])
        chamfered = backend.query("measure")["volume"]
        fillet = BlenderBackend()
        run_ops(fillet, BOX_OPS + [{"op": "fillet", "edges": [], "radius": 2.0}])
        # A chamfer cuts the corner straight across; a fillet of the same size leaves
        # the material under the arc, so it must remove strictly less.
        self.assertLess(chamfered, fillet.query("measure")["volume"])
        self.assertLess(chamfered, BOX_VOLUME)

    def test_glb_export_keeps_blender_z_up_and_model_scale(self):
        """bpy.ops.export_scene.gltf(export_yup=...) defaults to TRUE: the exporter
        rotates Blender's Z-up frame into glTF's Y-up one, silently turning the part
        -90 degrees about X. Every other format this backend emits is Z-up in model
        units, so the exporter is pinned to export_yup=False and the unit scale to
        1.0 -- a rotated or 1000x-scaled part is a correctness bug, not a preference.
        """
        import json as _json
        import struct as _struct

        backend = BlenderBackend()
        run_ops(backend, BOX_OPS)
        data = backend.export("glb")
        self.assertEqual(data[:4], b"glTF")
        chunk_len = _struct.unpack("<I", data[12:16])[0]
        doc = _json.loads(data[20:20 + chunk_len])
        bounds = [(a["min"], a["max"]) for a in doc["accessors"]
                  if len(a.get("min", ())) == 3]
        self.assertTrue(bounds)
        lo, hi = bounds[0]
        for i in range(3):
            self.assertAlmostEqual(hi[i] - lo[i], SHELL_BOX[i], delta=1e-3)

    def test_the_result_cache_is_keyed_on_the_build_script_too(self):
        """The on-disk cache is content-addressed on program(); if the bpy script (the
        kernel recipe) were not part of that text, fixing a modifier setting would
        leave every previously cached STL in place and the backend would keep serving
        geometry built by the old, wrong script."""
        backend = BlenderBackend()
        run_ops(backend, BOX_OPS)
        program = json.loads(backend.program())
        self.assertIn("plan", program)
        self.assertIn("script", program)
        self.assertTrue(program["script"])


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


@unittest.skipUnless(HAVE_BLENDER, REASON)
class BlenderVsCadQueryTest(unittest.TestCase):
    """The differential oracle: CadQuery is OCCT, a B-rep kernel that is right by
    construction. Where the two kernels model the same thing, they must AGREE."""

    def _pair(self, ops):
        from harnesscad.io.backends.cadquery import CadQueryBackend

        if not CadQueryBackend.available():
            self.skipTest("cadquery is not installed on this machine")
        blend, occt = BlenderBackend(), CadQueryBackend()
        run_ops(blend, ops)
        run_ops(occt, ops)
        return blend.query("measure"), occt.query("measure")

    def test_shell_bbox_agrees_with_occt(self):
        """The heart of the shell bug: OCCT's MakeThickSolid hollows inward and never
        moves the outer surface. Blender's solidify must not either."""
        blend, occt = self._pair(SHELL_BOX_OPS)
        for i in range(3):
            self.assertAlmostEqual(blend["bbox"][i], occt["bbox"][i], delta=1e-3)
            self.assertAlmostEqual(blend["bbox"][i], SHELL_BOX[i], delta=1e-3)

    def test_bevel_volume_agrees_with_occt(self):
        """Blender's bevel modifier against OCCT's BRepFilletAPI: the same edges, the
        same radius, the same material removed (to the faceting of the arc)."""
        blend, occt = self._pair(BOX_OPS
                                 + [{"op": "fillet", "edges": [], "radius": 2.0}])
        self.assertAlmostEqual(blend["volume"], occt["volume"],
                               delta=0.001 * occt["volume"])

    def test_chamfer_volume_agrees_with_occt(self):
        blend, occt = self._pair(BOX_OPS
                                 + [{"op": "chamfer", "edges": [], "distance": 2.0}])
        self.assertAlmostEqual(blend["volume"], occt["volume"],
                               delta=1e-3 * occt["volume"])

    def test_per_edge_fillet_agrees_with_occt(self):
        """The differential oracle could not catch the "fillet everything" bug,
        because every backend dropped ``Fillet.edges`` and so they agreed while all
        being wrong. Now both kernels parse the SAME selector with the SAME DSL, and
        must select the same edges: a mesh fillet's arc is a polygon, so it is very
        slightly under OCCT's true cylinder, but no more than that.

        20x10x5 plate, r=1: |Z (4 vertical edges) -> OCCT 995.708; all 12 -> 971.295.
        Two different parts -- which is the whole point.
        """
        for selector, tolerance in ((["|Z"], 0.002), ([">Z"], 0.002), ([], 0.002)):
            blend, occt = self._pair(
                PLATE_OPS + [{"op": "fillet", "edges": list(selector), "radius": 1.0}])
            self.assertAlmostEqual(blend["volume"], occt["volume"],
                                   delta=tolerance * occt["volume"],
                                   msg="selector %r" % (selector,))

    def test_two_radii_agree_with_occt(self):
        blend, occt = self._pair(PLATE_OPS + [
            {"op": "fillet", "edges": ["|Z"], "radius": 1.0},
            {"op": "fillet", "edges": [">Z"], "radius": 0.5},
        ])
        self.assertAlmostEqual(blend["volume"], occt["volume"],
                               delta=0.002 * occt["volume"])

    def test_per_edge_chamfer_agrees_with_occt_exactly(self):
        """A chamfer is flat, so there is no faceting error at all: the two kernels
        must return the same number."""
        blend, occt = self._pair(
            PLATE_OPS + [{"op": "chamfer", "edges": ["|Z"], "distance": 1.0}])
        self.assertAlmostEqual(blend["volume"], occt["volume"], delta=1e-3)

    def test_booleans_agree_with_occt(self):
        blend, occt = self._pair(CUT_OPS)
        self.assertAlmostEqual(blend["volume"], occt["volume"],
                               delta=1e-3 * occt["volume"])
        for i in range(3):
            self.assertAlmostEqual(blend["bbox"][i], occt["bbox"][i], delta=1e-3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
