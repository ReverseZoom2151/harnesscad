"""End-to-end tests for the kernel-free FRep geometry backend.

These drive a real CISP op stream through HarnessSession -> FRepBackend and
assert on real geometry: a watertight 2-manifold mesh, a volume that matches the
analytic extruded-rectangle volume, a boolean cut that actually removes
material, and an STL that round-trips through harnesscad.io.formats.stl.

Deterministic: no randomness, no wall clock, no third-party dependency.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.io.backends.base import GeometryBackend
from harnesscad.io.backends.frep import FRepBackend, eval_node
from harnesscad.io.formats import stl as stl_fmt
from harnesscad.io.surfaces.server import CISPServer

# A 20 x 10 rectangle extruded 5 -> analytic volume 1000.
PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]

# The same plate, then a d=6 circular boss extruded and cut away from it.
CUT_OPS = PLATE_OPS + [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk2", "cx": 10.0, "cy": 5.0, "r": 3.0},
    {"op": "extrude", "sketch": "sk2", "distance": 5.0},
    {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"},
]

PLATE_VOLUME = 20.0 * 10.0 * 5.0


def _apply(ops, resolution=40):
    backend = FRepBackend(resolution=resolution)
    session = HarnessSession(backend)
    result = session.apply_ops([parse_op(o) for o in ops])
    return backend, result


class FRepProtocolTest(unittest.TestCase):
    def test_implements_geometry_backend_protocol(self):
        self.assertIsInstance(FRepBackend(), GeometryBackend)

    def test_selectable_through_the_server(self):
        server = CISPServer(backend="frep")
        self.assertEqual(server.backend_name, "frep")
        self.assertIsNone(server.backend_note)
        self.assertIsInstance(server.backend, FRepBackend)

    def test_stub_backend_still_default(self):
        self.assertEqual(CISPServer().backend_name, "stub")


class FRepExtrudeTest(unittest.TestCase):
    def setUp(self):
        self.backend, self.result = _apply(PLATE_OPS)

    def test_op_stream_applies(self):
        self.assertTrue(self.result.ok, self.result.diagnostics)
        self.assertEqual(self.result.applied, len(PLATE_OPS))

    def test_produces_a_real_mesh(self):
        verts, faces = self.backend.mesh()
        self.assertGreater(len(verts), 8)
        self.assertGreater(len(faces), 12)

    def test_mesh_is_watertight_and_manifold(self):
        verts, faces = self.backend.mesh()
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])
        self.assertTrue(he.is_closed())
        self.assertEqual(he.boundary_halfedges(), [])
        self.assertEqual(he.genus(), 0)

    def test_regenerate_reports_no_manifold_diagnostics(self):
        self.assertEqual(self.backend.regenerate(), [])

    def test_validity_query(self):
        v = self.backend.query("validity")
        self.assertTrue(v["solid_present"])
        self.assertTrue(v["manifold"])
        self.assertTrue(v["watertight"])
        self.assertTrue(v["is_valid"])

    def test_volume_matches_analytic_within_grid_tolerance(self):
        vol = self.backend.query("measure")["volume"]
        self.assertAlmostEqual(vol / PLATE_VOLUME, 1.0, delta=0.02)

    def test_bbox_matches_the_sketch(self):
        bbox = self.backend.query("metrics")["bbox"]
        for got, want in zip(bbox, (20.0, 10.0, 5.0)):
            self.assertAlmostEqual(got, want, delta=0.5)

    def test_deterministic_replay(self):
        other, _ = _apply(PLATE_OPS)
        self.assertEqual(other.state_digest(), self.backend.state_digest())
        self.assertEqual(other.mesh()[1], self.backend.mesh()[1])

    def test_sdf_field_signs(self):
        field = self.backend.field()
        self.assertLess(field((10.0, 5.0, 2.5)), 0.0)     # inside
        self.assertGreater(field((50.0, 5.0, 2.5)), 0.0)  # outside


class FRepBooleanTest(unittest.TestCase):
    def test_cut_removes_material(self):
        plate, _ = _apply(PLATE_OPS)
        cut, result = _apply(CUT_OPS)
        self.assertTrue(result.ok, result.diagnostics)
        v_plate = plate.query("measure")["volume"]
        v_cut = cut.query("measure")["volume"]
        self.assertLess(v_cut, v_plate)
        # a d=6 x 5 deep through-pocket removes pi*9*5 = 141.4
        removed = v_plate - v_cut
        self.assertAlmostEqual(removed / (3.141592653589793 * 9.0 * 5.0), 1.0, delta=0.1)

    def test_cut_result_is_still_manifold(self):
        cut, _ = _apply(CUT_OPS)
        verts, faces = cut.mesh()
        ok, issues = HalfedgeMesh(verts, faces).is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])

    def test_cut_field_is_empty_in_the_pocket(self):
        cut, _ = _apply(CUT_OPS)
        field = cut.field()
        self.assertGreater(field((10.0, 5.0, 2.5)), 0.0)  # in the removed cylinder
        self.assertLess(field((1.0, 1.0, 2.5)), 0.0)      # still solid at the corner

    def test_union_keeps_both_bodies(self):
        ops = PLATE_OPS + [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk2",
             "x": 25.0, "y": 0.0, "w": 5.0, "h": 5.0},
            {"op": "extrude", "sketch": "sk2", "distance": 5.0},
            {"op": "boolean", "kind": "union", "target": "f1", "tool": "f2"},
        ]
        backend, result = _apply(ops, resolution=32)
        self.assertTrue(result.ok, result.diagnostics)
        vol = backend.query("measure")["volume"]
        self.assertGreater(vol, PLATE_VOLUME)

    def test_boolean_without_two_solids_is_rejected(self):
        backend, result = _apply(PLATE_OPS + [{"op": "boolean", "kind": "cut"}])
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[-1].code, "no-solid")


class FRepFilletTest(unittest.TestCase):
    def test_fillet_rounds_and_stays_manifold(self):
        sharp, _ = _apply(PLATE_OPS)
        rounded, result = _apply(PLATE_OPS + [{"op": "fillet", "edges": [],
                                               "radius": 1.0}])
        self.assertTrue(result.ok, result.diagnostics)
        verts, faces = rounded.mesh()
        ok, issues = HalfedgeMesh(verts, faces).is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])
        # rounding a convex solid removes material but keeps the bounding box
        self.assertLess(rounded.query("measure")["volume"],
                        sharp.query("measure")["volume"])
        for got, want in zip(rounded.query("metrics")["bbox"], (20.0, 10.0, 5.0)):
            self.assertAlmostEqual(got, want, delta=0.6)


class FRepExportTest(unittest.TestCase):
    def setUp(self):
        self.backend, _ = _apply(PLATE_OPS)

    def test_ascii_stl_round_trips_with_the_same_triangle_count(self):
        text = self.backend.export("stl")
        self.assertTrue(text.startswith("solid "))
        tris = stl_fmt.parse_stl(text.encode("utf-8"))
        self.assertEqual(len(tris), len(self.backend.mesh()[1]))

    def test_binary_stl_round_trips_with_the_same_triangle_count(self):
        data = self.backend.export("stl-binary")
        self.assertTrue(stl_fmt.is_binary_stl(data))
        tris = stl_fmt.parse_stl(data)
        self.assertEqual(len(tris), len(self.backend.mesh()[1]))

    def test_written_stl_file_round_trips(self):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "plate.stl")
        try:
            n = self.backend.write_stl(path, binary=True)
            with open(path, "rb") as fh:
                data = fh.read()
        finally:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(tmp)
        tris = stl_fmt.parse_stl(data)
        self.assertEqual(len(tris), n)
        self.assertEqual(len(tris), len(self.backend.mesh()[1]))
        # the STL's own signed volume agrees with the analytic volume
        self.assertAlmostEqual(abs(stl_fmt.signed_volume(tris)) / PLATE_VOLUME,
                               1.0, delta=0.02)

    def test_stl_surface_area_matches_analytic(self):
        tris = stl_fmt.parse_stl(self.backend.export("stl").encode("utf-8"))
        analytic = 2 * (20 * 10) + 2 * (20 * 5) + 2 * (10 * 5)  # 700
        self.assertAlmostEqual(stl_fmt.surface_area(tris) / analytic, 1.0, delta=0.05)

    def test_glb_export(self):
        data = self.backend.export("glb")
        self.assertEqual(data[:4], b"glTF")

    def test_sdf_export_is_the_frep_tree(self):
        spec = self.backend.export("sdf")
        self.assertIn('"t":"extrude"', spec)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            self.backend.export("step")


class FRepRejectionTest(unittest.TestCase):
    def test_empty_sketch_extrude_is_blocked(self):
        backend, result = _apply([
            {"op": "new_sketch", "plane": "XY"},
            {"op": "extrude", "sketch": "sk1", "distance": 5.0},
        ])
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[-1].code, "empty-sketch")
        self.assertFalse(backend.solid_present)

    def test_bad_sketch_reference_is_blocked(self):
        _, result = _apply([{"op": "extrude", "sketch": "sk9", "distance": 5.0}])
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[-1].code, "bad-ref")

    def test_zero_distance_extrude_is_blocked(self):
        _, result = _apply(PLATE_OPS[:2] + [
            {"op": "extrude", "sketch": "sk1", "distance": 0.0}])
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[-1].code, "bad-value")


class FRepHoleAndRevolveTest(unittest.TestCase):
    def test_through_hole_is_manifold_and_has_genus_one(self):
        ops = PLATE_OPS + [{"op": "hole", "face_or_sketch": "f1",
                            "x": 10.0, "y": 5.0, "diameter": 4.0, "through": True}]
        backend, result = _apply(ops, resolution=40)
        self.assertTrue(result.ok, result.diagnostics)
        verts, faces = backend.mesh()
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])
        self.assertEqual(he.genus(), 1)  # one through-hole
        self.assertLess(backend.query("measure")["volume"], PLATE_VOLUME)

    def test_revolve_makes_a_solid_of_revolution(self):
        # a 2x4 rectangle offset 3 from the Y axis, revolved 360 -> a ring:
        # volume = 2*pi*R_mean*A = 2*pi*4*(2*4) = 201.06
        ops = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1",
             "x": 3.0, "y": 0.0, "w": 2.0, "h": 4.0},
            {"op": "revolve", "sketch": "sk1",
             "axis": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "angle": 360.0},
        ]
        backend, result = _apply(ops, resolution=40)
        self.assertTrue(result.ok, result.diagnostics)
        verts, faces = backend.mesh()
        ok, issues = HalfedgeMesh(verts, faces).is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])
        analytic = 2.0 * 3.141592653589793 * 4.0 * (2.0 * 4.0)
        self.assertAlmostEqual(backend.query("measure")["volume"] / analytic,
                               1.0, delta=0.05)


class FRepNodeTest(unittest.TestCase):
    def test_eval_node_is_exposed_for_direct_sdf_use(self):
        backend, _ = _apply(PLATE_OPS)
        node = backend.root()
        self.assertIsNotNone(node)
        self.assertEqual(node.t, "extrude")
        # the SDF of a box is Euclidean: 5 units to the right of the x=20 face
        self.assertAlmostEqual(eval_node(node, (25.0, 5.0, 2.5)), 5.0, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
