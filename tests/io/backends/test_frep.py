"""End-to-end tests for the kernel-free FRep geometry backend.

These drive a real CISP op stream through HarnessSession -> FRepBackend and
assert on real geometry: a watertight 2-manifold mesh, a volume that matches the
analytic extruded-rectangle volume, a boolean cut that actually removes
material, and an STL that round-trips through harnesscad.io.formats.stl.

Deterministic: no randomness, no wall clock, no third-party dependency.
"""

from __future__ import annotations

import math
import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.domain.geometry.mesh.winding_number import signed_volume as mesh_signed_volume
from harnesscad.domain.numeric import interval_arithmetic
from harnesscad.io.backends import frep
from harnesscad.io.backends import frep_ir
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


class FRepMesherChoiceTest(unittest.TestCase):
    """marching_cubes and surface_nets are RIVALS: selectable, never blended."""

    def test_marching_cubes_is_still_the_default(self):
        backend, _ = _apply(PLATE_OPS)
        self.assertEqual(backend.mesher, "marching_cubes")
        self.assertEqual(frep.DEFAULT_MESHER, "marching_cubes")
        self.assertEqual(backend.mesh(), backend.mesh(mesher="marching_cubes"))

    def test_surface_nets_is_a_different_but_valid_mesh_of_the_same_field(self):
        backend, _ = _apply(CUT_OPS)
        mc_v, mc_f = backend.mesh(mesher="marching_cubes")
        sn_v, sn_f = backend.mesh(mesher="surface_nets")
        # different: a dual method puts one vertex per CELL, not on cell edges
        self.assertNotEqual(sorted(mc_v), sorted(sn_v))
        # valid: closed 2-manifold, and the same solid to within tessellation error
        ok, issues = HalfedgeMesh(sn_v, sn_f).is_2manifold()
        self.assertTrue(ok, [str(i) for i in issues[:5]])
        self.assertTrue(HalfedgeMesh(sn_v, sn_f).is_closed())
        expected = PLATE_VOLUME - 3.141592653589793 * 9.0 * 5.0
        volume = abs(mesh_signed_volume(sn_v, sn_f))
        self.assertAlmostEqual(volume / expected, 1.0, delta=0.05)

    def test_an_unknown_mesher_is_refused(self):
        backend, _ = _apply(PLATE_OPS)
        with self.assertRaises(ValueError):
            backend.mesh(mesher="dual_contouring")   # 2D only; not a 3D rival
        with self.assertRaises(ValueError):
            FRepBackend(mesher="nope")

    def test_the_mesher_choice_survives_a_setparam_replay(self):
        backend = FRepBackend(resolution=24, mesher="surface_nets", prune=True)
        session = HarnessSession(backend)
        session.apply_ops([parse_op(o) for o in PLATE_OPS])
        r = session.apply_ops([parse_op({"op": "set_param", "target": 1,
                                         "param": "w", "value": 30.0})])
        self.assertTrue(r.ok, r.diagnostics)
        self.assertEqual(backend.mesher, "surface_nets")
        self.assertTrue(backend.prune)

    def test_tolerance_drives_the_grid_resolution(self):
        backend, _ = _apply(PLATE_OPS)
        bounds = backend.bounds()
        coarse = frep.resolution_for_tolerance(bounds, 1.0)
        fine = frep.resolution_for_tolerance(bounds, 0.05)
        self.assertGreater(fine, coarse)
        self.assertTrue(backend.mesh(tolerance=0.5)[1])


class FRepAutodiffNormalTest(unittest.TestCase):
    """Exact (forward-AD) normals vs the finite-difference estimator."""

    def test_the_default_normal_method_is_unchanged(self):
        backend, _ = _apply(PLATE_OPS)
        self.assertEqual(backend.normals, "finite_difference")
        self.assertEqual(frep.DEFAULT_NORMALS, "finite_difference")

    def test_autodiff_normals_match_finite_difference_normals(self):
        backend, _ = _apply(CUT_OPS, resolution=24)
        verts, _faces = backend.mesh()
        self.assertTrue(verts)
        worst = 0.0
        for v in verts:
            fd = backend.normal(v, method="finite_difference")
            ad = backend.normal(v, method="autodiff")
            worst = max(worst, math.dist(fd, ad))
        self.assertLess(worst, 1e-4, "autodiff normal disagrees with the FD normal")

    def test_the_autodiff_normal_of_a_box_face_is_exact(self):
        backend, _ = _apply(PLATE_OPS)
        # a point 2 units out from the x = 20 face: the true normal is +x
        n = backend.normal((22.0, 5.0, 2.5), method="autodiff")
        self.assertAlmostEqual(n[0], 1.0, places=9)
        self.assertAlmostEqual(n[1], 0.0, places=9)
        self.assertAlmostEqual(n[2], 0.0, places=9)

    def test_the_ir_evaluates_to_the_same_field_as_the_python_tree(self):
        backend, _ = _apply(CUT_OPS, resolution=16)
        compiled = backend.ir()
        self.assertIsNotNone(compiled)
        node = backend.root()
        for p in ((0.0, 0.0, 0.0), (10.0, 5.0, 2.5), (22.0, 5.0, 2.5),
                  (10.0, 5.0, 9.0), (-3.0, 12.0, 1.0)):
            self.assertAlmostEqual(compiled.value(p), eval_node(node, p), places=9)

    def test_a_polygon_profile_is_honestly_reported_as_uncompilable(self):
        # a sketch made of LINES has a winding-number sign test, which the
        # arithmetic IR cannot express. It must say so, not fake a normal.
        ops = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_line", "sketch": "sk1", "x1": 0.0, "y1": 0.0,
             "x2": 10.0, "y2": 0.0},
            {"op": "add_line", "sketch": "sk1", "x1": 10.0, "y1": 0.0,
             "x2": 5.0, "y2": 8.0},
            {"op": "add_line", "sketch": "sk1", "x1": 5.0, "y1": 8.0,
             "x2": 0.0, "y2": 0.0},
            {"op": "extrude", "sketch": "sk1", "distance": 4.0},
        ]
        backend, result = _apply(ops, resolution=20)
        self.assertTrue(result.ok, result.diagnostics)
        self.assertIsNone(backend.ir())
        with self.assertRaises(ValueError):
            backend.normal((1.0, 1.0, 1.0), method="autodiff")
        # the finite-difference route still works, and so does meshing
        self.assertTrue(backend.mesh()[1])
        self.assertEqual(len(backend.normal((1.0, 1.0, 2.0))), 3)


class FRepIntervalPruningTest(unittest.TestCase):
    """Interval pruning must change the WORK, never the mesh."""

    def test_pruning_is_off_by_default(self):
        backend, _ = _apply(PLATE_OPS)
        self.assertFalse(backend.prune)
        stats = {}
        backend.mesh(stats=stats)
        self.assertEqual(stats["pruned_samples"], 0)
        self.assertEqual(stats["field_evals"], stats["samples"])

    def test_pruning_produces_the_identical_mesh_from_fewer_evaluations(self):
        backend, _ = _apply(CUT_OPS, resolution=32)
        plain_stats = {}
        pruned_stats = {}
        plain = backend.mesh(stats=plain_stats)
        pruned = backend.mesh(prune=True, stats=pruned_stats)

        # identical geometry: same vertices, same triangles, in the same order
        self.assertEqual(plain[0], pruned[0])
        self.assertEqual(plain[1], pruned[1])
        self.assertTrue(plain[1])

        # strictly less work: whole blocks of cells were never sampled
        self.assertEqual(plain_stats["samples"], pruned_stats["samples"])
        self.assertLess(pruned_stats["field_evals"], plain_stats["field_evals"])
        self.assertGreater(pruned_stats["blocks_pruned"], 0)
        self.assertLess(pruned_stats["blocks_pruned"], pruned_stats["blocks"])

    def test_pruning_leaves_the_measured_volume_alone(self):
        backend, _ = _apply(PLATE_OPS, resolution=24)
        plain = backend.query("measure")["volume"]
        backend.prune = True
        backend._invalidate()
        self.assertAlmostEqual(backend.query("measure")["volume"], plain, places=9)

    def test_interval_classification_is_conservative(self):
        backend, _ = _apply(PLATE_OPS)
        compiled = backend.ir()
        node = backend.root()
        # a box deep inside the plate must be FILLED, one far away EMPTY, and one
        # straddling the x = 20 face AMBIGUOUS
        self.assertEqual(frep_ir.classify_box(compiled, (8.0, 4.0, 2.0),
                                              (12.0, 6.0, 3.0)), frep_ir.FILLED)
        self.assertEqual(frep_ir.classify_box(compiled, (100.0, 100.0, 100.0),
                                              (110.0, 110.0, 110.0)), frep_ir.EMPTY)
        self.assertEqual(frep_ir.classify_box(compiled, (18.0, 4.0, 2.0),
                                              (22.0, 6.0, 3.0)), frep_ir.AMBIGUOUS)
        # and the bound really does enclose the field over the box
        box = interval_arithmetic.eval_interval(compiled.root, (0.0, 0.0, 0.0),
                                                (25.0, 12.0, 6.0))
        for p in ((0.0, 0.0, 0.0), (12.5, 6.0, 3.0), (25.0, 12.0, 6.0),
                  (20.0, 10.0, 5.0)):
            self.assertTrue(box.contains(eval_node(node, p)),
                            "the interval must enclose the true field value")


class FRepMassPropertiesTest(unittest.TestCase):
    """Gauss-quadrature mass properties vs. closed-form values."""

    def test_volume_and_inertia_of_a_box(self):
        backend, _ = _apply(PLATE_OPS, resolution=32)
        mp = backend.mass_properties(density=1.0)
        verts, faces = backend.mesh()

        # the quadrature volume is the EXACT volume of that tessellation: it must
        # agree with the divergence-theorem volume to machine precision
        self.assertAlmostEqual(mp["volume"] / abs(mesh_signed_volume(verts, faces)),
                               1.0, places=9)
        # ...and with the analytic box volume to within tessellation error
        self.assertAlmostEqual(mp["volume"] / PLATE_VOLUME, 1.0, delta=0.03)

        # centre of mass of a 20 x 10 x 5 box cornered at the origin
        for got, want in zip(mp["center_of_mass"], (10.0, 5.0, 2.5)):
            self.assertAlmostEqual(got, want, delta=0.05)

        # inertia about the centre of mass: I_xx = m (b^2 + c^2) / 12
        m = PLATE_VOLUME
        ixx, iyy, izz = mp["principal_moments"]
        self.assertAlmostEqual(ixx / (m * (10.0 ** 2 + 5.0 ** 2) / 12.0), 1.0, delta=0.05)
        self.assertAlmostEqual(iyy / (m * (20.0 ** 2 + 5.0 ** 2) / 12.0), 1.0, delta=0.05)
        self.assertAlmostEqual(izz / (m * (20.0 ** 2 + 10.0 ** 2) / 12.0), 1.0, delta=0.05)
        # a box aligned with the axes has no products of inertia
        self.assertAlmostEqual(mp["inertia_tensor"][0][1] / ixx, 0.0, places=3)
        self.assertAlmostEqual(mp["inertia_tensor"][0][2] / ixx, 0.0, places=3)

    def test_mass_scales_with_density(self):
        backend, _ = _apply(PLATE_OPS, resolution=20)
        a = backend.mass_properties(density=1.0)
        b = backend.mass_properties(density=7.8)
        self.assertAlmostEqual(b["mass"] / a["mass"], 7.8, places=6)
        self.assertAlmostEqual(b["volume"], a["volume"], places=9)

    def test_mass_properties_is_reachable_as_a_query(self):
        backend, _ = _apply(PLATE_OPS, resolution=20)
        q = backend.query("mass_properties")
        self.assertIn("inertia_tensor", q)
        self.assertEqual(len(q["inertia_tensor"]), 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
