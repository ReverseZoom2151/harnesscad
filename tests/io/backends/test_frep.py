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
from harnesscad.io.backends.frep import MESHERS, FRepBackend, eval_node
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


class FRepShellTest(unittest.TestCase):
    """A CAD shell hollows INWARD; it must never grow the part.

    The backend used Curv's two-sided shell (|f| - t/2), which dilates by t/2
    per side: a 60x40x20 box shelled at t=3 measured 63x43x23 and every
    verifier stayed silent.
    """

    BOX = [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
         "w": 60.0, "h": 40.0},
        {"op": "extrude", "sketch": "sk1", "distance": 20.0},
    ]

    def test_shell_does_not_grow_the_part(self):
        solid, _ = _apply(self.BOX)
        shelled, result = _apply(self.BOX + [{"op": "shell", "faces": [],
                                              "thickness": 3.0}])
        self.assertTrue(result.ok, result.diagnostics)
        before = solid.query("metrics")["bbox"]
        after = shelled.query("metrics")["bbox"]
        for axis, b, a in zip("XYZ", before, after):
            self.assertLessEqual(a, b + 1e-6, f"shell grew the part along {axis}")
        for got, want in zip(after, (60.0, 40.0, 20.0)):
            self.assertAlmostEqual(got, want, delta=0.6)

    def test_shell_removes_material(self):
        solid, _ = _apply(self.BOX)
        shelled, _ = _apply(self.BOX + [{"op": "shell", "faces": [],
                                         "thickness": 3.0}])
        self.assertLess(shelled.query("measure")["volume"],
                        solid.query("measure")["volume"])

    def test_shell_too_thick_leaves_the_solid_intact_not_inflated(self):
        # 60x40x5 plate, 9 mm wall: no cavity can open. The old two-sided shell
        # inflated it to 69x49x14 (volume 44941). It must stay the plate.
        tray = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0,
             "w": 60.0, "h": 40.0},
            {"op": "extrude", "sketch": "sk1", "distance": 5.0},
            {"op": "shell", "faces": [], "thickness": 9.0},
        ]
        backend, _ = _apply(tray)
        for got, want in zip(backend.query("metrics")["bbox"], (60.0, 40.0, 5.0)):
            self.assertAlmostEqual(got, want, delta=0.6)
        self.assertLess(backend.query("measure")["volume"], 60.0 * 40.0 * 5.0 * 1.02)


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
        # dual_contouring IS a 3D rival now (see FRepDualContouringTest).
        self.assertIn("dual_contouring", MESHERS)
        self.assertTrue(backend.mesh(mesher="dual_contouring")[1])
        with self.assertRaises(ValueError):
            backend.mesh(mesher="nope")
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


# ===========================================================================
# The f-rep literature audit: shell semantics, dual contouring, and the
# distance-property violations that min/max booleans introduce.
# ===========================================================================

#: 60 x 40 x 20 block -- the part from the shell bug report. Analytic V = 48000.
BLOCK_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 60.0, "h": 40.0},
    {"op": "extrude", "sketch": "sk1", "distance": 20.0},
]
BLOCK_VOLUME = 60.0 * 40.0 * 20.0


def _block(resolution=64, mesher="dual_contouring", **kw):
    backend = FRepBackend(resolution=resolution, mesher=mesher, **kw)
    for spec in BLOCK_OPS:
        assert backend.apply(parse_op(spec)).ok
    return backend


def _bbox(backend):
    return backend.query("metrics")["bbox"]


class FRepShellSemanticsTest(unittest.TestCase):
    """A CAD Shell hollows INWARD. It must never grow the part.

    ``abs(f) - t/2`` is Inigo Quilez's ``opOnion``
    (https://iquilezles.org/articles/distfunctions/) -- a *two-sided* shell,
    centred on the boundary, which keeps half the wall OUTSIDE the original
    surface and therefore dilates the solid by ``t/2`` in every direction. Curv
    ships the same operator under the name ``shell``. It is correct for graphics
    and wrong for CAD: SolidWorks/Fusion/Onshape all leave the outer faces
    exactly where they were, because a Shell only ever removes material. The CAD
    operator is ``max(f, -(f + t))``.
    """

    def test_shell_does_not_grow_the_part(self):
        # THE regression: 60x40x20 shelled at t=3 used to come out 63x43x23.
        # (The tolerance is a MESHING tolerance, not a shell one: the QEF places
        # a dual-contouring vertex within ~3e-5 of the true corner, not on it.
        # The old two-sided shell missed by 3.0 -- five orders of magnitude out.)
        backend = _block()
        before = _bbox(backend)
        self.assertTrue(backend.apply(parse_op(
            {"op": "shell", "thickness": 3.0, "faces": []})).ok)
        after = _bbox(backend)
        for axis, (b, a) in enumerate(zip(before, after)):
            self.assertAlmostEqual(a, b, delta=1e-3,
                                   msg="shell grew axis %d: %s -> %s" % (axis, b, a))
        for got, want in zip(after, (60.0, 40.0, 20.0)):
            self.assertAlmostEqual(got, want, delta=1e-3)

    def test_a_closed_shell_is_hollow_with_the_right_wall(self):
        backend = _block()
        backend.apply(parse_op({"op": "shell", "thickness": 3.0, "faces": []}))
        # outer 60x40x20 minus the inward-offset cavity 54x34x14
        expected = BLOCK_VOLUME - 54.0 * 34.0 * 14.0
        self.assertEqual(expected, 22296.0)
        volume = backend.query("metrics")["volume"]
        self.assertAlmostEqual(volume / expected, 1.0, delta=2e-3)
        # a sealed void: two closed surfaces, so chi = 2 + 2 = 4
        validity = backend.query("validity")
        self.assertTrue(validity["is_valid"])
        self.assertEqual(validity["euler_characteristic"], 4)

    def test_the_wall_is_actually_t_thick_everywhere(self):
        """A bbox check alone cannot prove a shell is right.

        An inward shell can preserve the bounding box exactly and still leave the
        wall t/sqrt(3) thick (42% thin) if the inward offset is taken along an
        uncorrected corner normal instead of the true distance. So probe the
        FIELD across each of the six walls: at t-eps inside the outer surface we
        must still be in material, and at t+eps we must be in the cavity.
        """
        backend = _block()
        backend.apply(parse_op({"op": "shell", "thickness": 3.0, "faces": []}))
        field = backend.field()
        eps = 0.05
        # (a point just inside each outer face, the inward direction)
        walls = [
            ((0.0, 20.0, 10.0), (1.0, 0.0, 0.0)),    # -x wall
            ((60.0, 20.0, 10.0), (-1.0, 0.0, 0.0)),  # +x wall
            ((30.0, 0.0, 10.0), (0.0, 1.0, 0.0)),    # -y wall
            ((30.0, 40.0, 10.0), (0.0, -1.0, 0.0)),  # +y wall
            ((30.0, 20.0, 0.0), (0.0, 0.0, 1.0)),    # -z wall
            ((30.0, 20.0, 20.0), (0.0, 0.0, -1.0)),  # +z wall
        ]
        for surface, inward in walls:
            def at(depth):
                return field(tuple(surface[i] + inward[i] * depth for i in range(3)))
            # still solid just short of the wall thickness ...
            self.assertLess(at(3.0 - eps), 0.0,
                            "wall at %s is thinner than t=3" % (surface,))
            # ... and into the cavity just past it
            self.assertGreater(at(3.0 + eps), 0.0,
                               "wall at %s is thicker than t=3" % (surface,))

    def test_an_open_face_removes_only_that_face_not_the_walls(self):
        """faces=['top'] punches the CAVITY out through the top wall.

        It does NOT cut the solid with a half-space -- that would saw the side
        walls down to z = 17 as well. The part stays 20 tall.
        """
        backend = _block()
        backend.apply(parse_op({"op": "shell", "thickness": 3.0, "faces": ["top"]}))
        # the side walls keep their FULL height: the part is still 20 tall.
        for got, want in zip(_bbox(backend), (60.0, 40.0, 20.0)):
            self.assertAlmostEqual(got, want, delta=1e-3)
        # the cavity now runs the full 17mm up to the top face
        expected = BLOCK_VOLUME - 54.0 * 34.0 * 17.0
        self.assertAlmostEqual(
            backend.query("metrics")["volume"] / expected, 1.0, delta=2e-3)
        validity = backend.query("validity")
        self.assertTrue(validity["is_valid"])
        # an open tub is a topological sphere again: chi = 2
        self.assertEqual(validity["euler_characteristic"], 2)

    def test_two_open_faces_make_a_tube(self):
        backend = _block()
        backend.apply(parse_op(
            {"op": "shell", "thickness": 3.0, "faces": ["top", "bottom"]}))
        for got, want in zip(_bbox(backend), (60.0, 40.0, 20.0)):
            self.assertAlmostEqual(got, want, delta=1e-3)
        expected = BLOCK_VOLUME - 54.0 * 34.0 * 20.0
        self.assertAlmostEqual(
            backend.query("metrics")["volume"] / expected, 1.0, delta=2e-3)
        # open at both ends -> a genus-1 tube -> chi = 0
        self.assertEqual(backend.query("validity")["euler_characteristic"], 0)

    def test_an_unknown_shell_face_is_refused(self):
        backend = _block()
        result = backend.apply(parse_op(
            {"op": "shell", "thickness": 3.0, "faces": ["sideways"]}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-value")


class FRepDualContouringTest(unittest.TestCase):
    """Marching cubes cannot put a vertex on a sharp corner; dual contouring can.

    MC constrains every vertex to a grid EDGE, so a box corner is chamfered off
    by up to half a cell -- a systematic, one-sided loss of material, and the
    whole of MC's volume error on a prismatic part. Dual contouring (Ju, Losasso,
    Schaefer & Warren, "Dual Contouring of Hermite Data", SIGGRAPH 2002) puts one
    vertex per CELL, placed at the minimiser of a QEF over the cell's Hermite
    data, which lands exactly on the intersection of the three face planes.
    """

    def test_dual_contouring_is_near_exact_on_a_prismatic_part(self):
        for resolution in (16, 32, 64):
            dc = _block(resolution=resolution, mesher="dual_contouring")
            error = abs(dc.query("metrics")["volume"] - BLOCK_VOLUME) / BLOCK_VOLUME
            self.assertLess(error, 1e-3,
                            "DC volume error %.4f%% at res %d"
                            % (100 * error, resolution))

    def test_dual_contouring_beats_marching_cubes_at_every_resolution(self):
        for resolution in (16, 32, 64):
            mc = _block(resolution=resolution, mesher="marching_cubes")
            dc = _block(resolution=resolution, mesher="dual_contouring")
            e_mc = abs(mc.query("metrics")["volume"] - BLOCK_VOLUME)
            e_dc = abs(dc.query("metrics")["volume"] - BLOCK_VOLUME)
            self.assertLess(e_dc, e_mc / 10.0,
                            "res %d: DC %.1f vs MC %.1f" % (resolution, e_dc, e_mc))

    def test_dual_contouring_at_16_beats_marching_cubes_at_96(self):
        """The headline: DC needs far less grid to do far better."""
        coarse_dc = _block(resolution=16, mesher="dual_contouring")
        fine_mc = _block(resolution=96, mesher="marching_cubes")
        e_dc = abs(coarse_dc.query("metrics")["volume"] - BLOCK_VOLUME)
        e_mc = abs(fine_mc.query("metrics")["volume"] - BLOCK_VOLUME)
        self.assertLess(e_dc, e_mc)

    def test_dual_contouring_reproduces_the_sharp_corners(self):
        """Every one of the block's 8 corners must appear as an actual vertex.

        This is the property marching cubes structurally cannot have: its
        vertices are pinned to grid EDGES, so the nearest one to a true corner
        sits ~half a cell away and the corner is chamfered off. The QEF vertex
        lands on the intersection of the three face planes instead.
        """
        corners = [(cx, cy, cz)
                   for cx in (0.0, 60.0) for cy in (0.0, 40.0) for cz in (0.0, 20.0)]

        def worst(mesher):
            verts, _ = _block(resolution=32, mesher=mesher).mesh()
            return max(min(math.dist(v, c) for v in verts) for c in corners)

        dc, mc = worst("dual_contouring"), worst("marching_cubes")
        # DC lands on every corner to within a rounding error of the QEF solve
        self.assertLess(dc, 1e-3, "DC missed a corner by %.5f" % dc)
        # ... while MC misses by a real fraction of a cell
        self.assertGreater(mc, 0.1)
        self.assertLess(dc, mc / 100.0, "DC %.6f vs MC %.6f" % (dc, mc))

    def test_dual_contouring_output_is_a_watertight_manifold(self):
        for ops in (BLOCK_OPS, CUT_OPS):
            backend = FRepBackend(resolution=32, mesher="dual_contouring")
            for spec in ops:
                self.assertTrue(backend.apply(parse_op(spec)).ok)
            verts, faces = backend.mesh()
            mesh = HalfedgeMesh(verts, faces)
            self.assertTrue(mesh.is_closed())
            self.assertTrue(mesh.is_2manifold()[0])

    def test_dual_contouring_is_deterministic(self):
        a = _block(resolution=24, mesher="dual_contouring").mesh()
        b = _block(resolution=24, mesher="dual_contouring").mesh()
        self.assertEqual(a, b)


class FRepConvergenceTest(unittest.TestCase):
    """The resolution/error curve, and whether the default 48 is defensible."""

    def test_marching_cubes_error_shrinks_with_resolution(self):
        errors = []
        for resolution in (16, 32, 64):
            backend = _block(resolution=resolution, mesher="marching_cubes")
            volume = backend.query("metrics")["volume"]
            errors.append(abs(volume - BLOCK_VOLUME) / BLOCK_VOLUME)
            # MC only ever CHAMFERS material off; it never adds any
            self.assertLess(volume, BLOCK_VOLUME)
        self.assertTrue(errors[0] > errors[1] > errors[2], errors)
        # roughly first-order in the cell size: halving the cell halves the error
        self.assertLess(errors[2], errors[0] / 3.0)

    def test_the_default_resolution_holds_half_a_percent_under_mc(self):
        backend = _block(resolution=frep.DEFAULT_RESOLUTION, mesher="marching_cubes")
        error = abs(backend.query("metrics")["volume"] - BLOCK_VOLUME) / BLOCK_VOLUME
        self.assertLess(error, 5e-3)


class FRepDistancePropertyTest(unittest.TestCase):
    """min/max booleans do NOT preserve the distance property.

    Inigo Quilez, https://iquilezles.org/articles/distfunctions/ : "the Xor and
    the Union of two SDFs produces a true SDF, but not the Subtraction or
    Intersection [...] this is only true in the exterior of the SDF (where
    distances are positive) and not in the interior."

    Every op that reads |f| INSIDE the solid -- shell, offset -- is therefore
    reading a bound, not a distance. These tests pin the violation so it cannot
    silently regress, and pin the guards that keep it from corrupting a mesh.
    """

    def test_min_union_underestimates_the_interior_depth(self):
        """Two 40-cubes overlapping into a 60x40x40 block."""
        backend = FRepBackend(resolution=16)
        for spec in [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1",
             "x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
            {"op": "extrude", "sketch": "sk1", "distance": 40.0},
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk2",
             "x": 20.0, "y": 0.0, "w": 40.0, "h": 40.0},
            {"op": "extrude", "sketch": "sk2", "distance": 40.0},
            {"op": "boolean", "kind": "union"},
        ]:
            self.assertTrue(backend.apply(parse_op(spec)).ok)
        field = backend.field()
        # centre of the resulting 60x40x40 block: the true distance is -20
        # (x:30, y:20, z:20). min() of the two box fields reports only -10.
        self.assertAlmostEqual(field((30.0, 20.0, 20.0)), -10.0, places=6)
        # ... and the field is still a valid BOUND: it never overstates depth.
        self.assertGreaterEqual(field((30.0, 20.0, 20.0)), -20.0)

    def test_prune_never_changes_the_mesh_of_a_shelled_part(self):
        """The guard on the unsound IR.

        frep_ir compiles a shell as the old two-sided ``abs(d) - t/2``, which is
        no longer the function eval_node samples. Interval pruning would then be
        bounding the WRONG function and would classify blocks that straddle the
        real surface as FILLED and delete them -- this exact model lost 66% of
        its volume that way, and the half-edge check passed it, because the
        result is a perfectly closed manifold of the wrong solid. FRepBackend.ir
        refuses to compile a tree containing a shell, so prune degrades to a full
        sample instead of corrupting the mesh.
        """
        def build(prune):
            backend = FRepBackend(resolution=64, prune=prune)
            for spec in [
                {"op": "new_sketch", "plane": "XY"},
                {"op": "add_rectangle", "sketch": "sk1",
                 "x": 0.0, "y": 0.0, "w": 60.0, "h": 40.0},
                {"op": "extrude", "sketch": "sk1", "distance": 40.0},
                {"op": "shell", "thickness": 12.0, "faces": []},
            ]:
                assert backend.apply(parse_op(spec)).ok
            return backend

        plain, pruned = build(False), build(True)
        self.assertIsNone(pruned.ir())          # refused, on purpose
        self.assertEqual(plain.mesh(), pruned.mesh())
        expected = 60.0 * 40.0 * 40.0 - 36.0 * 16.0 * 16.0
        self.assertAlmostEqual(
            pruned.query("metrics")["volume"] / expected, 1.0, delta=2e-3)

    def test_prune_still_prunes_a_part_it_can_soundly_bound(self):
        backend = _block(resolution=48, mesher="marching_cubes", prune=True)
        stats = {}
        backend.mesh(stats=stats)
        self.assertGreater(stats["blocks_pruned"], 0)
        self.assertLess(stats["field_evals"], stats["samples"])


class FRepMarchingCubesAmbiguityTest(unittest.TestCase):
    """Does our 256-case table crack on an ambiguous face?

    The original Lorensen & Cline paper is famous for it. The answer, verified
    exhaustively below, is NO: the table we ship (Bourke's) is *face-consistent*
    -- the segments it lays down on any cube face are a function of that face's
    four corner signs alone, so two cells sharing a face always agree on it and
    the mesh cannot have a hole.

    That is watertightness, NOT topological correctness: the table can still
    resolve an interior ambiguity the wrong way and get the GENUS wrong (that is
    what Chernyaev's MC33 fixes). A genus error is silent -- the mesh stays a
    closed manifold -- so the half-edge check would not catch it. Documented, not
    fixed; dual contouring is the recommended escape.
    """

    def test_the_marching_cubes_table_cannot_crack(self):
        from harnesscad.domain.geometry.volumes.marching_cubes import TRI_TABLE

        # each cube face as (its 4 corner ids, the 4 cube-edge ids lying on it)
        faces = {
            "z0": ((0, 1, 2, 3), {0, 1, 2, 3}),
            "z1": ((4, 5, 6, 7), {4, 5, 6, 7}),
            "y0": ((0, 1, 5, 4), {0, 9, 4, 8}),
            "y1": ((3, 2, 6, 7), {2, 10, 6, 11}),
            "x0": ((0, 3, 7, 4), {3, 11, 7, 8}),
            "x1": ((1, 2, 6, 5), {1, 10, 5, 9}),
        }

        def triangles(config):
            row = TRI_TABLE[config]
            return [tuple(row[i:i + 3]) for i in range(0, len(row), 3)
                    if row[i] != -1]

        comparisons = 0
        for corners, face_edges in faces.values():
            seen = {}
            for config in range(256):
                signature = tuple((config >> c) & 1 for c in corners)
                segments = set()
                for tri in triangles(config):
                    for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                        if a in face_edges and b in face_edges:
                            segments.add(frozenset((a, b)))
                if signature in seen:
                    comparisons += 1
                    self.assertEqual(
                        seen[signature], segments,
                        "face signs %s are triangulated inconsistently -> the "
                        "table can leave a hole" % (signature,))
                else:
                    seen[signature] = segments
        self.assertEqual(comparisons, 1440)

    def test_a_saddle_shaped_part_still_meshes_watertight(self):
        """Four pillars joined by a plate: plenty of ambiguous-face cells."""
        specs = [
            {"op": "new_sketch", "plane": "XY"},
            {"op": "add_rectangle", "sketch": "sk1",
             "x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
            {"op": "extrude", "sketch": "sk1", "distance": 4.0},
        ]
        for index, (cx, cy) in enumerate(((8.0, 8.0), (32.0, 8.0),
                                          (8.0, 32.0), (32.0, 32.0))):
            specs += [
                {"op": "new_sketch", "plane": "XY"},
                {"op": "add_circle", "sketch": "sk%d" % (index + 2),
                 "cx": cx, "cy": cy, "r": 5.0},
                {"op": "extrude", "sketch": "sk%d" % (index + 2), "distance": 20.0},
                {"op": "boolean", "kind": "union"},
            ]
        for mesher in MESHERS:
            backend = FRepBackend(resolution=40, mesher=mesher)
            for spec in specs:
                self.assertTrue(backend.apply(parse_op(spec)).ok, spec)
            verts, faces = backend.mesh()
            self.assertTrue(HalfedgeMesh(verts, faces).is_closed(),
                            "%s produced a hole" % mesher)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
