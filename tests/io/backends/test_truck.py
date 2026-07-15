"""End-to-end tests for the truck (Rust B-rep NURBS kernel) geometry backend.

These drive a real CISP op stream through the TruckBackend -> the compiled Rust
driver (`truck_driver/`, a subprocess over the truck crates) and assert on the
geometry that comes back: a watertight 2-manifold mesh, the EXACT analytic volume
of a box, a padded boolean cut that removes the right material, and a solid of
revolution within the tessellation budget.

The oracle payoff is the whole point of this backend. Every other B-rep engine
the harness has -- cadquery, freecad, build123d -- is a wrapper around the SAME
kernel (OpenCASCADE), so they agree by construction. truck is a from-scratch,
independent B-rep NURBS kernel written in Rust (NOT OCCT), so where truck AGREES
with cadquery/freecad it is a genuinely independent second lineage confirming the
result. The planar box is asserted to machine precision (truck builds it with
flat faces, no tessellation error at all); the curved/boolean parts are within
the tessellation budget, because truck reads volume/bbox back from its OWN
triangulation of its NURBS surfaces.

Skips cleanly (unittest.skipUnless) when the Rust driver binary was not built
(no toolchain, or `cargo build --release` was never run in truck_driver/).
"""

from __future__ import annotations

import math
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Chamfer, CircularPattern, Draft, Extrude,
    Fillet, Hole, LinearPattern, Loft, Mirror, NewSketch, Revolve, Shell, Sweep,
)
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.truck import TruckBackend, TruckError
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

HAVE_TRUCK = TruckBackend.available()
REASON = "the truck Rust driver binary is not built (cargo build --release)"

# The optional third-voice comparison is opt-in on the OCCT kernels being present.
HAVE_CADQUERY = False
HAVE_FREECAD = False
try:
    from harnesscad.io.backends.cadquery import CadQueryBackend
    HAVE_CADQUERY = CadQueryBackend.available()
except Exception:  # noqa: BLE001
    HAVE_CADQUERY = False
try:
    from harnesscad.io.backends.freecad import FreeCADBackend
    HAVE_FREECAD = FreeCADBackend.available()
except Exception:  # noqa: BLE001
    HAVE_FREECAD = False

#: Everything comes back through a binary STL whose vertices are float32; this is
#: the slack for float32 round-off on a box whose corners are exactly
#: representable (its volume is still analytically exact).
STL_FLOAT32_TOLERANCE = 1e-2

# A 60 x 40 x 20 box -> analytic volume 48000 (flat faces: no tessellation error).
BOX_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0), Extrude("sk1", 20.0),
]
BOX_VOLUME = 60.0 * 40.0 * 20.0

# The box with a through-hole. The Hole op pads its cutter beyond both caps, so
# the boolean has no coplanar faces -- the case truck-shapeops handles robustly.
# (Hole's 4th argument is DIAMETER, so the removed cylinder has radius D/2.)
HOLE_DIAMETER = 10.0
HOLE_OPS = BOX_OPS + [Hole(">Z", 30.0, 20.0, HOLE_DIAMETER, None, True, "simple")]
HOLE_REMOVED = math.pi * (HOLE_DIAMETER / 2.0) ** 2 * 20.0
HOLE_VOLUME = BOX_VOLUME - HOLE_REMOVED
# A cylinder Ø20 x 20 tall (curved: NURBS, so faceted on readback).
CYL_OPS = [NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 10.0), Extrude("sk1", 20.0)]
CYL_VOLUME = math.pi * 10.0 * 10.0 * 20.0
# A square-section tube: rectangle (r=20..30, h=0..10) revolved 360 about Y.
REVOLVE_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 20.0, 0.0, 10.0, 10.0),
    Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0),
]
REVOLVE_VOLUME = 2.0 * math.pi * 25.0 * (10.0 * 10.0)  # Pappus: 2*pi*R_c*Area

# A 3x linear pattern of a 10x10x5 block, 20 apart along X: three DISJOINT bodies.
# truck-shapeops cannot union disjoint solids, so the driver assembles them as a
# single multi-shell Solid instead -- exact, no boolean. Volume = 3 * 500 = 1500.
LINPAT_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 10.0, 10.0), Extrude("sk1", 5.0),
    LinearPattern("f1", (1.0, 0.0, 0.0), 3, 20.0),
]
LINPAT_VOLUME = 3 * 10.0 * 10.0 * 5.0
# A 4x circular pattern about Z of a block offset to r=20..30: four disjoint bodies.
CIRCPAT_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 20.0, -5.0, 10.0, 10.0), Extrude("sk1", 5.0),
    CircularPattern("f1", (0.0, 0.0, 0.0, 0.0, 0.0, 1.0), 4, 360.0),
]
CIRCPAT_VOLUME = 4 * 10.0 * 10.0 * 5.0
# A mirror across YZ of a block at x in [10,30]: the reflection lands at x in
# [-30,-10], disjoint from the original -> a two-shell Solid. Volume = 2 * 4000.
MIRROR_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 10.0, 0.0, 20.0, 20.0), Extrude("sk1", 10.0),
    Mirror("f1", "YZ"),
]
MIRROR_VOLUME = 2 * 20.0 * 20.0 * 10.0

#: Curved solids are faceted by truck's own tessellation, so a small relative
#: error is expected (and is what the differential oracle tolerates for a B-rep
#: read back through a mesh).
CURVED_REL_TOLERANCE = 1e-2


def apply_direct(backend, ops):
    """Apply ops straight to a backend (no HarnessSession verify loop), returning
    the first rejecting result or None. Used so a geometry assertion is about the
    backend, not about the harness's own verifiers."""
    for op in ops:
        result = backend.apply(op)
        if not result.ok:
            return result
    return None


def run_ops(backend, ops) -> None:
    session = HarnessSession(backend)
    result = session.apply_ops(list(ops))
    if not result.ok:
        raise AssertionError("op stream rejected: %s"
                             % [d.to_dict() for d in result.diagnostics])


class TruckAvailabilityTest(unittest.TestCase):
    """The graceful-absence contract holds whether or not the binary is built."""

    def test_registered_in_the_backend_table(self):
        self.assertIn("truck", BACKENDS)

    def test_registered_in_the_probe(self):
        from harnesscad.eval.selftest.probe import (
            BACKENDS as PB, GEOMETRIC_BACKENDS, TOLERANCES)
        self.assertIn("truck", PB)
        self.assertIn("truck", GEOMETRIC_BACKENDS)
        self.assertIn("truck", TOLERANCES)

    def test_available_never_raises(self):
        self.assertIsInstance(TruckBackend.available(), bool)

    def test_server_never_crashes_on_a_missing_binary(self):
        server = CISPServer(backend="truck")
        if HAVE_TRUCK:
            self.assertEqual(server.backend_name, "truck")
            self.assertIsNone(server.backend_note)
        else:
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("truck", server.backend_note)


@unittest.skipUnless(HAVE_TRUCK, REASON)
class TruckBackendTest(unittest.TestCase):

    def test_satisfies_the_geometry_backend_protocol(self):
        self.assertIsInstance(TruckBackend(), GeometryBackend)

    def test_box_builds_and_has_the_exact_analytic_volume(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        self.assertTrue(backend.query("summary")["solid_present"])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], BOX_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 4) for v in measure["bbox"]], [60.0, 40.0, 20.0])

    def test_box_mesh_is_a_watertight_2_manifold(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        verts, faces = backend.mesh()
        self.assertGreater(len(faces), 0)
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        self.assertTrue(ok, "truck mesh is not 2-manifold: %s" % issues)
        self.assertTrue(he.is_closed())
        self.assertEqual(he.genus(), 0)

    def test_box_brep_topology_counts(self):
        """truck's OWN B-rep report: a box is 6 faces and 12 edges, exactly."""
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        brep = backend.query("brep")
        self.assertTrue(brep["ok"])
        self.assertEqual(brep["n_faces"], 6)
        self.assertEqual(brep["n_edges"], 12)

    def test_cylinder_volume_within_the_tessellation_budget(self):
        backend = TruckBackend()
        run_ops(backend, CYL_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, CYL_VOLUME,
                               delta=CYL_VOLUME * CURVED_REL_TOLERANCE)

    def test_through_hole_removes_material(self):
        backend = TruckBackend()
        run_ops(backend, HOLE_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, HOLE_VOLUME,
                               delta=HOLE_VOLUME * CURVED_REL_TOLERANCE)

    def test_revolve_tube_matches_pappus_volume(self):
        backend = TruckBackend()
        run_ops(backend, REVOLVE_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, REVOLVE_VOLUME,
                               delta=REVOLVE_VOLUME * CURVED_REL_TOLERANCE)

    def test_linear_pattern_builds_disjoint_bodies_as_a_multi_shell_solid(self):
        # Three disjoint blocks: truck-shapeops cannot union them, so the driver
        # builds a single Solid with three boundary shells. Volume is EXACT (all
        # faces planar) and the mesh is three closed boxes.
        backend = TruckBackend()
        run_ops(backend, LINPAT_OPS)
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], LINPAT_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 4) for v in measure["bbox"]], [50.0, 10.0, 5.0])
        brep = backend.query("brep")
        self.assertTrue(brep["ok"])
        self.assertEqual(brep["n_faces"], 18)   # 3 boxes x 6 faces

    def test_circular_pattern_builds_disjoint_bodies(self):
        backend = TruckBackend()
        run_ops(backend, CIRCPAT_OPS)
        volume = backend.query("measure")["volume"]
        self.assertAlmostEqual(volume, CIRCPAT_VOLUME, delta=STL_FLOAT32_TOLERANCE)

    def test_mirror_reflects_a_body_across_a_datum_plane(self):
        # The reflection has a negative-determinant matrix; the driver un-inverts
        # the copy so the combined two-shell solid is watertight with outward
        # normals (not an inside-out reflection).
        backend = TruckBackend()
        run_ops(backend, MIRROR_OPS)
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], MIRROR_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 4) for v in measure["bbox"]], [60.0, 20.0, 10.0])
        verts, faces = backend.mesh()
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        self.assertTrue(ok, "mirrored solid not 2-manifold: %s" % issues)
        self.assertTrue(he.is_closed())

    def test_regenerate_reports_no_diagnostics_on_a_valid_solid(self):
        backend = TruckBackend()
        run_ops(backend, HOLE_OPS)
        self.assertEqual(backend.regenerate(), [])

    def test_state_digest_is_deterministic_across_identical_replays(self):
        a, b = TruckBackend(), TruckBackend()
        run_ops(a, HOLE_OPS)
        run_ops(b, HOLE_OPS)
        self.assertEqual(a.state_digest(), b.state_digest())

    def test_state_digest_folds_in_the_crate_versions_and_binary_hash(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        version = backend.tool_version()
        self.assertIn("truck-modeling-0.6.0", version)
        self.assertIn("bin=", version)

    def test_export_stl_is_nonempty(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        data = backend.export("stl")
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertGreater(len(data), 84)  # 80B header + count + >=1 facet

    def test_export_step_is_iso10303_for_a_pure_modeling_solid(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        data = backend.export("step")
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertIn(b"ISO-10303-21", data)

    def test_export_step_is_refused_after_a_boolean(self):
        """truck-stepio 0.3 cannot serialise a boolean result, so STEP is refused
        (never an empty/wrong file) for a model that contains one."""
        backend = TruckBackend()
        run_ops(backend, HOLE_OPS)
        with self.assertRaises(ValueError):
            backend.export("step")


@unittest.skipUnless(HAVE_TRUCK, REASON)
class TruckRefusalTest(unittest.TestCase):
    """Every op truck cannot honour is REFUSED with a typed diagnostic, never
    faked. A silently dropped field is the bug this whole codebase eradicated."""

    def _refusal_codes(self, ops):
        backend = TruckBackend()
        result = apply_direct(backend, ops)
        self.assertIsNotNone(result, "expected the op to be refused, but it passed")
        return {d.code for d in result.diagnostics}

    def test_fillet_is_refused(self):
        self.assertIn("unsupported-op", self._refusal_codes(BOX_OPS + [Fillet(("|Z",), 3.0)]))

    def test_chamfer_is_refused(self):
        self.assertIn("unsupported-op",
                      self._refusal_codes(BOX_OPS + [Chamfer(("|Z",), 3.0, None)]))

    def test_draft_is_refused(self):
        self.assertIn("unsupported-op",
                      self._refusal_codes(BOX_OPS + [Draft((">X",), 5.0, "<Z")]))

    def test_shell_is_refused(self):
        self.assertIn("unsupported-op", self._refusal_codes(BOX_OPS + [Shell((">Z",), 3.0, "arc")]))

    def test_loft_and_sweep_are_refused(self):
        self.assertIn("unsupported-op", self._refusal_codes(BOX_OPS + [Loft((), False, ())]))
        self.assertIn("unsupported-op", self._refusal_codes(BOX_OPS + [Sweep("sk1", "sk2")]))

    def test_countersink_hole_is_refused(self):
        codes = self._refusal_codes(
            BOX_OPS + [Hole(">Z", 30.0, 20.0, 6.0, None, True, "countersink", None, None, 12.0)])
        self.assertIn("unsupported-op", codes)

    def test_refused_op_taints_the_measurement_not_the_last_good_solid(self):
        # truck emits a typed unsupported-op at apply(), but the measurement used
        # to still read the LAST-GOOD (un-shelled) geometry: a shelled box came
        # back as 48000, the un-shelled stock, a silent wrong part. A refused op
        # means the requested part was never built, so measure must REFUSE.
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        self.assertAlmostEqual(backend.query("measure")["volume"], BOX_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        r = backend.apply(Shell((">Z",), 3.0, "arc"))       # refused
        self.assertFalse(r.ok)
        self.assertIn("unsupported-op", [d.code for d in r.diagnostics])
        for q in ("measure", "metrics"):
            self.assertIsNone(backend.query(q)["volume"], q)   # NOT 48000
            self.assertIsNone(backend.query(q)["bbox"], q)

    def test_fillet_refuses_measurement(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        self.assertFalse(backend.apply(Fillet(("|Z",), 3.0)).ok)
        self.assertIsNone(backend.query("measure")["volume"])

    def test_reset_clears_the_refusal_taint(self):
        backend = TruckBackend()
        run_ops(backend, BOX_OPS)
        self.assertFalse(backend.apply(Shell((">Z",), 3.0, "arc")).ok)
        self.assertIsNone(backend.query("measure")["volume"])
        backend.reset()
        run_ops(backend, BOX_OPS)
        self.assertAlmostEqual(backend.query("measure")["volume"], BOX_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)


@unittest.skipUnless(HAVE_TRUCK, REASON)
class TruckFieldLivenessTest(unittest.TestCase):
    """The census that matters: no field this backend was handed is silently
    dropped. Every (op, field) must be LIVE, or a typed refusal, never DEAD."""

    def test_no_dead_fields_on_the_truck_engine(self):
        from harnesscad.eval.selftest import field_liveness as fl
        report = fl.run(backends=["truck"])
        dead = [(c.op, c.field) for c in report.cells
                if c.dead and c.backend == "truck"]
        self.assertEqual(dead, [], "truck silently dropped fields: %s" % dead)
        self.assertEqual(report.unmapped, [])


@unittest.skipUnless(HAVE_TRUCK and (HAVE_CADQUERY or HAVE_FREECAD),
                     "truck plus at least one OCCT kernel are required for the "
                     "independent-B-rep agreement table")
class TruckAgreesWithOcctTest(unittest.TestCase):
    """The oracle payoff: on the ops all three support, the INDEPENDENT truck
    kernel must AGREE with the OCCT engines. The box is asserted to (near) machine
    precision -- truck builds it with flat faces, so there is no tessellation error
    at all; the curved/boolean parts are within the tessellation budget, because
    truck reads volume back through its own triangulation while OCCT is exact.

    This is the truck-vs-OCCT-vs-analytic table the whole backend exists to fill:
    a number two unrelated B-rep lineages both vouch for is a number to trust.
    """

    CASES = {
        "box": (BOX_OPS, BOX_VOLUME, STL_FLOAT32_TOLERANCE),
        "cylinder": (CYL_OPS, CYL_VOLUME, CYL_VOLUME * CURVED_REL_TOLERANCE),
        "through_hole": (HOLE_OPS, HOLE_VOLUME, HOLE_VOLUME * CURVED_REL_TOLERANCE),
        "revolve_tube": (REVOLVE_OPS, REVOLVE_VOLUME, REVOLVE_VOLUME * CURVED_REL_TOLERANCE),
        # newly-added ops: all-planar, so the independent kernels must agree to
        # (near) machine precision, not just within the tessellation budget.
        "linear_pattern": (LINPAT_OPS, LINPAT_VOLUME, STL_FLOAT32_TOLERANCE),
        "circular_pattern": (CIRCPAT_OPS, CIRCPAT_VOLUME, STL_FLOAT32_TOLERANCE),
        "mirror": (MIRROR_OPS, MIRROR_VOLUME, STL_FLOAT32_TOLERANCE),
    }

    def _volume(self, backend_cls, ops):
        backend = backend_cls()
        rejected = apply_direct(backend, ops)
        self.assertIsNone(rejected, "op stream unexpectedly rejected: %s"
                          % (rejected.diagnostics if rejected else None))
        return backend.query("measure")["volume"]

    def test_truck_agrees_with_occt_and_the_analytic(self):
        occt = []
        if HAVE_CADQUERY:
            occt.append(("cadquery", CadQueryBackend))
        if HAVE_FREECAD:
            occt.append(("freecad", FreeCADBackend))
        for name, (ops, analytic, delta) in self.CASES.items():
            truck_v = self._volume(TruckBackend, ops)
            # truck vs analytic
            self.assertAlmostEqual(
                truck_v, analytic, delta=delta,
                msg="truck %s: %g vs analytic %g" % (name, truck_v, analytic))
            # truck vs each independent OCCT kernel
            for occt_name, occt_cls in occt:
                occt_v = self._volume(occt_cls, ops)
                self.assertAlmostEqual(
                    truck_v, occt_v, delta=max(delta, abs(occt_v) * CURVED_REL_TOLERANCE),
                    msg="truck %s (%g) disagrees with %s (%g)"
                        % (name, truck_v, occt_name, occt_v))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
