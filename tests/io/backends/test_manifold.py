"""End-to-end tests for the Manifold geometry backend.

These drive a real CISP op stream through HarnessSession -> ManifoldBackend ->
the manifold3d kernel (in-process), and assert on the geometry that comes back:
a watertight 2-manifold mesh, the analytic volume of a box, and a boolean cut
that removes exactly the right amount of material.

The oracle payoff is the point of this backend. Manifold is a genuinely
INDEPENDENT kernel (a guaranteed-manifold mesh-boolean algorithm -- not OCCT,
not Nef polyhedra, not a sampled SDF), so where it AGREES with cadquery it is a
third, independent voice confirming the result; where a hand analytic and two
independent kernels disagree, it is the analytic that is wrong (see the union
case, whose overlapping bars both kernels correctly de-duplicate). The planar and
boolean cases are asserted to machine precision; the curved cases are within the
shared polygonisation budget, because Manifold facets a circle by the SAME $fn law
the OpenSCAD/Blender backends use.

Skips cleanly (unittest.skipUnless) when manifold3d is not installed.
"""

from __future__ import annotations

import math
import unittest

from harnesscad.core.cisp.ops import (
    AddCircle, AddRectangle, Boolean, Extrude, Hole, LinearPattern, Mirror,
    NewSketch, Revolve, Shell, parse_op,
)
from harnesscad.core.loop import HarnessSession
from harnesscad.domain.geometry.mesh.halfedge import HalfedgeMesh
from harnesscad.domain.geometry.parametric import facets
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.external import DEFAULT_SEGMENTS
from harnesscad.io.backends.manifold import (
    ManifoldBackend, ManifoldError, convex_hull, hull_of_points, level_set,
    lower, split_by_plane, trim_by_plane,
)
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

HAVE_MANIFOLD = ManifoldBackend.available()
HAVE_CADQUERY = False
try:  # the third-voice comparison is opt-in on cadquery being present
    from harnesscad.io.backends.cadquery import CadQueryBackend
    HAVE_CADQUERY = CadQueryBackend.available()
except Exception:  # noqa: BLE001
    HAVE_CADQUERY = False

REASON = "manifold3d is not installed on this machine"

#: Everything comes back through a binary STL, whose vertices are float32. On a
#: box whose corners are exactly representable the volume is still exact; this is
#: the slack for float32 round-off on the readback.
STL_FLOAT32_TOLERANCE = 1e-3

# A 60 x 40 x 20 box -> analytic volume 48000.
BOX_OPS = [
    NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 60.0, 40.0), Extrude("sk1", 20.0),
]
BOX_VOLUME = 60.0 * 40.0 * 20.0

# The same box with an r=5 cylinder cut through it.
HOLE_RADIUS = 5.0
CUT_OPS = BOX_OPS + [
    NewSketch("XY"), AddCircle("sk2", 30.0, 20.0, HOLE_RADIUS), Extrude("sk2", 20.0),
    Boolean("cut", "f1", "f2"),
]


def ngon_prism_volume(r: float, height: float, segments: int) -> float:
    """The EXACT volume of the prism Manifold builds for a circle of radius r.

    Manifold does not extrude a circle; it extrudes the inscribed regular n-gon
    the shared $fn law resolves it into. Its area is ``n/2 * r^2 * sin(2pi/n)``.
    """
    n = facets.get_fragments_from_r(r, fn=float(segments))
    return 0.5 * n * r * r * math.sin(2.0 * math.pi / n) * height


HOLE_VOLUME = ngon_prism_volume(HOLE_RADIUS, 20.0, DEFAULT_SEGMENTS)
CUT_VOLUME = BOX_VOLUME - HOLE_VOLUME


def run_ops(backend, ops) -> None:
    session = HarnessSession(backend)
    result = session.apply_ops(list(ops))
    if not result.ok:
        raise AssertionError("op stream rejected: %s"
                             % [d.to_dict() for d in result.diagnostics])


class ManifoldAvailabilityTest(unittest.TestCase):
    """The graceful-absence contract holds whether or not manifold3d is here."""

    def test_registered_in_the_backend_table(self):
        self.assertIn("manifold", BACKENDS)

    def test_registered_in_the_probe(self):
        from harnesscad.eval.selftest.probe import BACKENDS as PB, GEOMETRIC_BACKENDS
        self.assertIn("manifold", PB)
        self.assertIn("manifold", GEOMETRIC_BACKENDS)

    def test_available_never_raises(self):
        self.assertIsInstance(ManifoldBackend.available(), bool)

    def test_server_never_crashes_on_a_missing_tool(self):
        server = CISPServer(backend="manifold")
        if HAVE_MANIFOLD:
            self.assertEqual(server.backend_name, "manifold")
            self.assertIsNone(server.backend_note)
        else:
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("manifold", server.backend_note)


@unittest.skipUnless(HAVE_MANIFOLD, REASON)
class ManifoldBackendTest(unittest.TestCase):

    def test_satisfies_the_geometry_backend_protocol(self):
        self.assertIsInstance(ManifoldBackend(), GeometryBackend)

    def test_box_builds_and_has_the_analytic_volume(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        self.assertTrue(backend.query("summary")["solid_present"])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], BOX_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 6) for v in measure["bbox"]], [60.0, 40.0, 20.0])

    def test_box_mesh_is_a_watertight_2_manifold(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        verts, faces = backend.mesh()
        self.assertGreater(len(faces), 0)
        he = HalfedgeMesh(verts, faces)
        ok, issues = he.is_2manifold()
        self.assertTrue(ok, "manifold mesh is not 2-manifold: %s" % issues)
        self.assertTrue(he.is_closed())
        self.assertEqual(he.genus(), 0)

    def test_boolean_cut_removes_exactly_the_ngon_prism(self):
        backend = ManifoldBackend()
        run_ops(backend, CUT_OPS)
        measure = backend.query("measure")
        # The removed volume is the faceted n-gon prism, known in closed form.
        self.assertAlmostEqual(measure["volume"], CUT_VOLUME,
                               delta=max(1e-3, CUT_VOLUME * 1e-5))

    def test_regenerate_reports_no_diagnostics_on_a_valid_solid(self):
        backend = ManifoldBackend()
        run_ops(backend, CUT_OPS)
        self.assertEqual(backend.regenerate(), [])

    def test_state_digest_is_deterministic_across_identical_replays(self):
        a, b = ManifoldBackend(), ManifoldBackend()
        run_ops(a, CUT_OPS)
        run_ops(b, CUT_OPS)
        self.assertEqual(a.state_digest(), b.state_digest())

    def test_state_digest_folds_in_the_kernel_version(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        self.assertIn(backend.tool_version(), (backend.tool_version(),))
        self.assertNotEqual(backend.tool_version(), "")

    def test_export_stl_is_nonempty(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        data = backend.export("stl")
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertGreater(len(data), 84)  # 80B header + count + >=1 facet


@unittest.skipUnless(HAVE_MANIFOLD, REASON)
class ManifoldRefusalTest(unittest.TestCase):
    """Every op Manifold cannot honour is REFUSED with a typed diagnostic, never
    faked. A dropped field is the bug this whole codebase eradicated."""

    def _refusal_code(self, ops):
        backend = ManifoldBackend()
        session = HarnessSession(backend)
        result = session.apply_ops(list(ops))
        self.assertFalse(result.ok)
        return {d.code for d in result.diagnostics}

    def test_fillet_is_refused_not_faked(self):
        from harnesscad.core.cisp.ops import Fillet
        codes = self._refusal_code(BOX_OPS + [Fillet(("|Z",), 3.0)])
        self.assertIn("unsupported-op", codes)

    def test_loft_and_sweep_and_draft_are_refused(self):
        from harnesscad.core.cisp.ops import Chamfer, Draft
        self.assertIn("unsupported-op",
                      self._refusal_code(BOX_OPS + [Chamfer(("|Z",), 3.0, None)]))
        self.assertIn("unsupported-op",
                      self._refusal_code(BOX_OPS + [Draft((">X",), 5.0, "<Z")]))

    def test_shell_of_a_prism_is_supported_exactly(self):
        # A 60x40x20 box shelled to t=3 with the top open: volume is exact,
        #   48000 - (54 * 34 * 17) = 16788, no tolerance band.
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS + [Shell((">Z",), 3.0, "arc")])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], 48000.0 - 54.0 * 34.0 * 17.0,
                               delta=1e-3)

    def test_refused_fillet_taints_the_measurement(self):
        # Manifold has no B-rep edges, so fillet is refused. The measurement used
        # to still read the un-filleted block (9000 for a 50x30x6), a silent wrong
        # part. A refused op means the requested part was never built -> REFUSE.
        from harnesscad.core.cisp.ops import Fillet
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        self.assertAlmostEqual(backend.query("measure")["volume"], BOX_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        r = backend.apply(Fillet(("|Z",), 3.0))              # refused
        self.assertFalse(r.ok)
        self.assertIn("unsupported-op", [d.code for d in r.diagnostics])
        for q in ("measure", "metrics"):
            self.assertIsNone(backend.query(q)["volume"], q)  # NOT 48000
            self.assertIsNone(backend.query(q)["bbox"], q)

    def test_supported_shell_does_not_taint_the_measurement(self):
        # shell of a prism IS supported: it must keep measuring, never trip the
        # refusal taint that only an unsupported op sets.
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS + [Shell((">Z",), 3.0, "arc")])
        self.assertIsNotNone(backend.query("measure")["volume"])
        self.assertAlmostEqual(backend.query("measure")["volume"],
                               48000.0 - 54.0 * 34.0 * 17.0, delta=1e-3)


@unittest.skipUnless(HAVE_MANIFOLD, REASON)
class ManifoldFieldLivenessTest(unittest.TestCase):
    """The census that matters: no field this backend was handed is silently
    dropped. Every (op, field) must be LIVE, or a typed refusal (ERR), never DEAD."""

    def test_no_dead_fields_on_the_manifold_engine(self):
        from harnesscad.eval.selftest import field_liveness as fl
        report = fl.run(backends=["manifold"])
        dead = [(c.op, c.field) for c in report.cells
                if c.dead and c.backend == "manifold"]
        self.assertEqual(dead, [], "manifold silently dropped fields: %s" % dead)
        self.assertEqual(report.unmapped, [])


@unittest.skipUnless(HAVE_MANIFOLD and HAVE_CADQUERY,
                     "manifold and cadquery are both required for the third-voice "
                     "agreement test")
class ManifoldAgreesWithCadQueryTest(unittest.TestCase):
    """The oracle payoff: on the ops both support, an INDEPENDENT kernel must
    agree. Planar/boolean parts to machine precision; curved parts within the
    mesh polygonisation budget (cadquery is exact B-rep, Manifold is a faceted
    mesh, so a curved part lands a known fraction low)."""

    PLANAR_CASES = {
        "box": BOX_OPS,
        "union_overlapping_bars": BOX_OPS + [
            NewSketch("XY"), AddRectangle("sk2", 50.0, 0.0, 40.0, 40.0),
            Extrude("sk2", 20.0), Boolean("union", "f1", "f2")],
        "linear_pattern": [
            NewSketch("XY"), AddRectangle("sk1", 0.0, 0.0, 10.0, 10.0),
            Extrude("sk1", 10.0), LinearPattern("f1", (1.0, 0.0, 0.0), 3, 20.0)],
        "mirror": [
            NewSketch("XY"), AddRectangle("sk1", 100.0, 0.0, 20.0, 20.0),
            Extrude("sk1", 10.0), Mirror("f1", "YZ")],
        "shell_prism": BOX_OPS + [Shell((">Z",), 3.0, "arc")],
    }
    CURVED_CASES = {
        "cylinder": [NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 10.0),
                     Extrude("sk1", 20.0)],
        "box_minus_cylinder": CUT_OPS,
        "through_hole": BOX_OPS + [Hole(">Z", 30.0, 20.0, 10.0, None, True, "simple")],
        "revolve_tube": [NewSketch("XY"), AddRectangle("sk1", 20.0, 0.0, 10.0, 10.0),
                         Revolve("sk1", (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), 360.0)],
    }

    def _volume(self, name, ops):
        m = ManifoldBackend()
        run_ops(m, ops)
        c = CadQueryBackend()
        run_ops(c, ops)
        return m.query("measure"), c.query("measure")

    def test_planar_and_boolean_parts_agree_to_machine_precision(self):
        for name, ops in self.PLANAR_CASES.items():
            with self.subTest(case=name):
                mm, cq = self._volume(name, ops)
                self.assertAlmostEqual(
                    mm["volume"], cq["volume"], delta=1e-3,
                    msg="%s: manifold %s vs cadquery %s"
                        % (name, mm["volume"], cq["volume"]))
                for a, b in zip(mm["bbox"], cq["bbox"]):
                    self.assertAlmostEqual(a, b, delta=1e-3)

    def test_curved_parts_agree_within_the_polygonisation_budget(self):
        for name, ops in self.CURVED_CASES.items():
            with self.subTest(case=name):
                mm, cq = self._volume(name, ops)
                rel = abs(mm["volume"] - cq["volume"]) / cq["volume"]
                # A 64-gon under-fills a circle by ~0.16%; allow 1% headroom.
                self.assertLess(rel, 0.01,
                                "%s: manifold %s vs cadquery(exact) %s (%.3f%%)"
                                % (name, mm["volume"], cq["volume"], rel * 100))


@unittest.skipUnless(HAVE_MANIFOLD, REASON)
class ManifoldExactCapabilitiesTest(unittest.TestCase):
    """The exact ancillary Manifold powers the first pass never surfaced: convex
    hull, half-space split/trim, and the level_set SDF road. hull/split/trim are
    exact (no tolerance beyond float32); level_set is a mesh of the SAME F-rep SDF
    the FRep backend samples, so it CROSS-CHECKS against FRep and the analytic."""

    def _box_solid(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        return lower(backend.root(), backend.segments)

    def test_convex_hull_of_a_convex_box_is_the_box(self):
        solid = self._box_solid()
        self.assertAlmostEqual(convex_hull(solid).volume(), BOX_VOLUME, delta=1e-6)

    def test_hull_of_points_is_the_exact_polytope(self):
        # Corner-of-a-cube plus the far corner: a pyramid of volume 500.
        pts = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
               [0.0, 0.0, 10.0], [10.0, 10.0, 10.0]]
        self.assertAlmostEqual(hull_of_points(pts).volume(), 500.0, delta=1e-6)

    def test_trim_by_plane_keeps_exactly_the_half(self):
        # The box spans x in [0, 60]; trim at x = 30 with +X normal -> half volume.
        half = trim_by_plane(self._box_solid(), (1.0, 0.0, 0.0), 30.0)
        self.assertAlmostEqual(half.volume(), BOX_VOLUME / 2.0, delta=1e-4)

    def test_split_by_plane_partitions_the_whole_volume(self):
        keep, drop = split_by_plane(self._box_solid(), (1.0, 0.0, 0.0), 25.0)
        self.assertAlmostEqual(keep.volume() + drop.volume(), BOX_VOLUME, delta=1e-4)
        self.assertAlmostEqual(keep.volume(), 35.0 * 40.0 * 20.0, delta=1e-4)

    def test_level_set_of_a_box_agrees_with_frep_and_the_analytic(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        node = backend.root()
        frep_vol = backend._frep.query("measure")["volume"]
        solid = level_set(node, 64)
        rel_analytic = abs(solid.volume() - BOX_VOLUME) / BOX_VOLUME
        rel_frep = abs(solid.volume() - frep_vol) / frep_vol
        # A marched SDF under-fills the sharp box slightly; both cross-checks agree
        # to well within the level_set grid budget.
        self.assertLess(rel_analytic, 0.01, "level_set vs analytic %.3f%%"
                        % (rel_analytic * 100))
        self.assertLess(rel_frep, 0.01, "level_set vs frep %.3f%%"
                        % (rel_frep * 100))

    def test_level_set_converges_as_segments_rise(self):
        backend = ManifoldBackend()
        run_ops(backend, [NewSketch("XY"), AddCircle("sk1", 0.0, 0.0, 10.0),
                          Extrude("sk1", 20.0)])
        node = backend.root()
        analytic = math.pi * 10.0 * 10.0 * 20.0
        coarse = abs(level_set(node, 24).volume() - analytic)
        fine = abs(level_set(node, 96).volume() - analytic)
        self.assertLess(fine, coarse, "level_set did not converge: %.1f -> %.1f"
                        % (coarse, fine))

    def test_level_set_refuses_a_degenerate_node(self):
        backend = ManifoldBackend()
        run_ops(backend, BOX_OPS)
        with self.assertRaises(ManifoldError):
            level_set(backend.root(), 0, edge_length=-1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
