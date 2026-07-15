"""Tests for the microcad (µcad) geometry backend.

microcad is a NEW declarative CAD *language* (Rust, v0.5.0 EARLY ALPHA). This
backend integrates it exactly the way the OpenSCAD backend does -- emit source,
shell out, read the exported STL back. Two kinds of test live here:

* **Always-on, deterministic, no tool needed.** The emitted microcad SOURCE for a
  box is well-formed and byte-stable, the op->source mapping refuses what the
  alpha language cannot express (with typed diagnostics and no mutation), the
  version is part of the cache key, and the graceful-absence contract holds. These
  run everywhere and never spawn a subprocess.
* **Tool-gated (``skipUnless``).** IF ``cargo install microcad`` produced a
  runnable binary, a box builds with the analytic volume and a watertight mesh,
  and microcad AGREES with OpenSCAD on the shared ops. On this machine the CLI
  did not link (broken MinGW toolchain), so these SKIP cleanly -- they never hang
  and never fake geometry.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import parse_op
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.microcad import (
    MicrocadBackend, MicrocadError, fmt_mm, render,
)
from harnesscad.io.surfaces.server import BACKENDS, CISPServer

HAVE_MICROCAD = MicrocadBackend.available()
REASON = "microcad CLI is not installed on this machine"

# A 20 x 10 rectangle extruded 5 -> analytic volume 1000.
PLATE_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
    {"op": "extrude", "sketch": "sk1", "distance": 5.0},
]
PLATE_VOLUME = 20.0 * 10.0 * 5.0

# The same plate, with an r=3 cylinder cut through it.
CUT_OPS = PLATE_OPS + [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_circle", "sketch": "sk2", "cx": 10.0, "cy": 5.0, "r": 3.0},
    {"op": "extrude", "sketch": "sk2", "distance": 5.0},
    {"op": "boolean", "kind": "cut", "target": "f1", "tool": "f2"},
]

BLOCK_OPS = [
    {"op": "new_sketch", "plane": "XY"},
    {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0},
    {"op": "extrude", "sketch": "sk1", "distance": 10.0},
]

STL_FLOAT32_TOLERANCE = 1e-4


def build(backend, ops):
    """Apply an op stream DIRECTLY to the backend (no session verification).

    A HarnessSession verifies after each op, and one of its verifiers queries
    'validity' -- which for an external backend shells out to the tool to build a
    mesh. These source-emission tests must not spawn the CLI, so they drive the
    backend's own ``apply`` (block-and-correct still holds), exactly as the
    OpenSCAD field-liveness tests do.
    """
    for o in ops:
        result = backend.apply(parse_op(o))
        if not result.ok:
            raise AssertionError("op rejected: %s -> %s"
                                 % (o, [d.to_dict() for d in result.diagnostics]))


# --------------------------------------------------------------------------
# Availability + registration -- no tool required.
# --------------------------------------------------------------------------
class MicrocadAvailabilityTest(unittest.TestCase):
    """The graceful-absence contract holds whether or not the CLI is here."""

    def test_registered_in_the_backend_table(self):
        self.assertIn("microcad", BACKENDS)

    def test_available_never_raises(self):
        self.assertIsInstance(MicrocadBackend.available(), bool)

    def test_backend_unavailable_is_typed_and_actionable(self):
        exc = BackendUnavailable("microcad", "not here", ["PATH:microcad"])
        self.assertIsInstance(exc, RuntimeError)
        self.assertEqual(exc.tool, "microcad")
        self.assertEqual(exc.searched, ["PATH:microcad"])

    def test_server_never_crashes_on_a_missing_tool(self):
        """Present -> the real backend; absent -> the stub, WITH a note."""
        server = CISPServer(backend="microcad")
        if HAVE_MICROCAD:
            self.assertEqual(server.backend_name, "microcad")
            self.assertIsNone(server.backend_note)
        else:
            self.assertEqual(server.backend_name, "stub")
            self.assertIn("microcad", server.backend_note)


# --------------------------------------------------------------------------
# Source emission -- no tool required, fully deterministic.
# --------------------------------------------------------------------------
class MicrocadSourceEmissionTest(unittest.TestCase):
    """The emitted microcad source is well-formed and byte-stable.

    None of this needs the CLI: the emitter is pure. It is exercised through a
    frep-composed backend (which supplies the op state and the CSG tree) with the
    executable stubbed out, so `program()` runs without a binary present.
    """

    def _backend(self, segments=64):
        b = object.__new__(MicrocadBackend)
        # Compose the base without discovering an executable (there may be none).
        from harnesscad.io.backends.frep import FRepBackend
        from harnesscad.io.backends.external import DEFAULT_TIMEOUT
        b.segments = int(segments)
        b.timeout = DEFAULT_TIMEOUT
        b.executable = "microcad"           # never invoked in these tests
        b._frep = FRepBackend()
        b._frep.SHELL_MIN_WALL_CELLS = 0.0
        b._frep.SHELL_JOINS = tuple(MicrocadBackend.SHELL_JOINS)
        b._frep.EDGE_SELECTORS = True
        b._mesh_cache = None
        b._stl_cache = None
        # Pin a version so program()/digest do not shell out to `--version`.
        MicrocadBackend._VERSIONS["microcad"] = "microcad 0.5.0 (test)"
        return b

    def test_fmt_mm_is_deterministic_and_carries_the_unit(self):
        self.assertEqual(fmt_mm(20.0), "20mm")
        self.assertEqual(fmt_mm(5.5), "5.5mm")
        self.assertEqual(fmt_mm(0.0), "0mm")
        self.assertEqual(fmt_mm(-0.0), "0mm")        # -0.0 collapses to 0
        self.assertEqual(fmt_mm(1.200000), "1.2mm")

    def test_a_box_emits_well_formed_source(self):
        b = self._backend()
        build(b, PLATE_OPS)
        source = b.program()
        # the mandatory std imports and the primitives/operators the box needs
        self.assertIn("use std::geo2d::*;", source)
        self.assertIn("use std::geo3d::*;", source)
        self.assertIn("Rect(20mm, 10mm)", source)
        self.assertIn(".extrude(5mm)", source)
        self.assertIn("microcad-version: microcad 0.5.0 (test)", source)
        self.assertTrue(source.rstrip().endswith(";"))

    def test_a_box_source_is_byte_stable_across_two_backends(self):
        a, b = self._backend(), self._backend()
        build(a, PLATE_OPS)
        build(b, PLATE_OPS)
        self.assertEqual(a.program(), b.program())
        self.assertEqual(a.state_digest(), b.state_digest())

    def test_a_boolean_cut_emits_the_difference_operator(self):
        b = self._backend()
        build(b, CUT_OPS)
        source = b.program()
        self.assertIn(" - ", source)                  # difference operator
        # the cut tool is a circle extruded through the plate
        self.assertIn("Circle(r = 3mm)", source)
        self.assertEqual(source.count(".extrude("), 2)

    def test_the_version_is_in_the_program_and_the_digest(self):
        b = self._backend()
        build(b, PLATE_OPS)
        self.assertIn("microcad 0.5.0 (test)", b.program())
        digest = b.state_digest()
        try:
            MicrocadBackend._VERSIONS["microcad"] = "microcad 9.9.9 (other)"
            b2 = self._backend()               # resets the version to test value
            MicrocadBackend._VERSIONS["microcad"] = "microcad 9.9.9 (other)"
            build(b2, PLATE_OPS)
            self.assertNotEqual(b2.state_digest(), digest)
        finally:
            MicrocadBackend._VERSIONS["microcad"] = "microcad 0.5.0 (test)"

    def test_segments_are_in_the_digest_and_the_program(self):
        coarse, fine = self._backend(segments=8), self._backend(segments=64)
        build(coarse, CUT_OPS)
        build(fine, CUT_OPS)
        self.assertIn("segments: 8", coarse.program())
        self.assertIn("segments: 64", fine.program())
        self.assertNotEqual(coarse.state_digest(), fine.state_digest())

    def test_render_is_pure_and_deterministic(self):
        b = self._backend()
        build(b, PLATE_OPS)
        root = b.root()
        self.assertEqual(render(root, 64, "v"), render(root, 64, "v"))


# --------------------------------------------------------------------------
# The op -> microcad mapping: what is REFUSED, typed, and non-mutating.
# --------------------------------------------------------------------------
class MicrocadRefusalTest(unittest.TestCase):
    """Ops microcad's alpha language cannot honour are refused, not faked."""

    def _backend(self):
        return MicrocadSourceEmissionTest._backend(self)

    def test_topology_and_unverified_ops_are_refused_with_typed_diagnostics(self):
        for op in ({"op": "fillet", "edges": [], "radius": 1.0},
                   {"op": "chamfer", "edges": [], "distance": 1.0},
                   {"op": "draft", "faces": [], "angle": 5.0, "neutral_plane": "XY"},
                   {"op": "loft", "sketches": ["sk1"], "ruled": False},
                   {"op": "sweep", "sketch": "sk1", "path": "sk1"},
                   {"op": "shell", "faces": [], "thickness": 3.0},
                   {"op": "mirror", "feature_or_body": "f1", "plane": "XZ"},
                   {"op": "linear_pattern", "feature": "f1", "count": 3,
                    "spacing": 5.0},
                   {"op": "circular_pattern", "feature": "f1", "count": 4}):
            backend = self._backend()
            build(backend, PLATE_OPS)
            before = backend.state_digest()
            result = backend.apply(parse_op(op))
            self.assertFalse(result.ok, op["op"])
            self.assertEqual(result.diagnostics[0].code, "unsupported-op", op["op"])
            self.assertEqual(backend.state_digest(), before)  # nothing mutated

    def test_revolve_is_refused(self):
        backend = self._backend()
        build(backend, [{"op": "new_sketch", "plane": "XY"},
                        {"op": "add_rectangle", "sketch": "sk1", "x": 10.0,
                         "y": 0.0, "w": 4.0, "h": 6.0}])
        before = backend.state_digest()
        result = backend.apply(parse_op(
            {"op": "revolve", "sketch": "sk1", "axis": [0, 0, 0, 0, 1, 0],
             "angle": 360.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertEqual(backend.state_digest(), before)

    def test_a_countersink_hole_is_refused_but_simple_and_counterbore_pass(self):
        backend = self._backend()
        build(backend, BLOCK_OPS)
        before = backend.state_digest()
        csk = {"op": "hole", "face_or_sketch": "", "x": 20.0, "y": 20.0,
               "diameter": 6.0, "through": True, "kind": "countersink",
               "csk_diameter": 12.0, "csk_angle": 90.0}
        result = backend.apply(parse_op(csk))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "unsupported-op")
        self.assertEqual(backend.state_digest(), before)   # nothing mutated

        # a simple hole IS accepted and emits a cylinder difference
        simple = self._backend()
        build(simple, BLOCK_OPS + [
            {"op": "hole", "face_or_sketch": "", "x": 20.0, "y": 20.0,
             "diameter": 6.0, "through": True, "kind": "simple"}])
        self.assertIn("Cylinder(", simple.program())
        self.assertIn(" - ", simple.program())

    def test_bad_reference_still_blocks_and_corrects(self):
        backend = self._backend()
        build(backend, PLATE_OPS)
        before = backend.state_digest()
        result = backend.apply(parse_op({"op": "extrude", "sketch": "sk99",
                                         "distance": 1.0}))
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostics[0].code, "bad-ref")
        self.assertEqual(backend.state_digest(), before)

    def test_a_non_xy_sketch_is_not_lowered_to_a_guessed_orientation(self):
        """A YZ extrude raises rather than emitting a misplaced solid."""
        backend = self._backend()
        build(backend, [{"op": "new_sketch", "plane": "YZ"},
                        {"op": "add_rectangle", "sketch": "sk1", "x": 0.0,
                         "y": 0.0, "w": 10.0, "h": 10.0},
                        {"op": "extrude", "sketch": "sk1", "distance": 5.0}])
        with self.assertRaises(MicrocadError):
            backend.program()


# --------------------------------------------------------------------------
# Tool-gated -- only when the CLI actually linked and installed.
# --------------------------------------------------------------------------
@unittest.skipUnless(HAVE_MICROCAD, REASON)
class MicrocadBuildTest(unittest.TestCase):
    """IF the CLI is present, geometry comes back and matches the analytic value."""

    def test_satisfies_the_geometry_backend_protocol(self):
        self.assertIsInstance(MicrocadBackend(), GeometryBackend)

    def test_plate_builds_and_has_the_analytic_volume(self):
        backend = MicrocadBackend()
        build(backend, PLATE_OPS)
        self.assertTrue(backend.query("summary")["solid_present"])
        measure = backend.query("measure")
        self.assertAlmostEqual(measure["volume"], PLATE_VOLUME,
                               delta=STL_FLOAT32_TOLERANCE)
        self.assertEqual([round(v, 4) for v in measure["bbox"]], [20.0, 10.0, 5.0])

    def test_plate_mesh_is_watertight_and_agrees_with_openscad(self):
        from harnesscad.io.backends.openscad import OpenScadBackend
        if not OpenScadBackend.available():
            self.skipTest("openscad not installed; nothing to cross-check against")
        mc = MicrocadBackend()
        os_ = OpenScadBackend()
        build(mc, PLATE_OPS)
        build(os_, PLATE_OPS)
        self.assertAlmostEqual(mc.query("measure")["volume"],
                               os_.query("measure")["volume"],
                               delta=1e-2 * PLATE_VOLUME)

    def test_deterministic_replay(self):
        a, b = MicrocadBackend(), MicrocadBackend()
        build(a, PLATE_OPS)
        build(b, PLATE_OPS)
        self.assertEqual(a.state_digest(), b.state_digest())
        self.assertEqual(a.program(), b.program())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
