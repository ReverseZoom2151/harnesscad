"""ZooBackend: offline KCL emission + key-gated live paths that SKIP cleanly.

Everything here is offline and needs no API token. The live paths (mesh export
via Zoo's engine, the text-to-CAD comparator) are asserted to SKIP without a key
rather than fabricate a result. No network is touched; no secret is handled.
"""

import os
import tempfile
import unittest

from harnesscad.core.cisp.ops import (
    NewSketch, AddRectangle, AddCircle, Extrude, Boolean, Fillet,
)
from harnesscad.io.backends.base import BackendUnavailable, GeometryBackend
from harnesscad.io.backends.zoo import (
    ZooBackend, ZooTextToCadComparator, ComparatorPlan,
    TOKEN_ENV_VARS, token_present, MESH_EXPORT_FORMATS,
)


def _box(backend):
    backend.apply(NewSketch(plane="XY"))
    backend.apply(AddRectangle(sketch="sk1", x=0.0, y=0.0, w=20.0, h=10.0))
    backend.apply(Extrude(sketch="sk1", distance=5.0))


class _NoToken:
    """Context manager that removes any real token for the duration of a test."""

    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in TOKEN_ENV_VARS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v


class TestProtocolAndState(unittest.TestCase):
    def test_is_a_geometry_backend(self):
        self.assertIsInstance(ZooBackend(), GeometryBackend)

    def test_op_state_is_validated(self):
        # Block-and-correct is inherited from the composed stub: a bad ref fails
        # without mutating.
        b = ZooBackend()
        r = b.apply(Extrude(sketch="nope", distance=5))
        self.assertFalse(r.ok)
        self.assertEqual(b.query("summary")["feature_count"], 0)

    def test_query_summary(self):
        b = ZooBackend()
        _box(b)
        s = b.query("summary")
        self.assertTrue(s["solid_present"])
        self.assertEqual(s["feature_count"], 1)


class TestKclExport(unittest.TestCase):
    def test_export_kcl_offline(self):
        b = ZooBackend(name="box")
        _box(b)
        text = b.export("kcl")
        self.assertIn("startSketchOn(XY)", text)
        self.assertRegex(text, r"extrude\(profile\d+, length = 5\)")

    def test_digest_is_deterministic_and_reflects_kcl(self):
        b1 = ZooBackend()
        _box(b1)
        b2 = ZooBackend()
        _box(b2)
        self.assertEqual(b1.state_digest(), b2.state_digest())
        # A different model -> a different digest.
        b3 = ZooBackend()
        b3.apply(NewSketch(plane="XY"))
        b3.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=4))
        b3.apply(Extrude(sketch="sk1", distance=5))
        self.assertNotEqual(b1.state_digest(), b3.state_digest())

    def test_digest_never_contains_a_token(self):
        # Even if a token is set, it must not leak into the digest.
        b = ZooBackend()
        _box(b)
        os.environ["ZOO_API_TOKEN"] = "SECRET-should-not-appear"
        try:
            self.assertNotIn("SECRET", b.state_digest())
        finally:
            os.environ.pop("ZOO_API_TOKEN", None)


class TestLivePathsSkip(unittest.TestCase):
    def test_mesh_export_skips_without_key(self):
        with _NoToken():
            b = ZooBackend()
            _box(b)
            for fmt in ("stl", "obj", "step", "gltf"):
                with self.assertRaises(BackendUnavailable):
                    b.export(fmt)

    def test_unknown_format_refused(self):
        b = ZooBackend()
        _box(b)
        with self.assertRaises(BackendUnavailable):
            b.export("dwg")

    def test_mesh_formats_advertised(self):
        self.assertIn("stl", MESH_EXPORT_FORMATS)
        self.assertIn("step", MESH_EXPORT_FORMATS)


class TestTokenHandling(unittest.TestCase):
    def test_token_present_probe(self):
        with _NoToken():
            self.assertFalse(token_present())
            os.environ["KITTYCAD_API_TOKEN"] = "x"
            try:
                self.assertTrue(token_present())
            finally:
                os.environ.pop("KITTYCAD_API_TOKEN", None)


class TestComparator(unittest.TestCase):
    def test_plan_is_inert_and_skips_without_key(self):
        with _NoToken():
            plan = ZooTextToCadComparator().plan("a 20mm cube")
            self.assertIsInstance(plan, ComparatorPlan)
            self.assertFalse(plan.live_available)
            self.assertEqual(plan.output_format, "step")
            self.assertIn("skipped", plan.reason)

    def test_plan_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            ZooTextToCadComparator().plan("   ")

    def test_score_geometry_offline_gate(self):
        # The scoring half is fully offline: a valid cube passes our gate.
        from harnesscad.domain.geometry.mesh.polyhedron import unit_cube
        from harnesscad.io.formats.registry import Mesh

        c = unit_cube(1.0)
        mesh = Mesh.from_vertices_faces(
            [(x * 10, y * 10, z * 10) for (x, y, z) in c.vertices], c.faces)
        report = ZooTextToCadComparator().score_geometry(mesh)
        self.assertTrue(report.ok)
        self.assertAlmostEqual(report.measurement["volume"], 1000.0)

    def test_score_geometry_refuses_broken_mesh(self):
        # A single triangle is not a closed solid: the gate reports not-ok.
        from harnesscad.io.formats.stl import Triangle
        from harnesscad.io.formats.registry import Mesh

        bad = Mesh((Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)),))
        report = ZooTextToCadComparator().score_geometry(bad)
        self.assertFalse(report.ok)


class TestRegistryIntegration(unittest.TestCase):
    def test_kcl_write_through_registry_and_gate(self):
        from harnesscad.io.formats import registry as fmt

        b = ZooBackend(name="cyl")
        b.apply(NewSketch(plane="XY"))
        b.apply(AddCircle(sketch="sk1", cx=0, cy=0, r=8))
        b.apply(Extrude(sketch="sk1", distance=12))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cyl.kcl")
            fmt.write(b, path)                 # dispatch on extension, through gate
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        self.assertIn("circle(center = [0, 0], radius = 8)", text)

    def test_kcl_spec_is_write_only(self):
        from harnesscad.io.formats import registry as fmt

        spec = fmt.spec_for_extension(".kcl")
        self.assertTrue(spec.can_write)
        self.assertFalse(spec.can_read)
        self.assertEqual(spec.kind, "program")


if __name__ == "__main__":
    unittest.main()
