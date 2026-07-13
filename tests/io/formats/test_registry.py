"""Tests for the format registry -- the unified I/O surface.

Deterministic, stdlib-only, every artefact written to a TemporaryDirectory.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from harnesscad import registry as capabilities
from harnesscad.core import cli
from harnesscad.core.loop import HarnessSession
from harnesscad.io.backends.stub import StubBackend
from harnesscad.io.formats import registry as fmt
from harnesscad.io.formats import stl as stl_codec
from harnesscad.io.formats import svg as svg_codec


def unit_cube_mesh() -> fmt.Mesh:
    """A closed 1x1x1 cube: 8 vertices, 12 triangles. Fixed order, no randomness."""
    v = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # front
        (1, 2, 6, 5),  # right
        (2, 3, 7, 6),  # back
        (3, 0, 4, 7),  # left
    ]
    return fmt.Mesh.from_vertices_faces(v, quads, name="cube")


class TestDiscovery(unittest.TestCase):
    def test_discovers_real_codecs_through_the_capability_registry(self):
        specs = fmt.specs(refresh=True)
        self.assertGreaterEqual(len(specs), 6)
        tagged = {e.dotted for e in capabilities.find(tag="format")}
        for spec in specs:
            self.assertIn(spec.dotted, tagged, f"{spec.name} is not registry-discovered")
        names = {s.name for s in specs}
        for expected in ("stl", "obj", "glb", "amf", "step", "xcsg", "svg"):
            self.assertIn(expected, names)

    def test_specs_are_deterministic_and_sorted(self):
        first = fmt.specs(refresh=True)
        second = fmt.specs(refresh=True)
        self.assertEqual([s.name for s in first], [s.name for s in second])
        self.assertEqual([s.name for s in first], sorted(s.name for s in first))

    def test_supported_filters_by_kind_and_mode(self):
        mesh_writers = fmt.supported(kind="mesh", mode="write")
        self.assertGreaterEqual(len(mesh_writers), 4)
        self.assertTrue(all(s.kind == "mesh" and s.can_write for s in mesh_writers))
        readers = fmt.supported(mode="read")
        self.assertTrue(all(s.can_read for s in readers))
        self.assertNotIn("svg", {s.name for s in readers})
        with self.assertRaises(ValueError):
            fmt.supported(mode="append")


class TestCapabilityMatrixHonesty(unittest.TestCase):
    def test_every_claimed_capability_exists_in_the_codec_api(self):
        """A spec may only claim read/write if the codec really exposes the
        functions the adapter calls."""
        for spec in fmt.specs():
            adapter = fmt._ADAPTERS[spec.dotted]
            module = capabilities.load(spec.dotted)
            if spec.can_read:
                self.assertTrue(adapter.read_symbols)
                for sym in adapter.read_symbols:
                    self.assertTrue(callable(getattr(module, sym, None)),
                                    f"{spec.name}.{sym} is not a real public function")
            if spec.can_write:
                self.assertTrue(adapter.write_symbols)
                for sym in adapter.write_symbols:
                    self.assertTrue(callable(getattr(module, sym, None)),
                                    f"{spec.name}.{sym} is not a real public function")

    def test_svg_is_write_only_and_the_codec_confirms_it(self):
        spec = fmt.spec_for_extension(".svg")
        self.assertTrue(spec.can_write)
        self.assertFalse(spec.can_read)
        self.assertFalse(spec.round_trip)
        # The codec genuinely has no reader: nothing named parse*/load*/read*.
        public = [n for n in dir(svg_codec) if not n.startswith("_")]
        self.assertEqual(
            [n for n in public if n.startswith(("parse_", "load", "read"))], [])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "drawing.svg")
            spec.write(unit_cube_mesh(), path)
            self.assertTrue(os.path.getsize(path) > 0)
            with self.assertRaises(fmt.UnsupportedOperationError):
                spec.read(path)
            with self.assertRaises(fmt.UnsupportedOperationError):
                fmt.read(path)

    def test_dxf_is_a_contract_only_module(self):
        spec = fmt.spec_for_extension(".dxf")
        self.assertFalse(spec.can_read)
        self.assertFalse(spec.can_write)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fmt.UnsupportedOperationError):
                fmt.write(unit_cube_mesh(), os.path.join(tmp, "d.dxf"))

    def test_report_counts_match_the_matrix(self):
        report = fmt.format_report()
        rows = report["formats"]
        self.assertEqual(report["counts"]["total"], len(rows))
        self.assertEqual(report["counts"]["readable"],
                         sum(1 for r in rows if r["read"]))
        self.assertEqual(report["counts"]["writable"],
                         sum(1 for r in rows if r["write"]))
        self.assertEqual(report["counts"]["round_trip"],
                         sum(1 for r in rows if r["round_trip"]))
        for r in rows:
            if r["round_trip"]:
                self.assertTrue(r["read"] and r["write"])
        json.dumps(report)  # the report is JSON-serialisable


class TestDispatch(unittest.TestCase):
    def test_dispatch_by_extension(self):
        self.assertEqual(fmt.spec_for_path("/x/y/part.stl").name, "stl")
        self.assertEqual(fmt.spec_for_path("part.STL").name, "stl")
        self.assertEqual(fmt.spec_for_path("part.stp").name, "step")
        self.assertEqual(fmt.spec_for_extension("obj").name, "obj")

    def test_unknown_extension_raises_typed_error(self):
        with self.assertRaises(fmt.UnknownFormatError):
            fmt.spec_for_path("part.wibble")
        with self.assertRaises(fmt.UnknownFormatError):
            fmt.spec_for_path("part")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fmt.UnknownFormatError):
                fmt.write(unit_cube_mesh(), os.path.join(tmp, "m.wibble"))
            with self.assertRaises(fmt.UnknownFormatError):
                fmt.read(os.path.join(tmp, "m.wibble"))
        self.assertTrue(issubclass(fmt.UnknownFormatError, fmt.FormatError))
        self.assertTrue(issubclass(fmt.UnsupportedOperationError, fmt.FormatError))


class TestMeshRoundTrip(unittest.TestCase):
    def test_stl_round_trip_preserves_triangles_and_vertices(self):
        mesh = unit_cube_mesh()
        self.assertEqual(mesh.triangle_count, 12)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cube.stl")
            self.assertEqual(fmt.write(mesh, path), path)
            back = fmt.read(path)
        self.assertEqual(back.triangle_count, 12)
        self.assertEqual(len(back.indexed()[0]), 8)
        self.assertEqual([t.vertices for t in back.triangles],
                         [t.vertices for t in mesh.triangles])

    def test_stl_ascii_round_trip(self):
        mesh = unit_cube_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cube_ascii.stl")
            fmt.write(mesh, path, ascii=True)
            with open(path, "r", encoding="utf-8") as fh:
                head = fh.read(5)
            self.assertEqual(head, "solid")
            back = fmt.read(path)
        self.assertEqual(back.triangle_count, mesh.triangle_count)

    def test_obj_glb_amf_round_trip(self):
        mesh = unit_cube_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            for ext in (".obj", ".glb", ".amf"):
                path = os.path.join(tmp, "cube" + ext)
                fmt.write(mesh, path)
                back = fmt.read(path)
                self.assertEqual(back.triangle_count, 12, ext)
                verts = back.indexed()[0]
                self.assertEqual(len(verts), 8, ext)
                self.assertEqual(sorted(verts), sorted(mesh.indexed()[0]), ext)

    def test_write_accepts_triangles_vertices_faces_and_polyhedron(self):
        mesh = unit_cube_mesh()
        verts, faces = mesh.indexed()
        candidates = [
            list(mesh.triangles),
            (verts, faces),
            mesh.to_polyhedron(),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for i, model in enumerate(candidates):
                path = os.path.join(tmp, f"m{i}.stl")
                fmt.write(model, path)
                self.assertEqual(fmt.read(path).triangle_count, 12)

    def test_bytes_are_deterministic(self):
        mesh = unit_cube_mesh()
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.stl")
            b = os.path.join(tmp, "b.stl")
            fmt.write(mesh, a)
            fmt.write(mesh, b)
            with open(a, "rb") as fh:
                da = fh.read()
            with open(b, "rb") as fh:
                db = fh.read()
        self.assertEqual(da, db)


class TestBrepAndCsg(unittest.TestCase):
    STEP_TEXT = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('a part'),'2;1');\n"
        "FILE_NAME('part.step','',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#1=CARTESIAN_POINT('',(0.,0.,0.));\n"
        "#2=DIRECTION('',(0.,0.,1.));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )

    def test_step_text_round_trips_through_the_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.step")
            fmt.write(self.STEP_TEXT, path)
            step_file = fmt.read(path)
            self.assertEqual(sorted(step_file.entities), [1, 2])
            again = os.path.join(tmp, "again.stp")
            fmt.write(step_file, again)
            self.assertEqual(sorted(fmt.read(again).entities), [1, 2])

    def test_xcsg_tree_round_trips(self):
        from harnesscad.domain.programs.ast.typed_csg import Node

        tree = Node("difference3d", {}, (
            Node("cuboid", {"dx": 10.0, "dy": 10.0, "dz": 10.0, "center": True}),
            Node("sphere", {"r": 6.0}),
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tree.xcsg")
            fmt.write(tree, path)
            back = fmt.read(path)
        self.assertEqual(back.op, "difference3d")
        self.assertEqual([c.op for c in back.children], ["cuboid", "sphere"])

    def test_wrong_object_for_a_format_raises_export_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fmt.ExportError):
                fmt.write(object(), os.path.join(tmp, "x.stl"))
            with self.assertRaises(fmt.ExportError):
                fmt.write(unit_cube_mesh(), os.path.join(tmp, "x.xcsg"))


class TestSessionExport(unittest.TestCase):
    OPS = [
        {"op": "new_sketch", "plane": "XY"},
        {"op": "add_rectangle", "sketch": "sk1", "x": 0.0, "y": 0.0, "w": 20.0, "h": 10.0},
        {"op": "extrude", "sketch": "sk1", "distance": 5.0},
    ]

    def test_session_has_an_export_path_wired_to_the_registry(self):
        session = HarnessSession(StubBackend())
        self.assertTrue(hasattr(session, "export"))
        with tempfile.TemporaryDirectory() as tmp:
            # Unknown extensions are rejected before any backend work happens.
            with self.assertRaises(fmt.UnknownFormatError):
                session.export(os.path.join(tmp, "part.wibble"))

    def test_stub_backend_export_fails_honestly_rather_than_writing_a_fake_file(self):
        """The stub has no geometry; the surface must not pretend otherwise."""
        from harnesscad.core.cisp.ops import parse_op

        session = HarnessSession(StubBackend())
        result = session.apply_ops([parse_op(o) for o in self.OPS])
        self.assertTrue(result.ok)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "part.stl")
            with self.assertRaises(fmt.ExportError):
                session.export(path)
            self.assertFalse(os.path.exists(path))

    def test_mesh_from_a_backendlike_object_parses_real_stl(self):
        """A backend whose export('stl') is real STL is meshed by the surface."""
        mesh = unit_cube_mesh()
        ascii_stl = stl_codec.write_ascii_stl(mesh.triangles, name="cube")

        class FakeBackend:
            def export(self, fmt_name):
                return ascii_stl

            def state_digest(self):
                return "deadbeef"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "from_backend.stl")
            fmt.write(FakeBackend(), path)
            self.assertEqual(fmt.read(path).triangle_count, 12)


class TestCli(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(argv)
        return code, buf.getvalue()

    def test_formats_subcommand_runs(self):
        code, out = self._run(["formats"])
        self.assertEqual(code, 0)
        self.assertIn("FORMAT", out)
        self.assertIn("stl", out)
        self.assertIn("round-trippable", out)

    def test_formats_json_and_filters(self):
        code, out = self._run(["formats", "--json"])
        self.assertEqual(code, 0)
        report = json.loads(out)
        self.assertGreaterEqual(report["counts"]["total"], 6)
        code, out = self._run(["formats", "--kind", "mesh", "--mode", "write"])
        self.assertEqual(code, 0)
        self.assertNotIn("image/svg+xml", out)

    def test_export_subcommand_reports_the_backend_limit(self):
        """`export` runs the ops, then fails with a typed error on the stub
        backend (no geometry) instead of writing a bogus file."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "demo.stl")
            code, out = self._run(["export", path, "--backend", "stub"])
            self.assertEqual(code, 2)
            self.assertFalse(os.path.exists(path))
            self.assertIn("ok:", out)

    def test_existing_subcommands_still_work(self):
        code, out = self._run(["demo"])
        self.assertEqual(code, 0)
        self.assertIn("digest:", out)
        self.assertIn("summary:", out)

        with tempfile.TemporaryDirectory() as tmp:
            ops_path = os.path.join(tmp, "ops.json")
            with open(ops_path, "w", encoding="utf-8") as fh:
                json.dump(TestSessionExport.OPS, fh)
            code, out = self._run(["apply", ops_path])
            self.assertEqual(code, 0)
            self.assertIn("applied:  3", out)

        code, out = self._run(["capabilities", "--list", "--tag", "format"])
        self.assertEqual(code, 0)
        self.assertIn("harnesscad.io.formats.stl", out)

    def test_build_subcommand_is_still_registered(self):
        parser = cli.build_parser()
        args = parser.parse_args(["build", "a plate", "--out", "p.step"])
        self.assertEqual(args.command, "build")
        self.assertEqual(args.out, "p.step")


if __name__ == "__main__":
    unittest.main()
