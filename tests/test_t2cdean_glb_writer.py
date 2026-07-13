"""Tests for formats.t2cdean_glb_writer."""

import base64
import json
import struct
import unittest

from harnesscad.io.formats.glb import (
    CHUNK_BIN,
    CHUNK_JSON,
    GLB_MAGIC,
    GlbError,
    build_gltf_json,
    glb_data_uri,
    parse_glb,
    stl_to_glb,
    triangles_from_glb,
    vertex_normals,
    weld_vertices,
    write_glb,
)
from harnesscad.io.formats.stl import Triangle, write_ascii_stl, write_binary_stl


def unit_cube_triangles():
    """Axis-aligned unit cube: 8 distinct corners, 12 triangles, 36 soup verts."""
    c = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    quads = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    tris = []
    for a, b, d, e in quads:
        tris.append(Triangle(c[a], c[b], c[d]))
        tris.append(Triangle(c[a], c[d], c[e]))
    return tris


def one_triangle():
    return [Triangle((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]


class TestWeld(unittest.TestCase):
    def test_cube_welds_to_eight_vertices(self):
        verts, idx = weld_vertices(unit_cube_triangles())
        self.assertEqual(len(verts), 8)
        self.assertEqual(len(idx), 36)
        self.assertEqual(max(idx), 7)

    def test_weld_is_deterministic_and_first_seen_ordered(self):
        tris = unit_cube_triangles()
        v1, i1 = weld_vertices(tris)
        v2, i2 = weld_vertices(tris)
        self.assertEqual(v1, v2)
        self.assertEqual(i1, i2)
        self.assertEqual(v1[0], tris[0].v0)

    def test_near_duplicates_merge_within_tolerance(self):
        tris = [
            Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            Triangle((1e-9, 0, 0), (1, 0, 0), (0, 0, 1)),
        ]
        verts, _ = weld_vertices(tris, tolerance=1e-6)
        self.assertEqual(len(verts), 4)

    def test_distinct_beyond_tolerance_kept(self):
        tris = [
            Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0)),
            Triangle((0.5, 0, 0), (1, 0, 0), (0, 0, 1)),
        ]
        verts, _ = weld_vertices(tris, tolerance=1e-6)
        self.assertEqual(len(verts), 5)

    def test_bad_tolerance(self):
        with self.assertRaises(GlbError):
            weld_vertices(one_triangle(), tolerance=0.0)


class TestNormals(unittest.TestCase):
    def test_flat_triangle_normal(self):
        verts, idx = weld_vertices(one_triangle())
        normals = vertex_normals(verts, idx)
        self.assertEqual(len(normals), 3)
        for n in normals:
            self.assertAlmostEqual(n[0], 0.0)
            self.assertAlmostEqual(n[1], 0.0)
            self.assertAlmostEqual(n[2], 1.0)

    def test_normals_are_unit_length(self):
        verts, idx = weld_vertices(unit_cube_triangles())
        for n in vertex_normals(verts, idx):
            ln = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
            self.assertAlmostEqual(ln, 1.0, places=9)

    def test_isolated_vertex_gets_fallback(self):
        normals = vertex_normals([(0.0, 0.0, 0.0)], [])
        self.assertEqual(normals, [(0.0, 0.0, 1.0)])


class TestGlbContainer(unittest.TestCase):
    def test_header_and_chunks(self):
        data = write_glb(unit_cube_triangles())
        magic, version, total = struct.unpack_from("<III", data, 0)
        self.assertEqual(magic, GLB_MAGIC)
        self.assertEqual(version, 2)
        self.assertEqual(total, len(data))
        jlen, jtype = struct.unpack_from("<II", data, 12)
        self.assertEqual(jtype, CHUNK_JSON)
        blen, btype = struct.unpack_from("<II", data, 12 + 8 + jlen)
        self.assertEqual(btype, CHUNK_BIN)
        self.assertEqual(len(data), 12 + 8 + jlen + 8 + blen)

    def test_chunks_are_four_byte_aligned(self):
        # An odd-length name forces JSON padding.
        data = write_glb(one_triangle(), name="odd_name_x")
        jlen, _ = struct.unpack_from("<II", data, 12)
        blen, _ = struct.unpack_from("<II", data, 12 + 8 + jlen)
        self.assertEqual(jlen % 4, 0)
        self.assertEqual(blen % 4, 0)

    def test_json_chunk_padded_with_spaces(self):
        data = write_glb(one_triangle(), name="pad")
        jlen, _ = struct.unpack_from("<II", data, 12)
        chunk = data[20 : 20 + jlen]
        self.assertTrue(chunk.rstrip(b" ").endswith(b"}"))
        json.loads(chunk.decode("utf-8"))  # must still parse

    def test_output_is_byte_deterministic(self):
        tris = unit_cube_triangles()
        self.assertEqual(write_glb(tris), write_glb(tris))

    def test_empty_mesh_rejected(self):
        with self.assertRaises(GlbError):
            write_glb([])


class TestGltfDocument(unittest.TestCase):
    def test_position_accessor_has_min_max(self):
        doc, _ = parse_glb(write_glb(unit_cube_triangles()))
        prim = doc["meshes"][0]["primitives"][0]
        pos = doc["accessors"][prim["attributes"]["POSITION"]]
        self.assertEqual(pos["min"], [0.0, 0.0, 0.0])
        self.assertEqual(pos["max"], [1.0, 1.0, 1.0])
        self.assertEqual(pos["count"], 8)
        self.assertEqual(pos["type"], "VEC3")

    def test_index_accessor_and_mode(self):
        doc, _ = parse_glb(write_glb(unit_cube_triangles()))
        prim = doc["meshes"][0]["primitives"][0]
        self.assertEqual(prim["mode"], 4)
        idx = doc["accessors"][prim["indices"]]
        self.assertEqual(idx["count"], 36)
        self.assertEqual(idx["type"], "SCALAR")
        self.assertEqual(idx["componentType"], 5125)

    def test_buffer_view_offsets_are_contiguous_and_aligned(self):
        doc, binary = parse_glb(write_glb(unit_cube_triangles()))
        expect = 0
        for view in doc["bufferViews"]:
            self.assertEqual(view["byteOffset"], expect)
            self.assertEqual(view["byteOffset"] % 4, 0)
            expect += view["byteLength"]
        self.assertEqual(doc["buffers"][0]["byteLength"], expect)
        self.assertLessEqual(expect, len(binary))

    def test_asset_version(self):
        doc, _ = parse_glb(write_glb(one_triangle()))
        self.assertEqual(doc["asset"]["version"], "2.0")
        self.assertEqual(doc["scene"], 0)

    def test_no_normals_when_disabled(self):
        doc, _ = parse_glb(write_glb(one_triangle(), smooth_normals=False))
        prim = doc["meshes"][0]["primitives"][0]
        self.assertNotIn("NORMAL", prim["attributes"])
        self.assertEqual(len(doc["bufferViews"]), 2)

    def test_build_gltf_json_reports_buffer_length(self):
        doc = build_gltf_json([(0.0, 0.0, 0.0)], [0, 0, 0], None, 123, name="m")
        self.assertEqual(doc["buffers"][0]["byteLength"], 123)
        self.assertEqual(doc["nodes"][0]["name"], "m")


class TestParseGlb(unittest.TestCase):
    def test_bad_magic(self):
        with self.assertRaises(GlbError):
            parse_glb(b"XXXX" + b"\x00" * 20)

    def test_truncated(self):
        with self.assertRaises(GlbError):
            parse_glb(b"abc")

    def test_length_mismatch(self):
        data = bytearray(write_glb(one_triangle()))
        data[8:12] = struct.pack("<I", len(data) + 4)
        with self.assertRaises(GlbError):
            parse_glb(bytes(data))

    def test_unsupported_version(self):
        data = bytearray(write_glb(one_triangle()))
        data[4:8] = struct.pack("<I", 1)
        with self.assertRaises(GlbError):
            parse_glb(bytes(data))


class TestRoundTrip(unittest.TestCase):
    def test_triangles_survive_glb_round_trip(self):
        tris = unit_cube_triangles()
        back = triangles_from_glb(write_glb(tris))
        self.assertEqual(len(back), len(tris))
        for got, want in zip(back, tris):
            for gv, wv in zip(got.vertices, want.vertices):
                for g, w in zip(gv, wv):
                    self.assertAlmostEqual(g, w, places=6)

    def test_stl_to_glb_from_binary_stl(self):
        glb = stl_to_glb(write_binary_stl(unit_cube_triangles()))
        doc, _ = parse_glb(glb)
        pos = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]
        self.assertEqual(pos["count"], 8)

    def test_stl_to_glb_from_ascii_stl(self):
        glb = stl_to_glb(write_ascii_stl(unit_cube_triangles()).encode("utf-8"))
        self.assertEqual(len(triangles_from_glb(glb)), 12)


class TestDataUri(unittest.TestCase):
    def test_data_uri_prefix_and_payload(self):
        glb = write_glb(one_triangle())
        uri = glb_data_uri(glb)
        self.assertTrue(uri.startswith("data:model/gltf-binary;base64,"))
        payload = uri.split(",", 1)[1]
        self.assertEqual(base64.b64decode(payload), glb)


if __name__ == "__main__":
    unittest.main()
