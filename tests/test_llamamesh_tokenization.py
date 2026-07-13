"""Tests for LLaMA-Mesh mesh-as-text tokenization."""

import unittest

from harnesscad.io.formats.llamamesh_tokenization import (
    canonicalize_mesh,
    compression_ratio,
    dequantize_vertices,
    estimate_token_count,
    mesh_bounds,
    parse_obj,
    quantize_vertices,
    roundtrip,
    serialize_obj,
    serialize_obj_float,
)


# A unit cube (8 vertices, 12 triangles) used across several tests.
CUBE_VERTS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
    (1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4),
]


class BoundsAndQuantizeTest(unittest.TestCase):
    def test_bounds(self):
        lo, hi = mesh_bounds(CUBE_VERTS)
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 1.0, 1.0))

    def test_empty_bounds_raises(self):
        with self.assertRaises(ValueError):
            mesh_bounds([])

    def test_quantize_range(self):
        q, bbox = quantize_vertices(CUBE_VERTS, bins=64)
        self.assertEqual(bbox["bins"], 64)
        for vertex in q:
            for c in vertex:
                self.assertIsInstance(c, int)
                self.assertGreaterEqual(c, 0)
                self.assertLessEqual(c, 64)

    def test_quantize_maps_corners(self):
        q, _ = quantize_vertices(CUBE_VERTS, bins=64)
        # min corner -> 0, opposite corner -> 64 (uniform, cube extent==1)
        self.assertEqual(q[0], (0, 0, 0))
        self.assertEqual(q[6], (64, 64, 64))

    def test_bins_must_be_positive(self):
        with self.assertRaises(ValueError):
            quantize_vertices(CUBE_VERTS, bins=0)

    def test_uniform_scaling_preserves_aspect(self):
        # A flat, wide mesh: x-extent 4, y-extent 2, z-extent 0.
        verts = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0)]
        q, _ = quantize_vertices(verts, bins=64)
        # Uniform scale = 64/4 = 16 -> x maps to 0..64, y to 0..32.
        self.assertEqual(q[1], (64, 0, 0))
        self.assertEqual(q[2], (64, 32, 0))

    def test_degenerate_mesh(self):
        verts = [(3.0, 3.0, 3.0)] * 3
        q, bbox = quantize_vertices(verts, bins=64)
        self.assertEqual(q, [(0, 0, 0), (0, 0, 0), (0, 0, 0)])
        deq = dequantize_vertices(q, bbox)
        self.assertEqual(deq[0], (3.0, 3.0, 3.0))

    def test_dequantize_recovers_corners(self):
        q, bbox = quantize_vertices(CUBE_VERTS, bins=64)
        deq = dequantize_vertices(q, bbox)
        self.assertAlmostEqual(deq[0][0], 0.0, places=6)
        self.assertAlmostEqual(deq[6][0], 1.0, places=6)


class CanonicalOrderTest(unittest.TestCase):
    def test_vertices_sorted_z_y_x(self):
        verts = [(1, 1, 1), (0, 0, 0), (0, 1, 0), (1, 0, 0)]
        faces = [(0, 1, 2)]
        sverts, _ = canonicalize_mesh(verts, faces)
        # ascending by (z, y, x)
        self.assertEqual(sverts[0], (0, 0, 0))
        self.assertEqual(sverts[-1], (1, 1, 1))
        keys = [(v[2], v[1], v[0]) for v in sverts]
        self.assertEqual(keys, sorted(keys))

    def test_faces_reindexed_correctly(self):
        verts = [(2, 0, 0), (0, 0, 0), (1, 0, 0)]  # x order 1,2,0
        faces = [(0, 1, 2)]
        sverts, sfaces = canonicalize_mesh(verts, faces)
        # vertex (0,0,0) is new index 0, (1,0,0)->1, (2,0,0)->2
        self.assertEqual(sverts, [(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        # face rotated so min index leads, winding preserved
        self.assertEqual(sfaces[0][0], min(sfaces[0]))

    def test_face_rotation_preserves_winding(self):
        verts = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        faces = [(2, 0, 1)]
        _, sfaces = canonicalize_mesh(verts, faces)
        # (2,0,1) rotated so 0 leads -> (0,1,2), cyclic order preserved
        self.assertEqual(sfaces[0], (0, 1, 2))

    def test_faces_sorted_by_lowest_index(self):
        verts = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        faces = [(2, 3, 1), (0, 1, 2)]
        _, sfaces = canonicalize_mesh(verts, faces)
        keys = [sorted(f) for f in sfaces]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(sfaces[0][0], 0)

    def test_degenerate_face_raises(self):
        with self.assertRaises(ValueError):
            canonicalize_mesh([(0, 0, 0), (1, 0, 0)], [(0, 1)])

    def test_canonical_is_idempotent(self):
        q, _ = quantize_vertices(CUBE_VERTS, bins=64)
        v1, f1 = canonicalize_mesh(q, CUBE_FACES)
        v2, f2 = canonicalize_mesh(v1, f1)
        self.assertEqual(v1, v2)
        self.assertEqual(f1, f2)


class SerializeParseTest(unittest.TestCase):
    def test_serialize_format(self):
        text = serialize_obj([(0, 0, 0), (1, 2, 3)], [(0, 1, 1)])
        lines = text.strip().splitlines()
        self.assertEqual(lines[0], "v 0 0 0")
        self.assertEqual(lines[1], "v 1 2 3")
        # faces are 1-based
        self.assertEqual(lines[2], "f 1 2 2")

    def test_parse_basic(self):
        text = "v 0 0 0\nv 1 2 3\nf 1 2 2\n"
        verts, faces = parse_obj(text)
        self.assertEqual(verts, [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0)])
        self.assertEqual(faces, [(0, 1, 1)])

    def test_parse_ignores_comments_and_other_tags(self):
        text = "# comment\nvn 0 0 1\nv 0 0 0\no obj\nv 1 1 1\nf 1 2 2\n"
        verts, faces = parse_obj(text)
        self.assertEqual(len(verts), 2)
        self.assertEqual(faces, [(0, 1, 1)])

    def test_parse_face_slash_form(self):
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"
        _, faces = parse_obj(text)
        self.assertEqual(faces, [(0, 1, 2)])

    def test_parse_rejects_short_vertex(self):
        with self.assertRaises(ValueError):
            parse_obj("v 0 0\n")

    def test_parse_rejects_short_face(self):
        with self.assertRaises(ValueError):
            parse_obj("v 0 0 0\nf 1 2\n")

    def test_float_serialize_precision(self):
        text = serialize_obj_float([(0.123456789, 0.0, 0.0)], [], precision=3)
        self.assertEqual(text.strip(), "v 0.123 0.000 0.000")


class RoundTripTest(unittest.TestCase):
    def test_roundtrip_recovers_canonical_int_mesh(self):
        int_verts, faces, text, bbox = roundtrip(CUBE_VERTS, CUBE_FACES, bins=64)
        # Re-quantize+canonicalize independently and compare.
        q, _ = quantize_vertices(CUBE_VERTS, bins=64)
        cverts, cfaces = canonicalize_mesh(q, CUBE_FACES)
        cverts = [tuple(v) for v in cverts]
        self.assertEqual(int_verts, cverts)
        self.assertEqual(faces, cfaces)
        self.assertIn("v ", text)

    def test_roundtrip_text_parses_back_equal(self):
        _, _, text, _ = roundtrip(CUBE_VERTS, CUBE_FACES, bins=32)
        verts, faces = parse_obj(text)
        # Serialize the parsed mesh again -> identical text (stable round-trip).
        int_verts = [tuple(int(round(c)) for c in v) for v in verts]
        text2 = serialize_obj(int_verts, faces)
        self.assertEqual(text, text2)

    def test_roundtrip_face_count_preserved(self):
        _, faces, _, _ = roundtrip(CUBE_VERTS, CUBE_FACES, bins=64)
        self.assertEqual(len(faces), len(CUBE_FACES))


class TokenMetricTest(unittest.TestCase):
    def test_token_count_positive(self):
        text = serialize_obj([(0, 0, 0)], [])
        self.assertGreater(estimate_token_count(text), 0)

    def test_integer_shorter_than_float(self):
        q, _ = quantize_vertices(CUBE_VERTS, bins=64)
        cverts, cfaces = canonicalize_mesh(q, CUBE_FACES)
        quant_text = serialize_obj(cverts, cfaces)
        float_text = serialize_obj_float(CUBE_VERTS, CUBE_FACES, precision=6)
        self.assertLess(estimate_token_count(quant_text),
                        estimate_token_count(float_text))

    def test_compression_ratio_gt_one(self):
        ratio = compression_ratio(CUBE_VERTS, CUBE_FACES, bins=64, precision=6)
        self.assertGreater(ratio, 1.0)

    def test_compression_deterministic(self):
        r1 = compression_ratio(CUBE_VERTS, CUBE_FACES)
        r2 = compression_ratio(CUBE_VERTS, CUBE_FACES)
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
