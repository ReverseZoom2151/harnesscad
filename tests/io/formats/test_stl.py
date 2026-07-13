"""Tests for formats.t2cdean_stl_codec."""

import struct
import unittest

from harnesscad.io.formats.stl import (
    StlError,
    Triangle,
    bounding_box,
    face_normal,
    is_binary_stl,
    parse_ascii_stl,
    parse_binary_stl,
    parse_stl,
    signed_volume,
    surface_area,
    write_ascii_stl,
    write_binary_stl,
)


def unit_tetra():
    """Closed outward-oriented tetrahedron with volume 1/6."""
    a = (0.0, 0.0, 0.0)
    b = (1.0, 0.0, 0.0)
    c = (0.0, 1.0, 0.0)
    d = (0.0, 0.0, 1.0)
    return [
        Triangle(a, c, b),
        Triangle(a, b, d),
        Triangle(a, d, c),
        Triangle(b, c, d),
    ]


class TestTriangle(unittest.TestCase):
    def test_normal_recomputed_when_zero(self):
        t = Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), normal=(0, 0, 0))
        self.assertEqual(t.normal, (0.0, 0.0, 1.0))

    def test_stored_normal_kept(self):
        t = Triangle((0, 0, 0), (1, 0, 0), (0, 1, 0), normal=(0, 0, -1))
        self.assertEqual(t.normal, (0.0, 0.0, -1.0))

    def test_area_and_degeneracy(self):
        t = Triangle((0, 0, 0), (2, 0, 0), (0, 2, 0))
        self.assertAlmostEqual(t.area(), 2.0)
        self.assertFalse(t.is_degenerate())
        flat = Triangle((0, 0, 0), (1, 0, 0), (2, 0, 0))
        self.assertTrue(flat.is_degenerate())
        self.assertEqual(flat.normal, (0.0, 0.0, 0.0))

    def test_bad_vector_length(self):
        with self.assertRaises(StlError):
            Triangle((0, 0), (1, 0, 0), (0, 1, 0))

    def test_face_normal_right_hand_rule(self):
        self.assertEqual(face_normal((0, 0, 0), (0, 1, 0), (1, 0, 0)), (0.0, 0.0, -1.0))


class TestDetection(unittest.TestCase):
    def test_binary_detected_by_length(self):
        data = write_binary_stl(unit_tetra())
        self.assertTrue(is_binary_stl(data))
        self.assertEqual(len(data), 84 + 50 * 4)

    def test_binary_header_starting_with_solid_still_binary(self):
        data = write_binary_stl(unit_tetra(), header=b"solid exported by something")
        self.assertTrue(data.startswith(b"solid"))
        self.assertTrue(is_binary_stl(data))
        self.assertEqual(len(parse_stl(data)), 4)

    def test_ascii_not_binary(self):
        text = write_ascii_stl(unit_tetra()).encode("utf-8")
        self.assertFalse(is_binary_stl(text))

    def test_short_buffer_not_binary(self):
        self.assertFalse(is_binary_stl(b"solid x"))


class TestParsing(unittest.TestCase):
    def test_binary_round_trip(self):
        # Binary STL stores float32, so compare within single-precision epsilon.
        tris = unit_tetra()
        back = parse_binary_stl(write_binary_stl(tris))
        self.assertEqual(len(back), len(tris))
        for got, want in zip(back, tris):
            for gv, wv in zip(got.vertices + (got.normal,), want.vertices + (want.normal,)):
                for g, w in zip(gv, wv):
                    self.assertAlmostEqual(g, w, places=6)

    def test_ascii_round_trip(self):
        tris = unit_tetra()
        back = parse_ascii_stl(write_ascii_stl(tris))
        self.assertEqual(len(back), 4)
        for got, want in zip(back, tris):
            for gv, wv in zip(got.vertices, want.vertices):
                for g, w in zip(gv, wv):
                    self.assertAlmostEqual(g, w, places=5)

    def test_parse_stl_dispatches(self):
        tris = unit_tetra()
        self.assertEqual(len(parse_stl(write_binary_stl(tris))), 4)
        self.assertEqual(len(parse_stl(write_ascii_stl(tris).encode("utf-8"))), 4)

    def test_ascii_without_normal_keyword(self):
        text = (
            "solid s\n"
            "facet\n outer loop\n"
            "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n"
            " endloop\nendfacet\nendsolid s\n"
        )
        tris = parse_ascii_stl(text)
        self.assertEqual(len(tris), 1)
        self.assertEqual(tris[0].normal, (0.0, 0.0, 1.0))

    def test_ascii_must_start_with_solid(self):
        with self.assertRaises(StlError):
            parse_ascii_stl("facet normal 0 0 1\n")

    def test_ascii_wrong_vertex_count(self):
        text = "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nendloop\nendfacet\nendsolid s\n"
        with self.assertRaises(StlError):
            parse_ascii_stl(text)

    def test_binary_truncated(self):
        data = write_binary_stl(unit_tetra())
        with self.assertRaises(StlError):
            parse_binary_stl(data[:-10])

    def test_declared_count_mismatch(self):
        data = bytearray(write_binary_stl(unit_tetra()))
        data[80:84] = struct.pack("<I", 9)
        with self.assertRaises(StlError):
            parse_stl(bytes(data))

    def test_parse_stl_rejects_str(self):
        with self.assertRaises(StlError):
            parse_stl("solid s\nendsolid s\n")


class TestWriting(unittest.TestCase):
    def test_ascii_is_deterministic(self):
        tris = unit_tetra()
        self.assertEqual(write_ascii_stl(tris), write_ascii_stl(tris))

    def test_ascii_has_no_negative_zero(self):
        tris = [Triangle((0, 0, 0), (0, 1, 0), (1, 0, 0))]
        text = write_ascii_stl(tris)
        self.assertNotIn("-0.000000", text)
        self.assertIn("endsolid model", text)

    def test_binary_header_padded_to_80(self):
        data = write_binary_stl([], header=b"hi")
        self.assertEqual(len(data), 84)
        self.assertEqual(data[:2], b"hi")
        self.assertEqual(data[2:80], b"\x00" * 78)

    def test_binary_attribute_bytes_zero(self):
        data = write_binary_stl(unit_tetra())
        (attr,) = struct.unpack_from("<H", data, 84 + 48)
        self.assertEqual(attr, 0)


class TestDerived(unittest.TestCase):
    def test_bounding_box(self):
        lo, hi = bounding_box(unit_tetra())
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 1.0, 1.0))

    def test_bounding_box_empty(self):
        with self.assertRaises(StlError):
            bounding_box([])

    def test_signed_volume_of_tetra(self):
        self.assertAlmostEqual(signed_volume(unit_tetra()), 1.0 / 6.0, places=9)

    def test_signed_volume_sign_flips_with_winding(self):
        flipped = [Triangle(t.v0, t.v2, t.v1) for t in unit_tetra()]
        self.assertAlmostEqual(signed_volume(flipped), -1.0 / 6.0, places=9)

    def test_surface_area(self):
        area = surface_area(unit_tetra())
        # 3 right triangles of area 1/2 plus the equilateral face sqrt(3)/2.
        self.assertAlmostEqual(area, 1.5 + (3 ** 0.5) / 2.0, places=9)


if __name__ == "__main__":
    unittest.main()
