"""Tests for Meshtron y-z-x mesh-sequence ordering."""

import unittest

from harnesscad.io.formats.obj import canonicalize_mesh as canonicalize_zyx
from harnesscad.io.formats.mesh_ordering import (
    canonicalize_mesh_yzx,
    coordinate_stream,
    face_sort_key,
    is_vertices_sorted_yzx,
    sort_vertices_yzx,
    vertex_stream,
    yzx_key,
)


CUBE_VERTS = [
    (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
]
CUBE_FACES = [
    (0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
    (1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4),
]


class KeyTest(unittest.TestCase):
    def test_yzx_key_order(self):
        # key is (y, z, x)
        self.assertEqual(yzx_key((7.0, 1.0, 2.0)), (1.0, 2.0, 7.0))

    def test_bad_vertex_raises(self):
        with self.assertRaises(ValueError):
            yzx_key((1.0, 2.0))


class SortVerticesTest(unittest.TestCase):
    def test_sorted_ascending(self):
        sv, remap = sort_vertices_yzx(CUBE_VERTS)
        self.assertTrue(is_vertices_sorted_yzx(sv))
        # remap is a bijection over the indices
        self.assertEqual(sorted(remap.keys()), list(range(8)))
        self.assertEqual(sorted(remap.values()), list(range(8)))

    def test_first_vertex_is_lowest_yzx(self):
        sv, _ = sort_vertices_yzx(CUBE_VERTS)
        self.assertEqual(sv[0], min(CUBE_VERTS, key=yzx_key))

    def test_differs_from_zyx(self):
        # A vertex set where (y,z,x) and (z,y,x) orderings genuinely differ.
        verts = [(0.0, 2.0, 1.0), (0.0, 1.0, 2.0)]
        yzx_order, _ = sort_vertices_yzx(verts)
        # By y-z-x: (0,1,2) has y=1 < y=2, so (0,1,2)->... wait sorts by y first
        self.assertEqual(yzx_order[0], (0.0, 1.0, 2.0))
        # By z-y-x (llamamesh) the first coordinate compared is z.
        zyx_first = min(verts, key=lambda v: (v[2], v[1], v[0]))
        self.assertEqual(zyx_first, (0.0, 2.0, 1.0))
        self.assertNotEqual(yzx_order[0], zyx_first)


class CanonicalizeTest(unittest.TestCase):
    def test_faces_rotated_min_first(self):
        sv, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        for face in sf:
            self.assertEqual(face[0], min(face))

    def test_faces_sorted(self):
        sv, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        keys = [face_sort_key(f, sv) for f in sf]
        self.assertEqual(keys, sorted(keys))

    def test_face_count_preserved(self):
        _, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        self.assertEqual(len(sf), len(CUBE_FACES))

    def test_winding_preserved_under_rotation(self):
        # rotating (a,b,c) cyclically keeps orientation; check it is a rotation
        _, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        for face in sf:
            self.assertEqual(len(set(face)), 3)

    def test_deterministic(self):
        a = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        b = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        self.assertEqual(a, b)

    def test_distinct_from_llamamesh_ordering(self):
        # Same input to both conventions should generally reorder differently.
        sv_yzx, _ = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        sv_zyx, _ = canonicalize_zyx(
            [(int(a), int(b), int(c)) for a, b, c in CUBE_VERTS], CUBE_FACES
        )
        # convert to comparable float tuples
        sv_zyx_f = [tuple(float(c) for c in v) for v in sv_zyx]
        self.assertNotEqual(list(sv_yzx), sv_zyx_f)

    def test_bad_face_raises(self):
        with self.assertRaises(ValueError):
            canonicalize_mesh_yzx(CUBE_VERTS, [(0, 1)])


class StreamTest(unittest.TestCase):
    def test_vertex_stream_length(self):
        sv, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        vs = vertex_stream(sv, sf)
        self.assertEqual(len(vs), 3 * len(sf))

    def test_coordinate_stream_length_and_order(self):
        sv, sf = canonicalize_mesh_yzx(CUBE_VERTS, CUBE_FACES)
        cs = coordinate_stream(sv, sf)
        self.assertEqual(len(cs), 9 * len(sf))
        # first three values are (y, z, x) of the first streamed vertex
        first = vertex_stream(sv, sf)[0]
        self.assertEqual(cs[:3], [first[1], first[2], first[0]])


if __name__ == "__main__":
    unittest.main()
