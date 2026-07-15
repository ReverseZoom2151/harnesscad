"""Tests for formats.ply -- Stanford PLY reader/writer and its round trip."""

import os
import tempfile
import unittest

from harnesscad.io.formats import ply
from harnesscad.domain.geometry.mesh.polyhedron import Polyhedron, unit_cube, verify


def _box(sx, sy, sz):
    """A unit cube scaled to distinct extents -- catches any axis swap."""
    c = unit_cube(1.0)
    verts = [(x * sx, y * sy, z * sz) for (x, y, z) in c.vertices]
    return Polyhedron(verts, c.faces)


class TestRoundTrip(unittest.TestCase):
    def test_ascii_geometry_and_unit(self):
        c = unit_cube(3.0)
        data = ply.serialize_ply(c.vertices, c.faces, unit="inch")
        v, t, unit = ply.parse_ply(data)
        p = Polyhedron(v, t)
        self.assertAlmostEqual(p.volume(), 27.0)
        self.assertAlmostEqual(p.surface_area(), 54.0)
        self.assertEqual(verify(p), [])
        self.assertEqual(unit, "inch")

    def test_binary_matches_ascii_geometry(self):
        c = unit_cube(2.0)
        va, ta, _ = ply.parse_ply(ply.serialize_ply(c.vertices, c.faces))
        vb, tb, _ = ply.parse_ply(
            ply.serialize_ply(c.vertices, c.faces, binary=True))
        self.assertEqual(va, vb)
        self.assertEqual(ta, tb)

    def test_axis_convention_no_rotation(self):
        # The glTF exporter silently rotated every part -90deg about X. A box with
        # three distinct extents makes any axis permutation observable.
        box = _box(1.0, 2.0, 4.0)
        v, t, _ = ply.parse_ply(ply.serialize_ply(box.vertices, box.faces))
        p = Polyhedron(v, t)
        lo, hi = p.bounds()
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 2.0, 4.0))

    def test_orientation_preserved(self):
        # unit_cube is outward-oriented (positive signed volume); winding must
        # survive so the sign does not flip.
        c = unit_cube(2.0)
        v, t, _ = ply.parse_ply(ply.serialize_ply(c.vertices, c.faces))
        self.assertGreater(Polyhedron(v, t).volume(), 0.0)

    def test_default_unit_is_millimeter(self):
        c = unit_cube()
        _, _, unit = ply.parse_ply(ply.serialize_ply(c.vertices, c.faces))
        self.assertEqual(unit, "millimeter")

    def test_ascii_is_deterministic(self):
        c = unit_cube()
        a = ply.serialize_ply(c.vertices, c.faces)
        b = ply.serialize_ply(c.vertices, c.faces)
        self.assertEqual(a, b)

    def test_binary_is_deterministic(self):
        c = unit_cube()
        a = ply.serialize_ply(c.vertices, c.faces, binary=True)
        b = ply.serialize_ply(c.vertices, c.faces, binary=True)
        self.assertEqual(a, b)

    def test_file_round_trip(self):
        c = unit_cube(2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.ply")
            ply.write_ply(path, c.vertices, c.faces, unit="meter")
            with open(path, "rb") as fh:
                v, t, unit = ply.parse_ply(fh.read())
        self.assertAlmostEqual(Polyhedron(v, t).volume(), 8.0)
        self.assertEqual(unit, "meter")


class TestErrors(unittest.TestCase):
    def test_not_ply(self):
        with self.assertRaises(ply.PlyError):
            ply.parse_ply(b"not a ply file at all")

    def test_index_out_of_range(self):
        with self.assertRaises(ply.PlyError):
            ply.serialize_ply([(0, 0, 0)], [(0, 1, 2)])

    def test_bad_face(self):
        with self.assertRaises(ply.PlyError):
            ply.serialize_ply([(0, 0, 0), (1, 0, 0)], [(0, 1)])

    def test_expects_bytes(self):
        with self.assertRaises(ply.PlyError):
            ply.parse_ply("ply\n")


if __name__ == "__main__":
    unittest.main()
