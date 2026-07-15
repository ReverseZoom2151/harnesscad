"""Tests for formats.threemf -- 3MF OPC reader/writer and its round trip."""

import io
import os
import tempfile
import unittest
import zipfile

from harnesscad.io.formats import threemf
from harnesscad.domain.geometry.mesh.polyhedron import Polyhedron, unit_cube, verify


def _box(sx, sy, sz):
    c = unit_cube(1.0)
    verts = [(x * sx, y * sy, z * sz) for (x, y, z) in c.vertices]
    return Polyhedron(verts, c.faces)


class TestModelXml(unittest.TestCase):
    def test_geometry_unit_colour_round_trip(self):
        c = unit_cube(3.0)
        xml = threemf.dumps_model(c.vertices, c.faces, unit="inch",
                                  color="#ff8800")
        v, t, unit, color = threemf.loads_model(xml)
        p = Polyhedron(v, t)
        self.assertAlmostEqual(p.volume(), 27.0)
        self.assertAlmostEqual(p.surface_area(), 54.0)
        self.assertEqual(verify(p), [])
        self.assertEqual(unit, "inch")
        self.assertEqual(color, "#FF8800")

    def test_default_unit_is_millimeter(self):
        c = unit_cube()
        _, _, unit, _ = threemf.loads_model(
            threemf.dumps_model(c.vertices, c.faces))
        self.assertEqual(unit, "millimeter")

    def test_no_colour_by_default(self):
        c = unit_cube()
        _, _, _, color = threemf.loads_model(
            threemf.dumps_model(c.vertices, c.faces))
        self.assertIsNone(color)

    def test_axis_convention_no_rotation(self):
        box = _box(1.0, 2.0, 4.0)
        xml = threemf.dumps_model(box.vertices, box.faces)
        v, t, _, _ = threemf.loads_model(xml)
        lo, hi = Polyhedron(v, t).bounds()
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (1.0, 2.0, 4.0))

    def test_orientation_preserved(self):
        c = unit_cube(2.0)
        v, t, _, _ = threemf.loads_model(
            threemf.dumps_model(c.vertices, c.faces))
        self.assertGreater(Polyhedron(v, t).volume(), 0.0)

    def test_deterministic(self):
        c = unit_cube()
        a = threemf.dumps_model(c.vertices, c.faces, color="#112233")
        b = threemf.dumps_model(c.vertices, c.faces, color="#112233")
        self.assertEqual(a, b)


class TestZipPackage(unittest.TestCase):
    def test_is_opc_zip(self):
        c = unit_cube()
        data = threemf.serialize(c.vertices, c.faces)
        self.assertTrue(data.startswith(b"PK\x03\x04"))
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
        self.assertIn("[Content_Types].xml", names)
        self.assertIn("_rels/.rels", names)
        self.assertIn("3D/3dmodel.model", names)

    def test_zip_is_reproducible(self):
        c = unit_cube(2.0)
        a = threemf.serialize(c.vertices, c.faces, color="#abcdef")
        b = threemf.serialize(c.vertices, c.faces, color="#abcdef")
        self.assertEqual(a, b)

    def test_file_round_trip(self):
        c = unit_cube(2.0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.3mf")
            threemf.write_3mf(path, c.vertices, c.faces, unit="meter",
                              color="#00ff00")
            v, t, unit, color = threemf.read_3mf(path)
        self.assertAlmostEqual(Polyhedron(v, t).volume(), 8.0)
        self.assertEqual(unit, "meter")
        self.assertEqual(color, "#00FF00")

    def test_reader_follows_rels(self):
        # Even if the model part were placed elsewhere, the reader must follow the
        # package relationship. Here we confirm the default layout reads back.
        c = unit_cube()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.3mf")
            threemf.write_3mf(path, c.vertices, c.faces)
            v, t, _, _ = threemf.read_3mf(path)
        self.assertEqual(len(v), 8)
        self.assertEqual(len(t), 12)


class TestErrors(unittest.TestCase):
    def test_bad_unit(self):
        with self.assertRaises(threemf.ThreeMFError):
            threemf.dumps_model([(0, 0, 0)], [], unit="furlong")

    def test_amf_feet_spelling_is_rejected(self):
        # AMF spells it "feet"; 3MF spells it "foot". Guard the divergence.
        with self.assertRaises(threemf.ThreeMFError):
            threemf.dumps_model([(0, 0, 0)], [], unit="feet")

    def test_bad_colour(self):
        c = unit_cube()
        with self.assertRaises(threemf.ThreeMFError):
            threemf.dumps_model(c.vertices, c.faces, color="orange")

    def test_index_out_of_range(self):
        with self.assertRaises(threemf.ThreeMFError):
            threemf.dumps_model([(0, 0, 0)], [(0, 1, 2)])

    def test_bad_root(self):
        with self.assertRaises(threemf.ThreeMFError):
            threemf.loads_model("<foo/>")

    def test_not_a_zip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.3mf")
            with open(path, "wb") as fh:
                fh.write(b"not a zip")
            with self.assertRaises(threemf.ThreeMFError):
                threemf.read_3mf(path)


if __name__ == "__main__":
    unittest.main()
