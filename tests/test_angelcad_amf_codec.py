"""Tests for formats.angelcad_amf_codec."""

import os
import tempfile
import unittest

from harnesscad.io.formats.angelcad_amf_codec import (
    AmfError,
    AmfObject,
    dumps,
    from_polyhedra,
    loads,
    merge_lumps,
    read_amf,
    to_polyhedra,
    write_amf,
)
from harnesscad.domain.geometry.mesh.angelcad_polyhedron import Polyhedron, tetrahedron, unit_cube, verify


def _shifted(p, dx):
    return Polyhedron([(x + dx, y, z) for (x, y, z) in p.vertices], p.faces)


class TestWriter(unittest.TestCase):
    def test_document_shape(self):
        xml = dumps(from_polyhedra([tetrahedron()]), metadata={"name": "tet"})
        self.assertTrue(xml.startswith('<?xml version="1.0" encoding="UTF-8"?>'))
        self.assertIn('<amf unit="millimeter" version="1.1">', xml)
        self.assertIn('<metadata type="name">tet</metadata>', xml)
        self.assertIn('<object id="1">', xml)
        self.assertIn("<coordinates>", xml)
        self.assertIn("<triangle>", xml)
        self.assertIn("<v1>0</v1>", xml)

    def test_deterministic(self):
        objs = from_polyhedra([unit_cube()])
        self.assertEqual(dumps(objs), dumps(objs))

    def test_bad_unit(self):
        with self.assertRaises(AmfError):
            dumps(from_polyhedra([tetrahedron()]), unit="furlong")

    def test_empty_document(self):
        with self.assertRaises(AmfError):
            dumps([])

    def test_units_survive(self):
        xml = dumps(from_polyhedra([tetrahedron()]), unit="inch")
        _, unit, _ = loads(xml)
        self.assertEqual(unit, "inch")


class TestRoundTrip(unittest.TestCase):
    def test_indexed_mesh_round_trip(self):
        cube = unit_cube(2.0)
        objs = from_polyhedra([cube])
        back, unit, meta = loads(dumps(objs, metadata={"name": "c"}))
        self.assertEqual(unit, "millimeter")
        self.assertEqual(meta, {"name": "c"})
        self.assertEqual(back, objs)
        # 8 shared vertices, not 36 like an STL would need
        self.assertEqual(len(back[0].vertices), 8)
        self.assertEqual(len(back[0].volumes[0]), 12)

    def test_geometry_preserved(self):
        cube = unit_cube(3.0)
        back, _, _ = loads(dumps(from_polyhedra([cube])))
        p = to_polyhedra(back)[0]
        self.assertAlmostEqual(p.volume(), 27.0)
        self.assertAlmostEqual(p.surface_area(), 54.0)
        self.assertEqual(verify(p), [])

    def test_document_round_trip_is_byte_stable(self):
        xml = dumps(from_polyhedra([tetrahedron()]))
        objs, unit, meta = loads(xml)
        self.assertEqual(dumps(objs, unit=unit, metadata=meta), xml)

    def test_object_metadata(self):
        obj = from_polyhedra([tetrahedron()])[0]
        obj.metadata["name"] = "lump"
        back, _, _ = loads(dumps([obj]))
        self.assertEqual(back[0].metadata, {"name": "lump"})

    def test_plain_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.amf")
            write_amf(path, from_polyhedra([unit_cube()]))
            objs, unit, _ = read_amf(path)
            self.assertEqual(unit, "millimeter")
            self.assertAlmostEqual(to_polyhedra(objs)[0].volume(), 1.0)

    def test_zip_file_round_trip_and_reproducible(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.amf")
            a = write_amf(path, from_polyhedra([unit_cube(2.0)]), compress=True)
            self.assertTrue(a.startswith(b"PK\x03\x04"))
            b = write_amf(path, from_polyhedra([unit_cube(2.0)]), compress=True)
            self.assertEqual(a, b)
            objs, _, _ = read_amf(path)
            self.assertAlmostEqual(to_polyhedra(objs)[0].volume(), 8.0)

    def test_zip_is_smaller_than_plain(self):
        objs = from_polyhedra([unit_cube()])
        with tempfile.TemporaryDirectory() as d:
            plain = write_amf(os.path.join(d, "a.amf"), objs)
            zipped = write_amf(os.path.join(d, "b.amf"), objs, compress=True)
        self.assertLess(len(zipped), len(plain))


class TestLumps(unittest.TestCase):
    def test_multi_volume_object(self):
        polys = [unit_cube(), _shifted(unit_cube(), 5.0)]
        objs = from_polyhedra(polys, one_object=True)
        self.assertEqual(len(objs), 1)
        self.assertEqual(len(objs[0].volumes), 2)
        self.assertEqual(len(objs[0].vertices), 16)
        back, _, _ = loads(dumps(objs))
        lumps = to_polyhedra(back)
        self.assertEqual(len(lumps), 2)
        for lump in lumps:
            self.assertAlmostEqual(lump.volume(), 1.0)
            self.assertEqual(verify(lump), [])

    def test_merge_lumps(self):
        polys = [unit_cube(), _shifted(unit_cube(), 5.0)]
        merged = merge_lumps(polys)
        self.assertEqual(merged.nvert(), 16)
        self.assertEqual(merged.nface(), 12)
        self.assertAlmostEqual(merged.volume(), 2.0)
        lo, hi = merged.bounds()
        self.assertEqual(lo, (0.0, 0.0, 0.0))
        self.assertEqual(hi, (6.0, 1.0, 1.0))

    def test_merge_lumps_needs_input(self):
        with self.assertRaises(AmfError):
            merge_lumps([])

    def test_separate_objects_by_default(self):
        objs = from_polyhedra([unit_cube(), tetrahedron()])
        self.assertEqual([o.id for o in objs], [1, 2])
        self.assertEqual(len(to_polyhedra(objs)), 2)


class TestReaderErrors(unittest.TestCase):
    def test_bad_root(self):
        with self.assertRaises(AmfError):
            loads("<foo/>")

    def test_bad_xml(self):
        with self.assertRaises(AmfError):
            loads("<amf>")

    def test_illegal_unit(self):
        with self.assertRaises(AmfError):
            loads('<amf unit="parsec"><object id="1"/></amf>')

    def test_no_object(self):
        with self.assertRaises(AmfError):
            loads('<amf unit="millimeter"></amf>')

    def test_missing_mesh(self):
        with self.assertRaises(AmfError):
            loads('<amf unit="millimeter"><object id="1"/></amf>')

    def test_index_out_of_range(self):
        doc = (
            '<amf unit="millimeter"><object id="1"><mesh><vertices>'
            "<vertex><coordinates><x>0</x><y>0</y><z>0</z></coordinates></vertex>"
            "</vertices><volume><triangle><v1>0</v1><v2>1</v2><v3>2</v3></triangle>"
            "</volume></mesh></object></amf>"
        )
        with self.assertRaises(AmfError):
            loads(doc)

    def test_bad_coordinate(self):
        doc = (
            '<amf unit="millimeter"><object id="1"><mesh><vertices>'
            "<vertex><coordinates><x>zero</x><y>0</y><z>0</z></coordinates></vertex>"
            "</vertices><volume/></mesh></object></amf>"
        )
        with self.assertRaises(AmfError):
            loads(doc)

    def test_bad_zip(self):
        with self.assertRaises(AmfError):
            loads(b"PK\x03\x04garbage")


if __name__ == "__main__":
    unittest.main()
