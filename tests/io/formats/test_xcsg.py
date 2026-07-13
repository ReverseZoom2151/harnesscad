"""Tests for formats.angelcad_xcsg_xml."""

import os
import tempfile
import unittest

from harnesscad.io.formats.xcsg import (
    XcsgError,
    dumps,
    flatten_transforms,
    loads,
    read_xcsg,
    write_xcsg,
)
from harnesscad.domain.programs.ast.typed_csg import (
    Node,
    TMatrix,
    check,
    circle,
    cuboid,
    cylinder,
    difference3d,
    linear_extrude,
    polygon,
    polyhedron,
    rotate_z,
    sphere,
    sweep,
    transform,
    translate,
    union3d,
)


def _model():
    return difference3d(
        transform(translate(5, 0, 0), cuboid(10, 10, 10, center=True)),
        sphere(6),
    )


class TestWriter(unittest.TestCase):
    def test_root_and_structure(self):
        xml = dumps(_model(), secant_tolerance=0.01, model_name="model")
        self.assertTrue(xml.startswith('<xcsg version="1.0" secant_tolerance="0.01">'))
        self.assertIn('<metadata name="model" />', xml)
        self.assertIn("<difference3d>", xml)
        self.assertIn('<cuboid center="true" dx="10" dy="10" dz="10">', xml)
        self.assertIn('<trow c0="1" c1="0" c2="0" c3="5" />', xml)
        self.assertIn('<sphere r="6" />', xml)

    def test_identity_transform_is_omitted(self):
        xml = dumps(transform(translate(0, 0, 0), sphere(1)))
        self.assertNotIn("tmatrix", xml)

    def test_deterministic_bytes(self):
        self.assertEqual(dumps(_model()), dumps(_model()))

    def test_polyhedron_serialisation(self):
        node = polyhedron(
            [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
            [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
        )
        xml = dumps(node)
        self.assertIn('<vertex x="0" y="0" z="0" />', xml)
        self.assertIn('<fv index="2" />', xml)
        self.assertEqual(loads(xml), node)

    def test_polygon_is_2d(self):
        node = polygon([(0, 0), (2, 0), (2, 1)])
        xml = dumps(node)
        self.assertNotIn(' z=', xml)
        self.assertEqual(loads(xml), node)

    def test_sweep_path(self):
        node = sweep(circle(1), [(0, 0, 0), (0, 0, 5), (1, 0, 9)])
        xml = dumps(node)
        self.assertIn("<spline_path>", xml)
        self.assertIn('<cpoint x="1" y="0" z="9" />', xml)
        self.assertEqual(loads(xml), node)

    def test_unknown_op_rejected(self):
        with self.assertRaises(XcsgError):
            dumps(Node("frobnicate"))


class TestTransformFlattening(unittest.TestCase):
    def test_nested_transforms_collapse_to_one_matrix(self):
        node = transform(translate(10, 0, 0), transform(rotate_z(deg=90), cylinder(4, 1)))
        xml = dumps(node)
        self.assertEqual(xml.count("<tmatrix>"), 1)
        back = loads(xml)
        self.assertEqual(back.op, "transform")
        m = back.params["matrix"]
        # outer * inner: rotate first, then translate
        p = m.apply_pos((1, 0, 0))
        self.assertAlmostEqual(p[0], 10.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_flatten_transforms_helper(self):
        node = transform(translate(1, 0, 0), transform(translate(2, 0, 0), sphere(1)))
        flat = flatten_transforms(node)
        self.assertEqual(flat.op, "transform")
        self.assertEqual(flat.children[0].op, "sphere")
        self.assertEqual(flat.params["matrix"].origin(), (3.0, 0.0, 0.0))

    def test_flatten_is_idempotent(self):
        flat = flatten_transforms(_model())
        self.assertEqual(flatten_transforms(flat), flat)

    def test_transform_inside_boolean_stays_local(self):
        node = union3d(transform(translate(0, 0, 3), sphere(1)), sphere(2))
        xml = dumps(node)
        self.assertEqual(xml.count("<tmatrix>"), 1)
        self.assertEqual(loads(xml), node)


class TestRoundTrip(unittest.TestCase):
    def test_document_round_trip(self):
        xml = dumps(_model(), secant_tolerance=0.01, model_name="m")
        self.assertEqual(dumps(loads(xml), secant_tolerance=0.01, model_name="m"), xml)

    def test_tree_round_trip_preserves_types(self):
        node = union3d(linear_extrude(circle(2), dz=4), sphere(1))
        back = loads(dumps(node))
        self.assertEqual(back, node)
        self.assertEqual(check(back), [])

    def test_matrix_round_trip(self):
        m = TMatrix([[1, 0, 0, 1.5], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        back = loads(dumps(Node("transform", {"matrix": m}, (sphere(1),))))
        self.assertEqual(back.params["matrix"].rows, m.rows)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.xcsg")
            write_xcsg(path, _model())
            self.assertEqual(read_xcsg(path), _model())


class TestReaderErrors(unittest.TestCase):
    def test_bad_root(self):
        with self.assertRaises(XcsgError):
            loads("<foo/>")

    def test_two_root_shapes(self):
        with self.assertRaises(XcsgError):
            loads('<xcsg version="1.0"><sphere r="1"/><sphere r="2"/></xcsg>')

    def test_bad_xml(self):
        with self.assertRaises(XcsgError):
            loads("<xcsg>")

    def test_bad_number(self):
        with self.assertRaises(XcsgError):
            loads('<xcsg version="1.0"><sphere r="big"/></xcsg>')

    def test_bad_bool(self):
        with self.assertRaises(XcsgError):
            loads('<xcsg version="1.0"><cube size="1" center="yes"/></xcsg>')

    def test_short_tmatrix(self):
        with self.assertRaises(XcsgError):
            loads(
                '<xcsg version="1.0"><sphere r="1"><tmatrix>'
                '<trow c0="1" c1="0" c2="0" c3="0"/></tmatrix></sphere></xcsg>'
            )


if __name__ == "__main__":
    unittest.main()
