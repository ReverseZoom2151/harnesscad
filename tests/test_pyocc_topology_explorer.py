"""Tests for geometry.pyocc_topology_explorer."""

import unittest

from harnesscad.domain.geometry.topology.pyocc_topology_explorer import (
    FORWARD,
    REVERSED,
    Shape,
    ancestors_of,
    count,
    make_box,
    make_edge,
    map_shapes_and_ancestors,
    sub_shapes,
    topology_summary,
)


class ShapeTests(unittest.TestCase):
    def test_rejects_bad_type(self):
        with self.assertRaises(ValueError):
            Shape("NOTATYPE", "x")

    def test_rejects_bad_orientation(self):
        with self.assertRaises(ValueError):
            Shape("EDGE", "e", "SIDEWAYS")

    def test_is_same_ignores_orientation(self):
        a = Shape("EDGE", "e0", FORWARD)
        b = Shape("EDGE", "e0", REVERSED)
        self.assertTrue(a.is_same(b))
        self.assertEqual(a.reversed().orientation, REVERSED)
        self.assertTrue(a.reversed().is_same(a))

    def test_is_same_distinguishes_geometry_and_type(self):
        self.assertFalse(Shape("EDGE", "e0").is_same(Shape("EDGE", "e1")))
        self.assertFalse(Shape("EDGE", "e0").is_same(Shape("WIRE", "e0")))


class CubeTopologyTests(unittest.TestCase):
    def setUp(self):
        self.box = make_box()

    def test_unique_counts(self):
        self.assertEqual(count(self.box, "SOLID"), 1)
        self.assertEqual(count(self.box, "SHELL"), 1)
        self.assertEqual(count(self.box, "FACE"), 6)
        self.assertEqual(count(self.box, "WIRE"), 6)
        self.assertEqual(count(self.box, "EDGE"), 12)
        self.assertEqual(count(self.box, "VERTEX"), 8)

    def test_orientation_dedup_collapses_occurrences(self):
        # 6 faces x 4 edges = 24 oriented edge occurrences collapse to 12 unique.
        self.assertEqual(len(sub_shapes(self.box, "EDGE", unique=False)), 24)
        self.assertEqual(len(sub_shapes(self.box, "EDGE", unique=True)), 12)
        # 24 edge occurrences x 2 vertices each = 48 occurrences -> 8 unique.
        self.assertEqual(len(sub_shapes(self.box, "VERTEX", unique=False)), 48)
        self.assertEqual(len(sub_shapes(self.box, "VERTEX", unique=True)), 8)

    def test_topology_summary(self):
        self.assertEqual(
            topology_summary(self.box),
            {
                "compound": 0,
                "compsolid": 0,
                "solid": 1,
                "shell": 1,
                "face": 6,
                "wire": 6,
                "edge": 12,
                "vertex": 8,
            },
        )

    def test_root_matching_type_included(self):
        # Exploring a shape for its own type returns it (TopExp_Explorer behaviour).
        self.assertEqual(count(self.box, "SOLID"), 1)

    def test_euler_characteristic(self):
        # V - E + F = 2 for a genus-0 closed solid.
        v = count(self.box, "VERTEX")
        e = count(self.box, "EDGE")
        f = count(self.box, "FACE")
        self.assertEqual(v - e + f, 2)


class AncestorMapTests(unittest.TestCase):
    def setUp(self):
        self.box = make_box()

    def test_each_edge_bounds_two_faces(self):
        amap = map_shapes_and_ancestors(self.box, "EDGE", "FACE")
        self.assertEqual(len(amap), 12)
        for edge_tshape, faces in amap.items():
            self.assertEqual(len(faces), 2, f"edge {edge_tshape} should bound 2 faces")

    def test_each_vertex_meets_three_edges(self):
        amap = map_shapes_and_ancestors(self.box, "VERTEX", "EDGE")
        self.assertEqual(len(amap), 8)
        for vtx, edges in amap.items():
            self.assertEqual(len(edges), 3, f"vertex {vtx} should meet 3 edges")

    def test_ancestors_of_specific_edge(self):
        # e0 is shared by the bottom face (f0) and the front face (f2).
        e0 = make_edge("e0", Shape("VERTEX", "v0"), Shape("VERTEX", "v1"))
        faces = ancestors_of(self.box, e0, "FACE")
        tshapes = sorted(f.tshape for f in faces)
        self.assertEqual(tshapes, ["f0", "f2"])

    def test_deterministic_repeat(self):
        a = map_shapes_and_ancestors(self.box, "EDGE", "FACE")
        b = map_shapes_and_ancestors(make_box(), "EDGE", "FACE")
        self.assertEqual(list(a.keys()), list(b.keys()))
        self.assertEqual(
            [[f.tshape for f in v] for v in a.values()],
            [[f.tshape for f in v] for v in b.values()],
        )

    def test_ancestor_must_be_outer_type(self):
        with self.assertRaises(ValueError):
            map_shapes_and_ancestors(self.box, "FACE", "EDGE")
        with self.assertRaises(ValueError):
            map_shapes_and_ancestors(self.box, "EDGE", "EDGE")

    def test_missing_entity_has_no_ancestors(self):
        stray = Shape("VERTEX", "nope")
        self.assertEqual(ancestors_of(self.box, stray, "EDGE"), [])


if __name__ == "__main__":
    unittest.main()
