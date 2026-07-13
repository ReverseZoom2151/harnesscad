import math
import unittest

from harnesscad.domain.geometry.topology.synthetic_brep import (
    BoundingBox,
    build_topology,
    direction_tags,
    synthetic_box_edges,
    synthetic_box_faces,
    synthetic_cylinder_faces,
    topology_summary,
)

BOX = BoundingBox(0.0, 0.0, 0.0, 2.0, 4.0, 6.0)
CYL = BoundingBox(-1.0, -1.0, 0.0, 1.0, 1.0, 5.0)


class TestDirectionTags(unittest.TestCase):
    def test_axis_tags(self):
        self.assertEqual(direction_tags((0.0, 0.0, 1.0)), ["top", "+Z"])
        self.assertEqual(direction_tags((-1.0, 0.0, 0.0)), ["left", "-X"])

    def test_off_axis_has_no_tags(self):
        self.assertEqual(direction_tags((1.0, 1.0, 0.0)), [])

    def test_undirected_matches_opposite(self):
        self.assertEqual(direction_tags((0.0, 0.0, -1.0), undirected=True), ["top", "+Z"])

    def test_none_and_zero(self):
        self.assertEqual(direction_tags(None), [])
        self.assertEqual(direction_tags((0.0, 0.0, 0.0)), [])


class TestBoxTopology(unittest.TestCase):
    def test_six_faces_with_areas(self):
        faces = synthetic_box_faces("box-1", BOX)
        self.assertEqual(len(faces), 6)
        top = faces[0]
        self.assertEqual(top.id, "box-1:face:0")
        self.assertIn("top", top.tags)
        self.assertAlmostEqual(top.area, 2.0 * 4.0)
        self.assertEqual(top.centroid, (1.0, 2.0, 6.0))
        self.assertEqual(faces[4].tags, ("right", "+X"))
        self.assertTrue(all(f.surface == "planar" for f in faces))

    def test_twelve_edges_with_lengths(self):
        edges = synthetic_box_edges("box-1", BOX)
        self.assertEqual(len(edges), 12)
        total = sum(e.length for e in edges)
        # 4 edges of each box dimension.
        self.assertAlmostEqual(total, 4 * (2.0 + 4.0 + 6.0))
        vertical = [e for e in edges if "vertical" in e.tags]
        self.assertEqual(len(vertical), 4)
        self.assertTrue(all(abs(e.length - 6.0) < 1e-9 for e in vertical))

    def test_edge_midpoints_within_bbox(self):
        for edge in synthetic_box_edges("b", BOX):
            self.assertTrue(0.0 <= edge.centroid[0] <= 2.0)
            self.assertTrue(0.0 <= edge.centroid[2] <= 6.0)


class TestCylinderAndSphere(unittest.TestCase):
    def test_cylinder_faces(self):
        faces = synthetic_cylinder_faces("cyl-1", CYL)
        self.assertEqual(len(faces), 3)
        self.assertAlmostEqual(faces[0].area, math.pi)
        self.assertEqual(faces[2].surface, "cylindrical")
        self.assertIsNone(faces[2].normal)
        self.assertAlmostEqual(faces[2].area, 2 * math.pi * 1.0 * 5.0)

    def test_cylinder_edges(self):
        topology = build_topology("cyl-1", "cylinder", CYL)
        self.assertEqual(len(topology.edges), 3)
        seam = topology.edges[2]
        self.assertIn("seam", seam.tags)
        self.assertAlmostEqual(seam.length, 5.0)

    def test_sphere(self):
        topology = build_topology("s", "sphere", BoundingBox(-2, -2, -2, 2, 2, 2))
        self.assertEqual(len(topology.faces), 1)
        self.assertEqual(topology.edges, [])
        self.assertAlmostEqual(topology.faces[0].area, 4 * math.pi * 4.0)


class TestTopologyMap(unittest.TestCase):
    def test_unknown_kind_falls_back_to_box(self):
        topology = build_topology("x", "prism", BOX)
        self.assertEqual(len(topology.faces), 6)

    def test_by_tag_and_ids(self):
        topology = build_topology("box-1", "box", BOX)
        self.assertEqual([r.id for r in topology.by_tag("bottom")][0], "box-1:face:1")
        self.assertEqual(len(topology.ids()), 18)

    def test_summary_and_determinism(self):
        first = topology_summary(build_topology("b", "box", BOX))
        second = topology_summary(build_topology("b", "box", BOX))
        self.assertEqual(first, second)
        self.assertEqual(first["face_count"], 6.0)
        self.assertAlmostEqual(
            first["total_area"], 2 * (2 * 4 + 2 * 6 + 4 * 6)
        )


if __name__ == "__main__":
    unittest.main()
