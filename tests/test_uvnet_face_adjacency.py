"""Tests for the UV-Net face-adjacency graph with UV-grid features."""

import math
import unittest

from harnesscad.domain.geometry.parametric import uvnet_u_grid as ug
from harnesscad.domain.geometry.parametric import uvnet_uv_grid as uvg
from harnesscad.domain.reconstruction.brep import uvnet_face_adjacency as fa


def unit_cube_faces():
    """The 6 planar faces of the axis-aligned unit cube [0,1]^3."""
    specs = [
        ((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),   # bottom
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),    # top
        ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),   # -x
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),    # +x
        ((0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),   # -y
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)),    # +y
    ]
    return [fa.FaceEntry(surface=uvg.Plane(origin=o, axis=a,
                                           u_range=(0.0, 1.0),
                                           v_range=(0.0, 1.0)))
            for o, a in specs]


def unit_cube_edges():
    """The 12 edges of the cube, each labelled with the two faces it joins."""
    b, t, mx, px, my, py = 0, 1, 2, 3, 4, 5
    e = []
    x = (1.0, 0.0, 0.0)
    y = (0.0, 1.0, 0.0)
    z = (0.0, 0.0, 1.0)
    # bottom square
    e.append(fa.EdgeEntry(ug.Line((0, 0, 0), x), (b, my)))
    e.append(fa.EdgeEntry(ug.Line((0, 1, 0), x), (b, py)))
    e.append(fa.EdgeEntry(ug.Line((0, 0, 0), y), (b, mx)))
    e.append(fa.EdgeEntry(ug.Line((1, 0, 0), y), (b, px)))
    # top square
    e.append(fa.EdgeEntry(ug.Line((0, 0, 1), x), (t, my)))
    e.append(fa.EdgeEntry(ug.Line((0, 1, 1), x), (t, py)))
    e.append(fa.EdgeEntry(ug.Line((0, 0, 1), y), (t, mx)))
    e.append(fa.EdgeEntry(ug.Line((1, 0, 1), y), (t, px)))
    # vertical edges
    e.append(fa.EdgeEntry(ug.Line((0, 0, 0), z), (mx, my)))
    e.append(fa.EdgeEntry(ug.Line((1, 0, 0), z), (px, my)))
    e.append(fa.EdgeEntry(ug.Line((1, 1, 0), z), (px, py)))
    e.append(fa.EdgeEntry(ug.Line((0, 1, 0), z), (mx, py)))
    return e


class CubeGraphTest(unittest.TestCase):
    def setUp(self):
        self.graph = fa.build_face_adjacency(unit_cube_faces(),
                                             unit_cube_edges(),
                                             curv_num_u=6,
                                             surf_num_u=5, surf_num_v=5)

    def test_shapes(self):
        g = self.graph
        self.assertEqual(g.num_nodes, 6)
        self.assertEqual(g.num_edges, 12)
        self.assertEqual(uvg.grid_shape(g.node_features[0]), (5, 5, 7))
        self.assertEqual(len(g.edge_features[0]), 6)
        self.assertEqual(len(g.edge_features[0][0]), 6)
        self.assertEqual(g.skipped_edges, [])

    def test_topology(self):
        g = self.graph
        self.assertEqual(fa.degrees(g), [4, 4, 4, 4, 4, 4])
        self.assertTrue(fa.is_connected(g))
        self.assertEqual(fa.connected_components(g), [[0, 1, 2, 3, 4, 5]])
        mat = fa.adjacency_matrix(g)
        # bottom (0) and top (1) are not adjacent; bottom and -x (2) are
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[0][2], 1)
        self.assertEqual(mat[2][0], 1)

    def test_summary(self):
        s = fa.graph_summary(self.graph)
        self.assertEqual(s["num_nodes"], 6)
        self.assertEqual(s["num_edges"], 12)
        self.assertEqual(s["node_feature_shape"], (5, 5, 7))
        self.assertEqual(s["edge_feature_shape"], (6, 6))
        self.assertTrue(s["connected"])
        self.assertEqual(s["skipped_edges"], 0)

    def test_bidirectional_doubles_edges_and_flips_tangents(self):
        bi = fa.to_bidirectional(self.graph)
        self.assertEqual(bi.num_edges, 24)
        self.assertEqual(bi.src[0], self.graph.src[0])
        self.assertEqual(bi.src[1], self.graph.dst[0])
        fwd = bi.edge_features[0]
        rev = bi.edge_features[1]
        self.assertAlmostEqual(rev[0][3], -fwd[-1][3], places=12)
        self.assertAlmostEqual(rev[0][0], fwd[-1][0], places=12)

    def test_node_points_lie_on_the_cube(self):
        pts = fa.all_masked_points(self.graph)
        self.assertEqual(len(pts), 6 * 25)
        for p in pts:
            self.assertTrue(any(abs(c) < 1e-9 or abs(c - 1.0) < 1e-9
                                for c in p))

    def test_determinism(self):
        again = fa.build_face_adjacency(unit_cube_faces(), unit_cube_edges(),
                                        curv_num_u=6, surf_num_u=5,
                                        surf_num_v=5)
        self.assertEqual(again.node_features, self.graph.node_features)
        self.assertEqual(again.edge_features, self.graph.edge_features)
        self.assertEqual(again.src, self.graph.src)


class CylinderGraphTest(unittest.TestCase):
    def test_capped_cylinder_with_seam_self_loop(self):
        faces = [
            fa.FaceEntry(surface=uvg.Cylinder(origin=(0, 0, 0), axis=(0, 0, 1),
                                              radius=1.0, v_range=(0.0, 2.0)),
                         name="side"),
            fa.FaceEntry(surface=uvg.Plane(origin=(0, 0, 0), axis=(0, 0, -1)),
                         name="bottom"),
            fa.FaceEntry(surface=uvg.Plane(origin=(0, 0, 2), axis=(0, 0, 1)),
                         name="top"),
        ]
        edges = [
            fa.EdgeEntry(ug.Circle((0, 0, 0), (0, 0, 1), 1.0), (0, 1)),
            fa.EdgeEntry(ug.Circle((0, 0, 2), (0, 0, 1), 1.0), (0, 2)),
            fa.EdgeEntry(ug.Line((1, 0, 0), (0, 0, 1), u_range=(0.0, 2.0)),
                         (0, 0), name="seam"),
        ]
        g = fa.build_face_adjacency(faces, edges, curv_num_u=8,
                                    surf_num_u=8, surf_num_v=4)
        self.assertEqual(g.num_edges, 3)
        self.assertEqual(fa.degrees(g), [3, 1, 1])
        self.assertTrue(fa.is_connected(g))
        self.assertEqual(fa.adjacency_matrix(g)[0][0], 1)

    def test_degenerate_edge_is_skipped(self):
        faces = [
            fa.FaceEntry(surface=uvg.Cone(origin=(0, 0, 0), axis=(0, 0, 1),
                                          radius=0.0,
                                          half_angle=math.radians(30),
                                          v_range=(0.0, 2.0))),
            fa.FaceEntry(surface=uvg.Plane(origin=(0, 0, 2), axis=(0, 0, 1))),
        ]
        edges = [
            fa.EdgeEntry(ug.Circle((0, 0, 2), (0, 0, 1),
                                   2.0 * math.tan(math.radians(30))), (0, 1)),
            # apex: a zero-length "edge" -> UV-Net drops it (no curve)
            fa.EdgeEntry(ug.Polyline([(0, 0, 0), (0, 0, 0)]), (0, 0),
                         name="apex"),
            # dangling reference to a non-existent face
            fa.EdgeEntry(ug.Line((0, 0, 0), (1, 0, 0)), (0, 7)),
        ]
        g = fa.build_face_adjacency(faces, edges, curv_num_u=5,
                                    surf_num_u=4, surf_num_v=4)
        self.assertEqual(g.num_edges, 1)
        self.assertEqual(g.skipped_edges, [1, 2])
        self.assertEqual(fa.graph_summary(g)["skipped_edges"], 2)


class DisconnectedAndValidationTest(unittest.TestCase):
    def test_two_components(self):
        faces = unit_cube_faces()[:4]
        edges = [fa.EdgeEntry(ug.Line((0, 0, 0), (1, 0, 0)), (0, 1)),
                 fa.EdgeEntry(ug.Line((0, 0, 0), (0, 1, 0)), (2, 3))]
        g = fa.build_face_adjacency(faces, edges, curv_num_u=3,
                                    surf_num_u=3, surf_num_v=3)
        self.assertFalse(fa.is_connected(g))
        self.assertEqual(fa.connected_components(g), [[0, 1], [2, 3]])

    def test_grid_size_validation(self):
        with self.assertRaises(ValueError):
            fa.build_face_adjacency(unit_cube_faces(), [], surf_num_u=1)


if __name__ == "__main__":
    unittest.main()
