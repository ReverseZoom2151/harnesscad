import unittest

from reconstruction.cadparser_brep_graph import (
    BRep, EdgeDef, FaceDef, adjacency_matrix, build_graph, coedge_feature,
    edge_feature, face_feature, node_features, CURVE_TYPES, SURFACE_TYPES,
)


def square_prism_brep():
    # A minimal watertight-ish sample: two faces sharing four edges (top + bottom
    # of a square), enough to exercise mates/next/prev.
    edges = tuple(EdgeDef(f"e{i}", "line", 1.0) for i in range(4))
    bottom = FaceDef("f_bot", "plane", 1.0,
                     loops=((("e0", True), ("e1", True), ("e2", True), ("e3", True)),))
    top = FaceDef("f_top", "plane", 1.0,
                  loops=((("e0", False), ("e1", False), ("e2", False), ("e3", False)),))
    return BRep(faces=(bottom, top), edges=edges)


class TestBRepGraph(unittest.TestCase):
    def setUp(self):
        self.graph = build_graph(square_prism_brep())

    def test_node_counts(self):
        self.assertEqual(len(self.graph.faces), 2)
        self.assertEqual(len(self.graph.edges), 4)
        self.assertEqual(len(self.graph.coedges), 8)
        self.assertEqual(self.graph.n_nodes, 14)

    def test_face_coedge_relation(self):
        # every coedge belongs to exactly one face
        self.assertEqual(len(self.graph.relations["face_coedge"]), 8)

    def test_mates_pair_opposite_coedges(self):
        # each edge has two coedges of opposite orientation -> 2 directed mate links
        self.assertEqual(len(self.graph.relations["coedge_mate"]), 8)

    def test_next_prev_form_cycles(self):
        self.assertEqual(len(self.graph.relations["coedge_next"]), 8)
        self.assertEqual(len(self.graph.relations["coedge_prev"]), 8)

    def test_adjacency_symmetric(self):
        m = adjacency_matrix(self.graph, symmetric=True)
        n = self.graph.n_nodes
        self.assertEqual(len(m), n)
        for i in range(n):
            for j in range(n):
                self.assertEqual(m[i][j], m[j][i])

    def test_unknown_edge_reference_raises(self):
        bad = BRep(faces=(FaceDef("f", "plane", 1.0, loops=((("missing", True),),)),),
                   edges=())
        with self.assertRaises(ValueError):
            build_graph(bad)

    def test_features(self):
        ff = face_feature(FaceDef("f", "cylinder", 3.0))
        self.assertEqual(len(ff), len(SURFACE_TYPES) + 1)
        self.assertEqual(ff[-1], 3.0)
        ef = edge_feature(EdgeDef("e", "arc", 2.0))
        self.assertEqual(len(ef), len(CURVE_TYPES) + 1)
        self.assertEqual(sum(ef[:-1]), 1.0)
        self.assertEqual(coedge_feature(self.graph.coedges[0]),
                         (1.0 if self.graph.coedges[0].orientation else 0.0,))

    def test_node_features_bundle(self):
        feats = node_features(self.graph)
        self.assertEqual(len(feats["face"]), 2)
        self.assertEqual(len(feats["edge"]), 4)
        self.assertEqual(len(feats["coedge"]), 8)

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            face_feature(FaceDef("f", "hyperboloid"))
        with self.assertRaises(ValueError):
            edge_feature(EdgeDef("e", "helix"))


if __name__ == "__main__":
    unittest.main()
