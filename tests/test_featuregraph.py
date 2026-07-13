"""Tests for the semantic feature / adjacency graph (featuregraph.py).

Everything here runs on the dependency-free StubBackend and the pure op-DAG path
(no network, no OCCT), so the feature-level graph is exercised deterministically.
The Hole op is imported defensively so the suite still runs if the extended op
vocabulary is ever trimmed.
"""

import unittest

from harnesscad.io.backends.stub import StubBackend
from harnesscad.core.state.opdag import OpDAG
from harnesscad.core.cisp.ops import NewSketch, AddRectangle, Extrude, Fillet

try:
    from harnesscad.core.cisp.ops import Hole
    HAVE_HOLE = True
except Exception:  # noqa: BLE001
    Hole = None
    HAVE_HOLE = False

from harnesscad.eval.quality.graph.feature_graph import build_feature_graph, FeatureGraph, FeatureNode, FeatureEdge


def _plate_ops(n_holes=0, fillet=False):
    """A rectangular plate: sketch + rectangle + extrude, optional holes/fillet."""
    ops = [
        NewSketch(plane="XY"),
        AddRectangle(sketch="sk1", x=0.0, y=0.0, w=60.0, h=40.0),
        Extrude(sketch="sk1", distance=8.0),
    ]
    if HAVE_HOLE:
        for i in range(n_holes):
            ops.append(Hole(face_or_sketch="", x=5.0 + i, y=5.0,
                            diameter=5.0, through=True))
    if fillet:
        ops.append(Fillet(edges=(), radius=2.0))
    return ops


def _apply(backend, ops):
    for op in ops:
        res = backend.apply(op)
        assert res.ok, (op, res.diagnostics and res.diagnostics[0].message)
    return backend


def _opdag(ops):
    dag = OpDAG()
    for op in ops:
        dag.append(op)
    return dag


class TestBuildFromOpDAG(unittest.TestCase):
    def test_basic_nodes_and_profile_edge(self):
        g = build_feature_graph(_opdag(_plate_ops()))
        self.assertIsInstance(g, FeatureGraph)
        sketches = g.find("sketch")
        extrudes = g.find("extrude")
        self.assertEqual(len(sketches), 1)
        self.assertEqual(len(extrudes), 1)
        # the sketch is the profile of the extrude
        rels = {(e.source, e.target, e.relation) for e in g.edges}
        self.assertIn(("sk1", "f1", "profile-of"), rels)
        # sketch node carries the rectangle primitive
        self.assertIn("rectangle", sketches[0].params["primitive_types"])

    def test_node_ids_mirror_backend_scheme(self):
        g = build_feature_graph(_opdag(_plate_ops()))
        self.assertIsNotNone(g.get("sk1"))
        self.assertIsNotNone(g.get("f1"))

    @unittest.skipUnless(HAVE_HOLE, "Hole op not available")
    def test_holes_become_nodes_with_relation(self):
        g = build_feature_graph(_opdag(_plate_ops(n_holes=4)))
        holes = g.find("hole")
        self.assertEqual(len(holes), 4)
        for h in holes:
            self.assertEqual(h.params["diameter"], 5.0)
            self.assertTrue(h.params["through"])
        # every hole relates to the extruded body as 'hole-through-wall'
        wall_rels = [e for e in g.edges if e.relation == "hole-through-wall"]
        self.assertEqual(len(wall_rels), 4)
        for e in wall_rels:
            self.assertEqual(e.target, "f1")  # the extrude body

    @unittest.skipUnless(HAVE_HOLE, "Hole op not available")
    def test_fillet_on_feature_relation(self):
        g = build_feature_graph(_opdag(_plate_ops(n_holes=1, fillet=True)))
        fillets = g.find("fillet")
        self.assertEqual(len(fillets), 1)
        on = [e for e in g.edges if e.relation == "fillet-on"]
        self.assertEqual(len(on), 1)
        self.assertEqual(on[0].target, "f1")  # fillet is on the extrude body

    def test_neighbors_and_find_features(self):
        g = build_feature_graph(_opdag(_plate_ops(fillet=True)))
        # the extrude body neighbours its sketch and its fillet
        neigh_types = {n.type for n in g.neighbors("f1")}
        self.assertIn("sketch", neigh_types)
        self.assertIn("fillet", neigh_types)
        # find_features excludes sketch nodes
        self.assertTrue(all(n.type != "sketch" for n in g.find_features()))

    def test_to_dict_roundtrip_shape(self):
        g = build_feature_graph(_opdag(_plate_ops()))
        d = g.to_dict()
        self.assertIn("nodes", d)
        self.assertIn("edges", d)
        self.assertTrue(all({"id", "type", "params"} <= set(n) for n in d["nodes"]))
        self.assertTrue(all({"source", "target", "relation"} <= set(e) for e in d["edges"]))


class TestBuildFromBackend(unittest.TestCase):
    def test_backend_and_opdag_agree(self):
        ops = _plate_ops(n_holes=2, fillet=True)
        backend = _apply(StubBackend(), ops)
        g_backend = build_feature_graph(backend)
        g_opdag = build_feature_graph(_opdag(ops))
        # same node ids and types (B-rep enrichment is off for the stub)
        self.assertEqual(
            sorted((n.id, n.type) for n in g_backend.nodes),
            sorted((n.id, n.type) for n in g_opdag.nodes),
        )

    def test_feature_count_matches_summary(self):
        ops = _plate_ops(n_holes=3, fillet=True)
        backend = _apply(StubBackend(), ops)
        g = build_feature_graph(backend)
        self.assertEqual(len(g.find_features()),
                         backend.query("summary")["feature_count"])


class TestDeterminism(unittest.TestCase):
    def test_identical_ops_yield_identical_graph(self):
        ops = _plate_ops(n_holes=2, fillet=True)
        d1 = build_feature_graph(_opdag(ops)).to_dict()
        d2 = build_feature_graph(_opdag(ops)).to_dict()
        self.assertEqual(d1, d2)


class TestDataclasses(unittest.TestCase):
    def test_node_and_edge_to_dict(self):
        n = FeatureNode("f1", "extrude", {"distance": 8.0})
        e = FeatureEdge("sk1", "f1", "profile-of")
        self.assertEqual(n.to_dict()["type"], "extrude")
        self.assertEqual(e.to_dict()["relation"], "profile-of")


if __name__ == "__main__":
    unittest.main()
