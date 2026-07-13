"""Tests for reconstruction.sketchgraphs_sequence."""

import unittest

from harnesscad.domain.reconstruction.sketch import sketchgraphs_graph as sg
from harnesscad.domain.reconstruction.sequences import sketchgraphs_sequence as seq


def _triangle():
    g = sg.SketchGraph()
    g.add_primitive("l0", "line")
    g.add_primitive("l1", "line")
    g.add_primitive("l2", "line")
    g.add_constraint("coincident", ("l0", "l1"))
    g.add_constraint("coincident", ("l1", "l2"))
    g.add_constraint("coincident", ("l2", "l0"))
    return g


class InterleavedTests(unittest.TestCase):
    def test_edge_follows_last_member(self):
        g = _triangle()
        s = seq.interleaved_sequence(g)
        self.assertTrue(s.is_valid())
        toks = s.tokens()
        # First edge (l0,l1) must appear right after l1 is inserted.
        i_l1 = toks.index("N:line:l1")
        self.assertEqual(toks[i_l1 + 1], "E:coincident:l0,l1")

    def test_all_ops_present(self):
        g = _triangle()
        s = seq.interleaved_sequence(g)
        self.assertEqual(len(s.node_ops()), 3)
        self.assertEqual(len(s.edge_ops()), 3)
        self.assertEqual(len(s), 6)

    def test_loop_emitted_after_its_node(self):
        g = sg.SketchGraph()
        g.add_primitive("c0", "circle")
        g.add_constraint("radius", ("c0",), value=4.0)
        s = seq.interleaved_sequence(g)
        toks = s.tokens()
        self.assertEqual(toks[0], "N:circle:c0")
        self.assertEqual(toks[1], "E:radius:c0")


class ConstraintsLastTests(unittest.TestCase):
    def test_nodes_then_edges(self):
        g = _triangle()
        s = seq.constraints_last_sequence(g)
        self.assertTrue(s.is_valid())
        kinds = [type(o).__name__ for o in s.ops]
        # all NodeOp before any EdgeOp.
        self.assertEqual(kinds, ["NodeOp"] * 3 + ["EdgeOp"] * 3)


class ReplayTests(unittest.TestCase):
    def test_replay_roundtrip_interleaved(self):
        g = _triangle()
        s = seq.interleaved_sequence(g)
        g2 = seq.replay(s)
        self.assertEqual(g2.num_nodes, g.num_nodes)
        self.assertEqual(len(g2.constraint_edges()), len(g.constraint_edges()))

    def test_replay_roundtrip_constraints_last(self):
        g = _triangle()
        s = seq.constraints_last_sequence(g)
        g2 = seq.replay(s)
        self.assertEqual(g2.num_nodes, 3)
        self.assertEqual(len(g2.constraint_edges()), 3)

    def test_replay_skips_structural_edges(self):
        g = sg.SketchGraph()
        g.add_primitive("l0", "line")
        g.add_subprimitive("l0.end", "l0")
        g.add_primitive("l1", "line")
        g.add_subprimitive("l1.start", "l1")
        g.add_constraint("distance", ("l0.end", "l1.start"), value=5.0)
        s = seq.constraints_last_sequence(g)
        self.assertTrue(s.is_valid())
        g2 = seq.replay(s)
        # 4 nodes, 1 real distance constraint (structural coincident edges skipped).
        self.assertEqual(g2.num_nodes, 4)
        self.assertEqual(len(g2.constraint_edges()), 1)

    def test_invalid_sequence_rejected(self):
        bad = seq.ConstructionSequence((
            seq.EdgeOp("parallel", ("a", "b"), 0),
            seq.NodeOp("a", "line"),
            seq.NodeOp("b", "line"),
        ))
        self.assertFalse(bad.is_valid())
        with self.assertRaises(ValueError):
            seq.replay(bad)


class OrderingStatsTests(unittest.TestCase):
    def test_degree_by_position(self):
        g = sg.build_from_sketch(
            [("a", "line"), ("b", "line"), ("c", "line")],
            [("parallel", ("a", "b")), ("parallel", ("a", "c"))],
        )
        # 'a' is a common anchor -> higher degree, and appears first.
        degs = seq.degree_by_position(g)
        self.assertEqual(degs[0], 2)
        self.assertTrue(degs[0] >= degs[1])

    def test_ordering_adjacency_fraction(self):
        g = _triangle()
        # consecutive nodes l0-l1 and l1-l2 are adjacent (both coincident) -> 1.0
        frac = seq.ordering_adjacency_fraction(g)
        self.assertEqual(frac, 1.0)

    def test_adjacency_fraction_partial(self):
        g = sg.build_from_sketch(
            [("a", "line"), ("b", "line"), ("c", "line")],
            [("parallel", ("a", "b"))],  # a-b adjacent, b-c not
        )
        self.assertAlmostEqual(seq.ordering_adjacency_fraction(g), 0.5)

    def test_adjacency_fraction_single_node(self):
        g = sg.SketchGraph()
        g.add_primitive("only", "circle")
        self.assertEqual(seq.ordering_adjacency_fraction(g), 0.0)


if __name__ == "__main__":
    unittest.main()
