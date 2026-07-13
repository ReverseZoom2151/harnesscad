"""Tests for reconstruction.dontmesh_cell_graph."""

import unittest

from harnesscad.domain.geometry.dontmesh_halfspace_csg import CSGModel, axis_box_cell
from harnesscad.domain.reconstruction.dontmesh_cell_graph import (
    build_cell_graph,
    is_plausible_sequence,
    plausible_sequences,
)


def _chain_model():
    # Three boxes in a touching row along x: 0-1, 1-2, 2-3.
    return CSGModel((
        axis_box_cell((0, 0, 0), (1, 1, 1)),
        axis_box_cell((1, 0, 0), (2, 1, 1)),
        axis_box_cell((2, 0, 0), (3, 1, 1)),
    ))


PROBE = ((-1, -1, -1), (4, 4, 4))


class TestCellGraph(unittest.TestCase):
    def test_chain_adjacency(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        self.assertEqual(g.n, 3)
        # 0-1 and 1-2 touch; 0-2 do not.
        self.assertIn(1, g.neighbours(0))
        self.assertIn(2, g.neighbours(1))
        self.assertNotIn(2, g.neighbours(0))

    def test_connected(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        self.assertTrue(g.is_connected())

    def test_disconnected(self):
        m = CSGModel((
            axis_box_cell((0, 0, 0), (1, 1, 1)),
            axis_box_cell((10, 10, 10), (11, 11, 11)),
        ))
        g = build_cell_graph(m, ((-1, -1, -1), (12, 12, 12)), res=14)
        self.assertFalse(g.is_connected())
        self.assertEqual(len(g.components()), 2)

    def test_edges(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        self.assertEqual(g.edges(), [(0, 1), (1, 2)])


class TestPlausibleSequences(unittest.TestCase):
    def test_chain_sequences(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        seqs = plausible_sequences(g)
        # A path graph 0-1-2 admits these connected orderings:
        # start 0: 0,1,2 ; start 1: 1,0,2 / 1,2,0 ; start 2: 2,1,0
        expected = {(0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0)}
        self.assertEqual(set(seqs), expected)
        # (0,2,1) is NOT plausible: 2 does not touch 0.
        self.assertNotIn((0, 2, 1), set(seqs))

    def test_all_are_valid(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        for seq in plausible_sequences(g):
            self.assertTrue(is_plausible_sequence(g, seq))

    def test_reject_invalid_sequence(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        self.assertFalse(is_plausible_sequence(g, (0, 2, 1)))
        self.assertFalse(is_plausible_sequence(g, (0, 1)))  # not a permutation

    def test_limit(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        limited = plausible_sequences(g, limit=2)
        self.assertEqual(len(limited), 2)

    def test_determinism(self):
        g = build_cell_graph(_chain_model(), PROBE, res=12)
        self.assertEqual(plausible_sequences(g), plausible_sequences(g))

    def test_single_cell(self):
        m = CSGModel((axis_box_cell((0, 0, 0), (1, 1, 1)),))
        g = build_cell_graph(m, PROBE, res=10)
        self.assertEqual(plausible_sequences(g), [(0,)])
        self.assertTrue(g.is_connected())


if __name__ == "__main__":
    unittest.main()
