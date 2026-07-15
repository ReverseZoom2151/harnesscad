"""Tests for reconstruction.fitting.wire_assembly."""

import math
import unittest

from harnesscad.domain.reconstruction.fitting.wire_assembly import (
    circle_from_three_points,
    weld_endpoints,
    assemble_loops,
)


class TestCircle(unittest.TestCase):
    def test_unit_circle_xy(self):
        c, r, n = circle_from_three_points((1, 0, 0), (0, 1, 0), (-1, 0, 0))
        self.assertAlmostEqual(c[0], 0.0)
        self.assertAlmostEqual(c[1], 0.0)
        self.assertAlmostEqual(c[2], 0.0)
        self.assertAlmostEqual(r, 1.0)
        # normal parallel to z
        self.assertAlmostEqual(abs(n[2]), 1.0)

    def test_offset_circle(self):
        c, r, _ = circle_from_three_points((3, 0, 5), (2, 1, 5), (1, 0, 5))
        self.assertAlmostEqual(c[0], 2.0)
        self.assertAlmostEqual(c[1], 0.0)
        self.assertAlmostEqual(c[2], 5.0)
        self.assertAlmostEqual(r, 1.0)

    def test_collinear_raises(self):
        with self.assertRaises(ValueError):
            circle_from_three_points((0, 0, 0), (1, 1, 1), (2, 2, 2))


class TestWeld(unittest.TestCase):
    def test_shared_corner_welded(self):
        edges = [
            [(0, 0, 0), (1, 0, 0)],
            [(1.00001, 0, 0), (1, 1, 0)],  # start ~ end of edge 0
        ]
        w = weld_endpoints(edges, tol=1e-3)
        self.assertEqual(len(w.node_coords), 3)  # (0,0,0),(1,0,0),(1,1,0)
        self.assertEqual(w.edge_nodes[0][1], w.edge_nodes[1][0])

    def test_no_weld_when_far(self):
        edges = [[(0, 0, 0), (1, 0, 0)], [(5, 0, 0), (6, 0, 0)]]
        w = weld_endpoints(edges, tol=1e-3)
        self.assertEqual(len(w.node_coords), 4)


class TestAssemble(unittest.TestCase):
    def test_square_loop(self):
        # four edges of a unit square, given in scrambled order & directions
        edges = [
            [(1, 0, 0), (1, 1, 0)],   # right
            [(0, 0, 0), (1, 0, 0)],   # bottom
            [(1, 1, 0), (0, 1, 0)],   # top
            [(0, 1, 0), (0, 0, 0)],   # left
        ]
        res = assemble_loops(edges, tol=1e-6)
        self.assertEqual(len(res.loops), 1)
        self.assertEqual(len(res.open_chains), 0)
        loop = res.loops[0]
        self.assertEqual(len(loop), 4)
        # every edge used exactly once
        self.assertEqual(sorted(ei for ei, _ in loop), [0, 1, 2, 3])

    def test_loop_directions_consistent(self):
        edges = [
            [(0, 0, 0), (1, 0, 0)],
            [(1, 0, 0), (1, 1, 0)],
            [(1, 1, 0), (0, 1, 0)],
            [(0, 1, 0), (0, 0, 0)],
        ]
        res = assemble_loops(edges, tol=1e-6)
        loop = res.loops[0]
        # traverse and confirm continuity: end node of each == start of next
        weld = weld_endpoints(edges, tol=1e-6)
        cur_end = None
        for (ei, d) in loop:
            a, b = weld.edge_nodes[ei]
            start, end = (a, b) if d == +1 else (b, a)
            if cur_end is not None:
                self.assertEqual(start, cur_end)
            cur_end = end

    def test_two_disjoint_loops(self):
        square = [
            [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)],
            [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],
        ]
        tri = [
            [(5, 0, 0), (6, 0, 0)], [(6, 0, 0), (5, 1, 0)],
            [(5, 1, 0), (5, 0, 0)],
        ]
        res = assemble_loops(square + tri, tol=1e-6)
        self.assertEqual(len(res.loops), 2)
        sizes = sorted(len(l) for l in res.loops)
        self.assertEqual(sizes, [3, 4])

    def test_open_chain_reported(self):
        edges = [
            [(0, 0, 0), (1, 0, 0)],
            [(1, 0, 0), (2, 0, 0)],  # open path, no closure
        ]
        res = assemble_loops(edges, tol=1e-6)
        self.assertEqual(len(res.loops), 0)
        self.assertEqual(len(res.open_chains), 1)

    def test_deterministic(self):
        edges = [
            [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)],
            [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],
        ]
        self.assertEqual(assemble_loops(edges).loops,
                         assemble_loops(edges).loops)


if __name__ == "__main__":
    unittest.main()
