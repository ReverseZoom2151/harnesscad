"""Tests for the segmentation boundary F-score (joint SDF paper)."""

import unittest

from bench.jointsdf_boundary_fscore import (
    boundary_edges,
    boundary_prf,
    boundary_f1,
)


def _chain(n):
    """Undirected path graph 0-1-2-...-(n-1)."""
    adj = {i: set() for i in range(n)}
    for i in range(n - 1):
        adj[i].add(i + 1)
        adj[i + 1].add(i)
    return adj


class BoundaryEdgesTest(unittest.TestCase):
    def test_single_boundary(self):
        adj = _chain(6)
        labels = [0, 0, 0, 1, 1, 1]
        self.assertEqual(boundary_edges(adj, labels), {(2, 3)})

    def test_no_boundary(self):
        adj = _chain(4)
        self.assertEqual(boundary_edges(adj, [7, 7, 7, 7]), set())


class BoundaryPRFTest(unittest.TestCase):
    def test_perfect_match(self):
        adj = _chain(6)
        gt = [0, 0, 0, 1, 1, 1]
        pred = [5, 5, 5, 9, 9, 9]  # palette differs, boundary identical
        self.assertEqual(boundary_prf(adj, pred, gt), (1.0, 1.0, 1.0))

    def test_both_empty_is_perfect(self):
        adj = _chain(4)
        self.assertEqual(boundary_prf(adj, [0, 0, 0, 0], [1, 1, 1, 1]), (1.0, 1.0, 1.0))

    def test_off_by_one_zero_tolerance(self):
        adj = _chain(6)
        gt = [0, 0, 0, 1, 1, 1]      # boundary (2,3)
        pred = [0, 0, 1, 1, 1, 1]    # boundary (1,2)
        p, r, f = boundary_prf(adj, pred, gt, tolerance=0)
        # endpoints of (1,2) are nodes 1,2; gt boundary nodes are 2,3 -> node 2
        # shared, so with zero tolerance it still matches via incident node.
        self.assertEqual(f, 1.0)

    def test_far_boundary_no_match(self):
        adj = _chain(8)
        gt = [0, 0, 0, 0, 1, 1, 1, 1]    # boundary (3,4)
        pred = [0, 1, 1, 1, 1, 1, 1, 1]  # boundary (0,1)
        p, r, f = boundary_prf(adj, pred, gt, tolerance=0)
        self.assertEqual(f, 0.0)

    def test_tolerance_recovers_match(self):
        adj = _chain(8)
        gt = [0, 0, 0, 0, 1, 1, 1, 1]    # boundary (3,4)
        pred = [0, 0, 1, 1, 1, 1, 1, 1]  # boundary (1,2), 2 hops away from node 4
        f_strict = boundary_f1(adj, pred, gt, tolerance=0)
        f_tol = boundary_f1(adj, pred, gt, tolerance=3)
        self.assertEqual(f_strict, 0.0)
        self.assertEqual(f_tol, 1.0)

    def test_extra_predicted_boundary_lowers_precision(self):
        adj = _chain(6)
        gt = [0, 0, 0, 1, 1, 1]          # boundary (2,3)
        pred = [0, 2, 0, 1, 1, 1]        # boundaries (0,1),(1,2),(2,3)
        p, r, f = boundary_prf(adj, pred, gt, tolerance=0)
        self.assertEqual(r, 1.0)
        self.assertLess(p, 1.0)


if __name__ == "__main__":
    unittest.main()
