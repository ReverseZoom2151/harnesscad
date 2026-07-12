"""Tests for geometry.libfive_dual_contour (dual contouring + QEF)."""

from __future__ import annotations

import math
import unittest

from geometry import libfive_frep_ir as ir
from geometry import libfive_dual_contour as dc
from geometry.libfive_dual_contour import QEF


class TestQEF(unittest.TestCase):
    def test_sharp_corner_placement(self):
        """Two perpendicular planes meeting at a corner: the QEF minimum must
        land exactly on the corner (not the rounded mass point)."""
        q = QEF(2)
        # plane 1: x = 1, normal (1, 0); crossing points along it
        q.insert((1.0, 0.2), (1.0, 0.0))
        q.insert((1.0, 0.9), (1.0, 0.0))
        # plane 2: y = 1, normal (0, 1)
        q.insert((0.3, 1.0), (0.0, 1.0))
        q.insert((0.7, 1.0), (0.0, 1.0))
        pos, rank = q.solve()
        self.assertEqual(rank, 2)  # both directions constrained
        self.assertAlmostEqual(pos[0], 1.0, places=6)
        self.assertAlmostEqual(pos[1], 1.0, places=6)

    def test_underdetermined_falls_back_to_mass_point(self):
        """A single plane leaves one direction free -> vertex stays at the mass
        point along that direction (rank 1)."""
        q = QEF(2)
        q.insert((0.0, 1.0), (0.0, 1.0))
        q.insert((1.0, 1.0), (0.0, 1.0))
        pos, rank = q.solve()
        self.assertEqual(rank, 1)
        self.assertAlmostEqual(pos[1], 1.0, places=6)      # constrained
        self.assertAlmostEqual(pos[0], 0.5, places=6)      # mass point (free)

    def test_diagonal_corner(self):
        q = QEF(2)
        s = math.sqrt(0.5)
        q.insert((0.5, 0.5), (s, s))       # plane through (0.5,0.5) normal 45deg
        q.insert((0.5, 0.5), (s, -s))      # orthogonal plane
        pos, rank = q.solve()
        self.assertEqual(rank, 2)
        self.assertAlmostEqual(pos[0], 0.5, places=6)
        self.assertAlmostEqual(pos[1], 0.5, places=6)


class TestDualContourCircle(unittest.TestCase):
    def setUp(self):
        self.g = ir.Graph()
        self.circle = ir.circle(self.g, 0.0, 0.0, 1.0)
        self.verts, self.segs = dc.dual_contour_2d(
            self.circle, (-2.0, -2.0, 2.0, 2.0), depth=5)

    def test_vertices_lie_on_circle(self):
        self.assertGreater(len(self.verts), 8)
        for (x, y) in self.verts:
            r = math.hypot(x, y)
            self.assertAlmostEqual(r, 1.0, delta=0.12)

    def test_forms_closed_loop(self):
        # every vertex on a closed curve has exactly two incident segments
        degree = [0] * len(self.verts)
        for (a, b) in self.segs:
            degree[a] += 1
            degree[b] += 1
        self.assertTrue(all(d == 2 for d in degree),
                        "contour is not a single closed loop")

    def test_segment_count_matches_vertices(self):
        # a closed loop has as many edges as vertices
        self.assertEqual(len(self.segs), len(self.verts))


class TestDualContourSquareKeepsCorners(unittest.TestCase):
    def test_square_has_sharp_corners(self):
        g = ir.Graph()
        sq = ir.rectangle(g, -1.0, -1.0, 1.0, 1.0)
        verts, segs = dc.dual_contour_2d(sq, (-2.0, -2.0, 2.0, 2.0), depth=5)
        self.assertGreater(len(verts), 4)
        # at least one vertex should sit very close to a true corner (1, 1),
        # which marching squares would round off
        near_corner = any(
            math.hypot(x - 1.0, y - 1.0) < 0.08 for (x, y) in verts)
        self.assertTrue(near_corner, "sharp corner was rounded off")


class TestIntervalPruning(unittest.TestCase):
    def test_only_boundary_cells_active(self):
        g = ir.Graph()
        c = ir.circle(g, 0.0, 0.0, 1.0)
        verts, _ = dc.dual_contour_2d(c, (-2.0, -2.0, 2.0, 2.0), depth=5)
        # With 2^5 = 32 cells per axis (1024 total), pruning must leave only a
        # thin boundary band active -- far fewer than the full grid.
        self.assertLess(len(verts), 200)


if __name__ == "__main__":
    unittest.main()
