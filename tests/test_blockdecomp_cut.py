"""Tests for geometry.blockdecomp_cut."""

import unittest

from harnesscad.domain.geometry.mesh.blockdecomp_domain import Shape
from harnesscad.domain.geometry.mesh.blockdecomp_cut import (
    CutAction,
    cut_candidates,
    cut_from_vertex,
    full_cut,
    merge,
    split_step,
)


class TestFullCut(unittest.TestCase):
    def test_vertical_cut_splits_rectangle(self):
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 2.0)])
        parts = full_cut(r, "vertical", 2.0)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(sum(p.area() for p in parts), 8.0)
        for p in parts:
            self.assertAlmostEqual(p.area(), 4.0)

    def test_horizontal_cut(self):
        r = Shape.from_rectangles([(0.0, 0.0, 2.0, 4.0)])
        parts = full_cut(r, "horizontal", 2.0)
        self.assertEqual(len(parts), 2)

    def test_cut_on_boundary_is_ineffective(self):
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 2.0)])
        parts = full_cut(r, "vertical", 0.0)
        self.assertEqual(len(parts), 1)

    def test_interior_line_refines_and_splits(self):
        # A cut at an interior line not yet in the mesh is refined in and splits.
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 2.0)])
        parts = full_cut(r, "vertical", 1.3)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(sum(p.area() for p in parts), 8.0)

    def test_cut_outside_extent_is_ineffective(self):
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 2.0)])
        parts = full_cut(r, "vertical", 5.0)
        self.assertEqual(len(parts), 1)


class TestLShapeCut(unittest.TestCase):
    def setUp(self):
        # L-shape; the reentrant corner is at (1, 1).
        self.l = Shape.from_rectangles(
            [(0.0, 0.0, 2.0, 1.0), (0.0, 0.0, 1.0, 2.0)]
        )

    def test_cut_from_reentrant_vertex_yields_two_rectangles(self):
        action = CutAction(vertex=(1.0, 1.0), direction="y")  # vertical x=1
        parts = cut_from_vertex(self.l, action)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p.is_rectangle() for p in parts))

    def test_split_step_classifies_quads(self):
        action = CutAction(vertex=(1.0, 1.0), direction="y")
        res = split_step(self.l, action)
        self.assertTrue(res.is_effective)
        self.assertEqual(len(res.quads), 2)
        self.assertEqual(len(res.non_quads), 0)

    def test_ineffective_cut_leaves_non_quad(self):
        # Horizontal cut from the top vertex at y=2 lies on the boundary.
        action = CutAction(vertex=(1.0, 2.0), direction="x")
        res = split_step(self.l, action)
        self.assertFalse(res.is_effective)


class TestCutCandidates(unittest.TestCase):
    def test_candidate_count(self):
        r = Shape.from_rectangles([(0.0, 0.0, 2.0, 2.0)])
        cands = cut_candidates(r)
        # 4 corners x 2 directions = 8 candidate actions.
        self.assertEqual(len(cands), 8)

    def test_deterministic_order(self):
        r = Shape.from_rectangles([(0.0, 0.0, 2.0, 2.0)])
        self.assertEqual(cut_candidates(r), cut_candidates(r))


class TestMerge(unittest.TestCase):
    def test_merge_recovers_area_and_records_boundary(self):
        r = Shape.from_rectangles([(0.0, 0.0, 4.0, 2.0)])
        parts = full_cut(r, "vertical", 2.0)
        merged, internal = merge(parts)
        self.assertAlmostEqual(merged.area(), 4.0 * 2.0)
        self.assertTrue(merged.is_rectangle())
        # One imprinted internal cut boundary at x = 2 (as a single edge).
        self.assertEqual(len(internal), 1)
        self.assertEqual(internal[0], ((2.0, 0.0), (2.0, 2.0)))


if __name__ == "__main__":
    unittest.main()
