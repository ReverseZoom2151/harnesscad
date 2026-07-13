"""Tests for editing.autocad_layout_ops."""

import unittest

from harnesscad.domain.editing.autocad_layout_ops import (
    Align,
    align,
    distribute_centers,
    distribute_gaps,
)


def _apply(box, d):
    return (box[0] + d[0], box[1] + d[1], box[2] + d[0], box[3] + d[1])


class TestAlign(unittest.TestCase):
    def setUp(self):
        # three boxes of width 2, height 2 at differing positions
        self.boxes = [
            (0.0, 0.0, 2.0, 2.0),
            (5.0, 3.0, 7.0, 5.0),
            (10.0, -4.0, 12.0, -2.0),
        ]

    def test_left(self):
        ds = align(self.boxes, Align.LEFT)
        lefts = [_apply(b, d)[0] for b, d in zip(self.boxes, ds)]
        self.assertTrue(all(abs(l - 0.0) < 1e-9 for l in lefts))

    def test_right(self):
        ds = align(self.boxes, Align.RIGHT)
        rights = [_apply(b, d)[2] for b, d in zip(self.boxes, ds)]
        self.assertTrue(all(abs(r - 12.0) < 1e-9 for r in rights))

    def test_bottom(self):
        ds = align(self.boxes, Align.BOTTOM)
        bottoms = [_apply(b, d)[1] for b, d in zip(self.boxes, ds)]
        self.assertTrue(all(abs(bt - (-4.0)) < 1e-9 for bt in bottoms))

    def test_center_x_all_equal(self):
        ds = align(self.boxes, Align.CENTER)
        cxs = [(_apply(b, d)[0] + _apply(b, d)[2]) / 2 for b, d in zip(self.boxes, ds)]
        self.assertTrue(all(abs(c - cxs[0]) < 1e-9 for c in cxs))

    def test_only_moves_one_axis(self):
        ds = align(self.boxes, Align.LEFT)
        self.assertTrue(all(d[1] == 0.0 for d in ds))

    def test_empty(self):
        self.assertEqual(align([], Align.LEFT), [])


class TestDistributeCenters(unittest.TestCase):
    def test_even_centers(self):
        boxes = [
            (0.0, 0.0, 1.0, 1.0),   # center 0.5
            (3.0, 0.0, 4.0, 1.0),   # center 3.5
            (10.0, 0.0, 11.0, 1.0), # center 10.5
        ]
        ds = distribute_centers(boxes, axis="x")
        cxs = sorted((b[0] + d[0]) + 0.5 for b, d in zip(boxes, ds))
        # extremes preserved at 0.5 and 10.5, middle at 5.5
        self.assertAlmostEqual(cxs[0], 0.5)
        self.assertAlmostEqual(cxs[1], 5.5)
        self.assertAlmostEqual(cxs[2], 10.5)

    def test_two_boxes_noop(self):
        boxes = [(0.0, 0.0, 1.0, 1.0), (5.0, 0.0, 6.0, 1.0)]
        self.assertEqual(distribute_centers(boxes), [(0.0, 0.0), (0.0, 0.0)])


class TestDistributeGaps(unittest.TestCase):
    def test_fixed_gap(self):
        boxes = [
            (0.0, 0.0, 2.0, 1.0),   # width 2, anchor
            (100.0, 0.0, 103.0, 1.0),  # width 3
            (200.0, 0.0, 201.0, 1.0),  # width 1
        ]
        ds = distribute_gaps(boxes, spacing=1.0, axis="x")
        placed = [_apply(b, d) for b, d in zip(boxes, ds)]
        placed.sort(key=lambda b: b[0])
        # anchor ends at 2; next starts at 3 (gap 1), ends at 6; next starts 7
        self.assertAlmostEqual(placed[0][2], 2.0)
        self.assertAlmostEqual(placed[1][0], 3.0)
        self.assertAlmostEqual(placed[1][2], 6.0)
        self.assertAlmostEqual(placed[2][0], 7.0)

    def test_empty(self):
        self.assertEqual(distribute_gaps([], 1.0), [])


if __name__ == "__main__":
    unittest.main()
