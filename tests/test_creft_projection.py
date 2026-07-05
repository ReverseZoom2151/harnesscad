"""Tests for creft_projection."""

import unittest

from drawings.creft_projection import (
    Box,
    Rect,
    View,
    model_bbox,
    project_three_views,
    project_view,
    silhouette_area,
)


class BoxTest(unittest.TestCase):
    def test_positive_sizes_required(self):
        Box(0, 0, 0, 1, 2, 3)
        with self.assertRaises(ValueError):
            Box(0, 0, 0, 0, 1, 1)
        with self.assertRaises(ValueError):
            Box(0, 0, 0, 1, -1, 1)

    def test_maxes(self):
        b = Box(1, 2, 3, 4, 5, 6)
        self.assertEqual((b.xmax, b.ymax, b.zmax), (5, 7, 9))


class ProjectViewTest(unittest.TestCase):
    def test_single_box_front_top_side(self):
        b = Box(0, 0, 0, 4, 6, 8)  # dx=4, dy=6, dz=8
        front = project_view([b], "front")
        top = project_view([b], "top")
        side = project_view([b], "side")
        self.assertEqual(front.rects[0], Rect(0, 0, 4, 8))   # X,Z
        self.assertEqual(top.rects[0], Rect(0, 0, 4, 6))     # X,Y
        self.assertEqual(side.rects[0], Rect(0, 0, 6, 8))    # Y,Z

    def test_axes_labels(self):
        v = project_view([Box(0, 0, 0, 1, 1, 1)], "front")
        self.assertEqual((v.horizontal, v.vertical), ("x", "z"))

    def test_unknown_view(self):
        with self.assertRaises(ValueError):
            project_view([], "isometric")


class ThreeViewTest(unittest.TestCase):
    def test_extents_match_bbox(self):
        boxes = [Box(0, 0, 0, 4, 6, 8), Box(2, 1, 3, 2, 2, 2)]
        views = project_three_views(boxes)
        dx, dy, dz = model_bbox(boxes)
        self.assertEqual((dx, dy, dz), (4, 6, 8))
        self.assertEqual(views["front"].horizontal_extent(), dx)
        self.assertEqual(views["front"].vertical_extent(), dz)
        self.assertEqual(views["top"].vertical_extent(), dy)
        self.assertEqual(views["side"].horizontal_extent(), dy)

    def test_empty_model(self):
        self.assertEqual(model_bbox([]), (0.0, 0.0, 0.0))
        v = View("front", "x", "z", ())
        self.assertEqual(v.bbox(), Rect(0.0, 0.0, 0.0, 0.0))


class SilhouetteAreaTest(unittest.TestCase):
    def test_single_rect(self):
        v = project_view([Box(0, 0, 0, 4, 1, 8)], "front")
        self.assertAlmostEqual(silhouette_area(v), 4 * 8)

    def test_overlap_counted_once(self):
        # Two identical front-projections overlap fully -> area of one.
        b = Box(0, 0, 0, 4, 1, 8)
        v = project_view([b, b], "front")
        self.assertAlmostEqual(silhouette_area(v), 32.0)

    def test_disjoint_sum(self):
        boxes = [Box(0, 0, 0, 2, 1, 2), Box(5, 0, 0, 2, 1, 2)]
        v = project_view(boxes, "front")
        self.assertAlmostEqual(silhouette_area(v), 4 + 4)

    def test_partial_overlap(self):
        # front rects: (0,0,4,4) and (2,2,4,4) -> union = 32 - overlap(4) = 28
        boxes = [Box(0, 0, 0, 4, 1, 4), Box(2, 0, 2, 4, 1, 4)]
        v = project_view(boxes, "front")
        self.assertAlmostEqual(silhouette_area(v), 28.0)

    def test_empty(self):
        self.assertEqual(silhouette_area(View("front", "x", "z", ())), 0.0)


if __name__ == "__main__":
    unittest.main()
