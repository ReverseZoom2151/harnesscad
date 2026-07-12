"""Tests for drawings.muse2_svg_view_metrics."""

import unittest

from drawings.muse2_svg_view_metrics import (
    all_views_present,
    analyze_svg_text,
    boxes_overlap,
    estimate_components,
    parse_dimension,
    path_bbox,
)

_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="200mm" height="150mm">'
    '<text>Isometric</text><text>Top</text><text>Front</text><text>Right</text>'
    '<path d="M 0 0 L 10 0 L 10 10 L 0 10 Z"/>'
    '<path d="M 2 2 L 8 2 L 8 8 L 2 8 Z"/>'
    '<path d="M 100 100 L 110 100 L 110 110 L 100 110 Z"/>'
    '</svg>'
)


class DimensionTests(unittest.TestCase):
    def test_mm(self):
        self.assertEqual(parse_dimension("120mm"), 120.0)

    def test_px(self):
        self.assertEqual(parse_dimension("96px"), 96.0)

    def test_bare(self):
        self.assertEqual(parse_dimension("42"), 42.0)

    def test_bad(self):
        self.assertEqual(parse_dimension("auto"), 0.0)
        self.assertEqual(parse_dimension(None), 0.0)


class BBoxTests(unittest.TestCase):
    def test_bbox_extent(self):
        self.assertEqual(path_bbox("M 0 0 L 10 4 L 3 9"), (0.0, 0.0, 10.0, 9.0))

    def test_too_few_points(self):
        self.assertIsNone(path_bbox("M 1 2"))

    def test_overlap_true(self):
        self.assertTrue(boxes_overlap((0, 0, 10, 10), (5, 5, 15, 15)))

    def test_overlap_false(self):
        self.assertFalse(boxes_overlap((0, 0, 10, 10), (20, 20, 30, 30)))


class EstimateComponentsTests(unittest.TestCase):
    def test_two_overlapping_boxes_one_group(self):
        boxes = [(0, 0, 10, 10), (2, 2, 8, 8)]
        self.assertEqual(estimate_components(boxes), 1)

    def test_two_disjoint_boxes_two_groups(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
        self.assertEqual(estimate_components(boxes), 2)

    def test_empty(self):
        self.assertEqual(estimate_components([]), 0)


class AnalyzeSvgTests(unittest.TestCase):
    def test_full_analysis(self):
        m = analyze_svg_text(_SVG)
        self.assertEqual(m["view_labels"], ("Isometric", "Top", "Front", "Right"))
        self.assertEqual(m["total_path_count"], 3)
        self.assertEqual(m["text_count"], 4)
        # Two nested boxes near origin -> 1 group; the far box -> 1 group = 2.
        self.assertEqual(m["estimated_component_count"], 2)
        self.assertEqual(m["width_mm"], 200.0)
        self.assertEqual(m["height_mm"], 150.0)
        self.assertTrue(all_views_present(m))

    def test_missing_views(self):
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
               '<text>Top</text></svg>')
        m = analyze_svg_text(svg)
        self.assertEqual(m["view_labels"], ("Top",))
        self.assertFalse(all_views_present(m))
        self.assertEqual(m["estimated_component_count"], 0)


if __name__ == "__main__":
    unittest.main()
