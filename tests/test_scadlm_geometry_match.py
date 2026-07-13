"""Tests for bench.scadlm_geometry_match (binary-free SCAD evaluation metrics)."""

from __future__ import annotations

import unittest

from harnesscad.eval.bench.geometry.scadlm_geometry_match import (
    MatchReport,
    bbox_iou,
    best_of_k,
    centroid_offset,
    compile_rate,
    score,
    voxel_iou,
    volume_ratio,
)
from harnesscad.domain.geometry.sdf.scadlm_csg_eval import evaluate_source


class TestCompileMetrics(unittest.TestCase):
    def test_compile_rate(self):
        sources = ["cube(1);", "sphere(", "frobnicate(2);", "cylinder(h=1, r=1);"]
        self.assertEqual(compile_rate(sources), 0.5)

    def test_compile_rate_of_empty_list(self):
        self.assertEqual(compile_rate([]), 0.0)

    def test_warnings_do_not_fail_the_gate(self):
        self.assertEqual(compile_rate(["difference() { cube(1); }"]), 1.0)


class TestVoxelIou(unittest.TestCase):
    def test_identical_programs_score_one(self):
        t = evaluate_source("cube([4,4,4]);")
        self.assertEqual(voxel_iou(t, t, 8), 1.0)

    def test_disjoint_programs_score_zero(self):
        a = evaluate_source("cube(2);")
        b = evaluate_source("translate([50,0,0]) cube(2);")
        self.assertEqual(voxel_iou(a, b, 8), 0.0)

    def test_partial_overlap_is_between(self):
        a = evaluate_source("cube([10,10,10]);")
        b = evaluate_source("translate([5,0,0]) cube([10,10,10]);")
        iou = voxel_iou(a, b, 16)
        self.assertGreater(iou, 0.2)
        self.assertLess(iou, 0.5)

    def test_both_empty_scores_one(self):
        self.assertEqual(voxel_iou(None, None, 4), 1.0)

    def test_one_empty_scores_zero(self):
        self.assertEqual(voxel_iou(evaluate_source("cube(1);"), None, 4), 0.0)

    def test_is_symmetric_and_deterministic(self):
        a = evaluate_source("difference() { cube(6, center=true); sphere(3); }")
        b = evaluate_source("cube(5, center=true);")
        self.assertEqual(voxel_iou(a, b, 10), voxel_iou(b, a, 10))
        self.assertEqual(voxel_iou(a, b, 10), voxel_iou(a, b, 10))

    def test_bad_resolution_raises(self):
        with self.assertRaises(ValueError):
            voxel_iou(evaluate_source("cube(1);"), evaluate_source("cube(1);"), 0)


class TestCoarseMetrics(unittest.TestCase):
    def test_volume_ratio_detects_scale_error(self):
        a = evaluate_source("cube([10,10,10]);")
        b = evaluate_source("cube([5,10,10]);")
        self.assertAlmostEqual(volume_ratio(a, b, 10), 0.5, delta=0.02)

    def test_volume_ratio_identical(self):
        a = evaluate_source("cylinder(h=4, r=2);")
        self.assertEqual(volume_ratio(a, a, 8), 1.0)

    def test_bbox_iou_of_translated_copy(self):
        a = evaluate_source("cube([10,10,10]);")
        b = evaluate_source("translate([10,0,0]) cube([10,10,10]);")
        self.assertEqual(bbox_iou(a, b), 0.0)

    def test_bbox_iou_identical(self):
        a = evaluate_source("cube([2,3,4]);")
        self.assertEqual(bbox_iou(a, a), 1.0)

    def test_bbox_iou_half_overlap(self):
        a = evaluate_source("cube([10,10,10]);")
        b = evaluate_source("translate([5,0,0]) cube([10,10,10]);")
        self.assertAlmostEqual(bbox_iou(a, b), 5.0 / 15.0, places=9)

    def test_centroid_offset_detects_position_error(self):
        a = evaluate_source("cube(2, center=true);")
        b = evaluate_source("translate([3,4,0]) cube(2, center=true);")
        self.assertAlmostEqual(centroid_offset(a, b), 5.0, places=9)

    def test_centroid_offset_empty(self):
        self.assertEqual(centroid_offset(None, None), 0.0)
        self.assertEqual(centroid_offset(evaluate_source("cube(1);"), None),
                         float("inf"))


class TestScore(unittest.TestCase):
    def test_perfect_match(self):
        src = "difference() { cube(10, center=true); cylinder(h=20, r=3, center=true); }"
        report = score(src, src, resolution=10)
        self.assertTrue(report.compiles)
        self.assertFalse(report.failed)
        self.assertEqual(report.voxel_iou, 1.0)
        self.assertEqual(report.bbox_iou, 1.0)
        self.assertEqual(report.centroid_offset, 0.0)

    def test_syntax_error_scores_zero_with_reason(self):
        report = score("cube(", "cube(10);")
        self.assertFalse(report.compiles)
        self.assertTrue(report.failed)
        self.assertEqual(report.voxel_iou, 0.0)
        self.assertIn("syntax", report.reason)

    def test_unknown_module_scores_zero(self):
        report = score("frobnicate(2);", "cube(10);")
        self.assertIn("unknown-module", report.reason)

    def test_unsupported_construct_reports_not_evaluable(self):
        report = score("hull() { cube(1); translate([5,0,0]) sphere(1); }",
                       "cube(10);")
        self.assertIn("not evaluable", report.reason)

    def test_no_geometry_scores_zero(self):
        report = score("w = 4;", "cube(10);")
        self.assertIn("no geometry", report.reason)

    def test_bad_reference_raises(self):
        with self.assertRaises(ValueError):
            score("cube(1);", "cube(")

    def test_wrong_size_penalised(self):
        report = score("cube([5,10,10]);", "cube([10,10,10]);", resolution=10)
        self.assertTrue(report.compiles)
        self.assertAlmostEqual(report.volume_ratio, 0.5, delta=0.02)
        self.assertLess(report.voxel_iou, 0.6)

    def test_report_is_deterministic(self):
        a = "translate([1,0,0]) sphere(r=3);"
        b = "sphere(r=3);"
        self.assertEqual(score(a, b, 8), score(a, b, 8))


class TestBestOfK(unittest.TestCase):
    def test_picks_the_closest_candidate(self):
        reference = "cube([10,10,10]);"
        candidates = ["sphere(", "cube([2,2,2]);", "cube([10,10,10]);",
                      "cube([9,10,10]);"]
        index, report = best_of_k(candidates, reference, resolution=8)
        self.assertEqual(index, 2)
        self.assertEqual(report.voxel_iou, 1.0)

    def test_all_broken_candidates(self):
        index, report = best_of_k(["sphere(", "frobnicate(1);"], "cube(1);", 4)
        self.assertEqual(report.voxel_iou, 0.0)
        self.assertTrue(report.failed)
        self.assertEqual(index, 0)

    def test_empty_candidate_list(self):
        index, report = best_of_k([], "cube(1);", 4)
        self.assertEqual(index, -1)
        self.assertIsInstance(report, MatchReport)
        self.assertEqual(report.reason, "no candidates")


if __name__ == "__main__":
    unittest.main()
