"""Tests for bench.muse2_geometry_issue_flags."""

import unittest

from harnesscad.eval.bench.protocols.muse2_geometry_issue_flags import (
    classify_geometry_issues,
    geometry_valid,
    to_funnel_geometry,
)


class ClassifyTests(unittest.TestCase):
    def test_clean_model_all_true(self):
        f = classify_geometry_issues([], code_valid=True)
        self.assertTrue(f["watertight"])
        self.assertTrue(f["manifold"])
        self.assertTrue(f["self_intersection_free"])
        self.assertTrue(geometry_valid(f))

    def test_code_invalid_all_none(self):
        f = classify_geometry_issues([{"issue_type": "Watertightness",
                                       "severity": "error"}], code_valid=False)
        self.assertIsNone(f["watertight"])
        self.assertIsNone(f["manifold"])
        self.assertFalse(geometry_valid(f))

    def test_non_manifold_error_fails_watertight_and_manifold(self):
        issues = [{"issue_type": "NonManifoldEdge", "severity": "error"}]
        f = classify_geometry_issues(issues, code_valid=True)
        # Combined watertight includes the manifold check -> both False.
        self.assertFalse(f["watertight"])
        self.assertFalse(f["manifold"])
        # ...but watertight_strict ignores manifoldness -> True.
        self.assertTrue(f["watertight_strict"])
        self.assertEqual(f["non_manifold_error_count"], 1)

    def test_open_edge_fails_watertight_not_manifold(self):
        issues = [{"issue_type": "Watertightness", "severity": "error"}]
        f = classify_geometry_issues(issues, code_valid=True)
        self.assertFalse(f["watertight"])
        self.assertFalse(f["watertight_strict"])
        self.assertTrue(f["manifold"])
        self.assertEqual(f["watertight_error_count"], 1)

    def test_warning_does_not_fail_flag(self):
        issues = [{"issue_type": "SelfIntersection", "severity": "warning"}]
        f = classify_geometry_issues(issues, code_valid=True)
        self.assertTrue(f["self_intersection_free"])

    def test_volume_error_count_sums_zero_and_negative(self):
        issues = [{"issue_type": "ZeroVolume", "severity": "error"},
                  {"issue_type": "NegativeVolume", "severity": "error"}]
        f = classify_geometry_issues(issues, code_valid=True)
        self.assertEqual(f["volume_error_count"], 2)
        self.assertFalse(f["volume_valid"])


class ProjectionTests(unittest.TestCase):
    def test_to_funnel_geometry(self):
        f = classify_geometry_issues([], code_valid=True)
        proj = to_funnel_geometry(f)
        self.assertEqual(proj, {"watertight": 1, "manifold": 1,
                                "self_intersection_free": 1})

    def test_to_funnel_none_is_zero(self):
        f = classify_geometry_issues([], code_valid=False)
        proj = to_funnel_geometry(f)
        self.assertEqual(proj["watertight"], 0)

    def test_geometry_valid_requires_bbox_and_volume(self):
        issues = [{"issue_type": "BoundingBox", "severity": "error"}]
        f = classify_geometry_issues(issues, code_valid=True)
        self.assertFalse(f["bbox_valid"])
        self.assertFalse(geometry_valid(f))


if __name__ == "__main__":
    unittest.main()
