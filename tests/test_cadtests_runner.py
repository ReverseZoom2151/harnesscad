"""Tests for bench.cadtests_runner (suite conjunction, requirements, poses)."""

import unittest

from harnesscad.eval.bench.cadtests_assertions import (
    assert_aspect_ratio,
    assert_bbox_dimension,
    assert_face_count,
    assert_typed_face_count,
    assert_valid_solid,
    assert_volume,
)
from harnesscad.eval.bench.cadtests_model import CADModel, Edge, Face, similarity_augmentations
from harnesscad.eval.bench.cadtests_runner import (
    requirement_groups,
    run_passing_set,
    run_suite,
)


def _box():
    faces = tuple(Face("plane", a) for a in (2.0, 2.0, 3.0, 3.0, 6.0, 6.0))
    edges = tuple(Edge("line", 1.0) for _ in range(12))
    return CADModel(faces, edges, 8, (0.0, 0.0, 0.0), (2.0, 3.0, 1.0), 6.0,
                    (1.0, 1.5, 0.5))


def _suite():
    return [
        assert_valid_solid(requirement="validity"),
        assert_face_count(6, requirement="topology"),
        assert_bbox_dimension("y", 3.0, requirement="dims"),
        assert_volume(6.0, requirement="volume"),
    ]


class TestConjunction(unittest.TestCase):
    def test_all_pass(self):
        sr = run_suite(_box(), _suite())
        self.assertTrue(sr.passed_all)
        self.assertEqual(sr.pass_rate, 1.0)
        self.assertEqual(sr.num_passed, 4)
        self.assertFalse(sr.invalid)

    def test_one_fail_breaks_conjunction(self):
        suite = _suite() + [assert_volume(999.0, requirement="volume")]
        sr = run_suite(_box(), suite)
        self.assertFalse(sr.passed_all)
        self.assertEqual(sr.pass_rate, 0.0)


class TestRequirementScore(unittest.TestCase):
    def test_full_requirement_score(self):
        sr = run_suite(_box(), _suite())
        self.assertAlmostEqual(sr.requirement_score, 1.0)

    def test_group_fails_if_any_test_fails(self):
        # two tests in one requirement group; one fails -> group unsatisfied
        suite = [
            assert_volume(6.0, requirement="volume"),
            assert_volume(999.0, requirement="volume"),
            assert_face_count(6, requirement="topology"),
        ]
        sr = run_suite(_box(), suite)
        self.assertFalse(sr.requirement_satisfied["volume"])
        self.assertTrue(sr.requirement_satisfied["topology"])
        # 1 of 2 requirement groups satisfied
        self.assertAlmostEqual(sr.requirement_score, 0.5)

    def test_ungrouped_tests_are_singletons(self):
        suite = [assert_volume(6.0), assert_face_count(6)]
        groups = requirement_groups(suite)
        self.assertEqual(len(groups), 2)
        sr = run_suite(_box(), suite)
        self.assertAlmostEqual(sr.requirement_score, 1.0)


class TestInvalidGeneration(unittest.TestCase):
    def test_none_model_is_invalid(self):
        sr = run_suite(None, _suite())
        self.assertTrue(sr.invalid)
        self.assertEqual(sr.pass_rate, 0.0)
        self.assertEqual(sr.requirement_score, 0.0)
        self.assertEqual(sr.num_passed, 0)
        # every test reported as failed with an error
        self.assertTrue(all(not r.passed and not r.valid for r in sr.results))


class TestCategoryCounts(unittest.TestCase):
    def test_counts(self):
        sr = run_suite(_box(), _suite())
        # validity, topology, dims, volumetric categories represented
        self.assertEqual(sr.category_counts["solid_shell_validity"], (1, 1))
        self.assertEqual(sr.category_counts["topology"], (1, 1))
        self.assertEqual(sr.category_counts["dimensions_ratios"], (1, 1))
        self.assertEqual(sr.category_counts["volumetric"], (1, 1))
        self.assertEqual(sr.category_counts["geometric_types"], (0, 0))


class TestPoseInvariance(unittest.TestCase):
    def test_pose_invariant_suite(self):
        m = _box()
        # a suite of pose/scale-invariant checks (counts, valid solid, aspect)
        suite = [
            assert_valid_solid(),
            assert_face_count(6),
            assert_typed_face_count("plane", 6),
        ]
        variants = similarity_augmentations(
            m, scales=(1.0, 2.0), rotations=(None, "z", "x"),
            translations=((0.0, 0.0, 0.0), (7.0, -3.0, 2.0)))
        inv = run_passing_set(variants, suite)
        self.assertTrue(inv.invariant_pass)
        self.assertEqual(inv.unstable_tests, ())

    def test_absolute_dimension_is_unstable_under_scaling(self):
        m = _box()
        # an absolute-dimension test is NOT scale invariant
        suite = [assert_bbox_dimension("y", 3.0)]
        variants = similarity_augmentations(m, scales=(1.0, 2.0))
        inv = run_passing_set(variants, suite)
        self.assertFalse(inv.invariant_pass)
        self.assertIn("assert_bbox_dimension", inv.unstable_tests[0])

    def test_rotation_invariant_largest_axis(self):
        # aspect ratio y/x = 1.5; after 90-deg z rotation dims swap so the
        # same physical faces move -- an absolute aspect test flips, confirming
        # the augmentation exposes pose sensitivity.
        m = _box()
        suite = [assert_aspect_ratio("y", "x", 1.5)]
        variants = similarity_augmentations(m, scales=(1.0,), rotations=(None, "z"),
                                            include_scale=False)
        inv = run_passing_set(variants, suite)
        self.assertFalse(inv.invariant_pass)

    def test_empty_passing_set_raises(self):
        with self.assertRaises(ValueError):
            run_passing_set([], _suite())


if __name__ == "__main__":
    unittest.main()
