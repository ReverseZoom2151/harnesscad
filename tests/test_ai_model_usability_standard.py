"""Tests for quality.ai_model_usability_standard."""

import unittest

from harnesscad.eval.quality.geometry.ai_model_usability_standard import (
    BASIC_DEFECT_CHECKS,
    POLYGON_BUDGETS,
    classify_variability,
    evaluate_model_usability,
    loop_size_statistics,
    loop_sizes_from_loops,
    mesh_defect_readiness,
    polygon_budget_check,
    quad_topology,
)


class LoopSizeStatisticsTest(unittest.TestCase):
    def test_uniform_loops_zero_cv_low(self):
        stats = loop_size_statistics([4, 4, 4, 4])
        self.assertEqual(stats.closed_loop_count, 4)
        self.assertEqual(stats.average_loop_size, 4.0)
        self.assertEqual(stats.standard_deviation, 0.0)
        self.assertEqual(stats.coefficient_of_variation, 0.0)
        self.assertEqual(stats.variability, "low")

    def test_known_mean_std_cv(self):
        # sizes 2 and 4: mean 3, population std 1, cv = 1/3.
        stats = loop_size_statistics([2, 4])
        self.assertAlmostEqual(stats.average_loop_size, 3.0)
        self.assertAlmostEqual(stats.standard_deviation, 1.0)
        self.assertAlmostEqual(stats.coefficient_of_variation, 1.0 / 3.0)
        self.assertEqual(stats.variability, "moderate")

    def test_high_variability_band(self):
        # mean 3, sizes 1 and 5 -> std 2, cv 2/3 > 0.5.
        stats = loop_size_statistics([1, 5])
        self.assertGreater(stats.coefficient_of_variation, 0.5)
        self.assertEqual(stats.variability, "high")

    def test_empty_is_zeroed_low(self):
        stats = loop_size_statistics([])
        self.assertEqual(stats.closed_loop_count, 0)
        self.assertEqual(stats.coefficient_of_variation, 0.0)
        self.assertEqual(stats.variability, "low")

    def test_classify_boundaries(self):
        self.assertEqual(classify_variability(0.05), "low")
        self.assertEqual(classify_variability(0.10), "moderate")  # not < 0.10
        self.assertEqual(classify_variability(0.50), "moderate")  # not > 0.50
        self.assertEqual(classify_variability(0.60), "high")

    def test_model_a_band_matches_paper(self):
        # Table 9 Model A: CV 0.293 -> the paper calls this consistent/uniform.
        self.assertEqual(classify_variability(0.293), "moderate")
        # Table 9 Model G: CV 1.682 -> high variability.
        self.assertEqual(classify_variability(1.682), "high")

    def test_loop_sizes_from_loops(self):
        loops = [["a", "b", "c"], ["d", "e"], []]
        self.assertEqual(loop_sizes_from_loops(loops), [3, 2, 0])


class QuadTopologyTest(unittest.TestCase):
    def test_quad_majority(self):
        qt = quad_topology(80, 100)
        self.assertAlmostEqual(qt.quad_fraction, 0.8)
        self.assertTrue(qt.quad_based)

    def test_triangle_majority(self):
        qt = quad_topology(10, 100)
        self.assertFalse(qt.quad_based)

    def test_zero_faces(self):
        qt = quad_topology(0, 0)
        self.assertEqual(qt.quad_fraction, 0.0)
        self.assertFalse(qt.quad_based)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            quad_topology(-1, 10)
        with self.assertRaises(ValueError):
            quad_topology(10, 5)


class PolygonBudgetTest(unittest.TestCase):
    def test_within(self):
        r = polygon_budget_check(5000, "low_detail_character")
        self.assertEqual(r.status, "within")

    def test_under_and_over(self):
        self.assertEqual(polygon_budget_check(100, "simple_prop").status, "under")
        self.assertEqual(polygon_budget_check(9000, "complex_prop").status, "over")

    def test_inclusive_bounds(self):
        self.assertEqual(polygon_budget_check(500, "simple_prop").status, "within")
        self.assertEqual(polygon_budget_check(1500, "simple_prop").status, "within")

    def test_all_categories_present(self):
        self.assertEqual(
            set(POLYGON_BUDGETS),
            {
                "low_detail_character",
                "high_detail_character",
                "simple_prop",
                "complex_prop",
            },
        )

    def test_unknown_category(self):
        with self.assertRaises(KeyError):
            polygon_budget_check(1000, "vehicle")

    def test_negative(self):
        with self.assertRaises(ValueError):
            polygon_budget_check(-1, "simple_prop")


class MeshDefectReadinessTest(unittest.TestCase):
    def test_clean(self):
        r = mesh_defect_readiness({})
        self.assertTrue(r.clean)
        self.assertEqual(r.total_defects, 0)
        self.assertEqual(r.failing_checks, ())
        self.assertEqual(set(r.counts), set(BASIC_DEFECT_CHECKS))

    def test_model_f_from_table8(self):
        # Table 8, model F: 2 intersecting, 3370 zero-area, 56 non-flat.
        r = mesh_defect_readiness(
            {
                "intersecting_faces": 2,
                "zero_area_faces": 3370,
                "non_flat_faces": 56,
            }
        )
        self.assertFalse(r.clean)
        self.assertEqual(r.total_defects, 2 + 3370 + 56)
        self.assertEqual(
            r.failing_checks,
            ("intersecting_faces", "zero_area_faces", "non_flat_faces"),
        )

    def test_model_d_non_manifold(self):
        # Table 8, model D: 6 non-manifold, 2 intersecting, 1 zero-length, 25 non-flat.
        r = mesh_defect_readiness(
            {
                "non_manifold_edges": 6,
                "intersecting_faces": 2,
                "zero_length_edges": 1,
                "non_flat_faces": 25,
            }
        )
        self.assertFalse(r.clean)
        self.assertIn("non_manifold_edges", r.failing_checks)

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            mesh_defect_readiness({"typo_edges": 1})

    def test_negative_count(self):
        with self.assertRaises(ValueError):
            mesh_defect_readiness({"zero_area_faces": -1})


class EvaluateModelUsabilityTest(unittest.TestCase):
    def test_usable(self):
        report = evaluate_model_usability(
            {},
            [4, 4, 4, 5],
            face_count=5000,
            category="low_detail_character",
        )
        self.assertEqual(report.verdict, "usable")
        self.assertEqual(report.reasons, ())

    def test_not_ready_on_defect(self):
        report = evaluate_model_usability(
            {"zero_area_faces": 91},
            [4, 4],
            face_count=8752,
            category="low_detail_character",
        )
        self.assertEqual(report.verdict, "not_ready")
        self.assertTrue(any("mesh defects" in r for r in report.reasons))

    def test_needs_refinement_high_cv(self):
        report = evaluate_model_usability(
            {},
            [1, 5, 1, 5],  # mean 3, std 2, cv 0.667 high
            face_count=5000,
            category="low_detail_character",
        )
        self.assertEqual(report.verdict, "needs_refinement")

    def test_needs_refinement_over_budget(self):
        report = evaluate_model_usability(
            {},
            [4, 4, 4],
            face_count=25000,
            category="high_detail_character",
        )
        self.assertEqual(report.verdict, "needs_refinement")
        self.assertTrue(any("budget" in r for r in report.reasons))

    def test_budget_optional(self):
        report = evaluate_model_usability({}, [4, 4, 4])
        self.assertIsNone(report.budget)
        self.assertEqual(report.verdict, "usable")

    def test_defect_dominates_over_budget_and_topology(self):
        report = evaluate_model_usability(
            {"non_flat_faces": 43},
            [1, 5, 1, 5],
            face_count=25000,
            category="high_detail_character",
        )
        self.assertEqual(report.verdict, "not_ready")


if __name__ == "__main__":
    unittest.main()
