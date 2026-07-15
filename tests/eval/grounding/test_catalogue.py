"""The field map is queryable, and it makes the CAD-viewport blind spot legible."""

import unittest

from harnesscad.eval.grounding.catalogue import (
    BENCHMARKS, GROUNDING_MODELS, bbox_grounders, benchmarks_for, cad_gap,
    counts, find_benchmark, find_grounding_model,
)


class TestBenchmarks(unittest.TestCase):
    def test_the_named_benchmarks_are_present(self):
        for name in ("OSWorld", "Windows Agent Arena", "ScreenSpot", "tau-bench"):
            self.assertIsNotNone(find_benchmark(name), name)

    def test_domain_selector(self):
        mobile = {b.name for b in benchmarks_for("mobile")}
        self.assertIn("AndroidWorld", mobile)
        self.assertIn("A3", mobile)

    def test_tool_api_benchmarks_have_no_a11y_tree(self):
        for name in ("tau-bench", "AppWorld"):
            self.assertFalse(find_benchmark(name).has_accessibility_tree)


class TestGroundingModels(unittest.TestCase):
    def test_os_atlas_returns_bbox(self):
        m = find_grounding_model("OS-Atlas")
        self.assertIsNotNone(m)
        self.assertTrue(m.outputs_bbox)

    def test_showui_returns_normalised_point(self):
        self.assertFalse(find_grounding_model("ShowUI").outputs_bbox)

    def test_bbox_grounders_need_midpoint_extraction(self):
        names = {m.name for m in bbox_grounders()}
        self.assertIn("OS-Atlas", names)
        self.assertNotIn("ShowUI", names)


class TestCadGap(unittest.TestCase):
    def test_no_cad_viewport_benchmark_exists(self):
        gap = cad_gap()
        self.assertEqual(gap["cad_viewport_benchmarks"], [])

    def test_all_listed_grounders_come_from_a11y_scrape(self):
        gap = cad_gap()
        self.assertEqual(len(gap["grounders_from_a11y_scrape"]),
                         len(GROUNDING_MODELS))

    def test_finding_names_the_opaque_node(self):
        self.assertIn("opaque node", cad_gap()["finding"])


class TestCounts(unittest.TestCase):
    def test_counts_match_the_tables(self):
        c = counts()
        self.assertEqual(c["benchmarks"], len(BENCHMARKS))
        self.assertEqual(c["grounding_models"], len(GROUNDING_MODELS))


if __name__ == "__main__":
    unittest.main()
