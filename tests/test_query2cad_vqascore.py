import unittest

from harnesscad.eval.bench.judges.vqa_score import (
    format_vqa_question, vqascore, meets_threshold, aggregate_vqascore,
    best_candidate, stopping_trajectory, DEFAULT_THRESHOLD,
)


class TestQuestion(unittest.TestCase):
    def test_template(self):
        q = format_vqa_question("a water bottle")
        self.assertEqual(
            q, "Does this figure show a water bottle? Please answer yes or no.")

    def test_strips(self):
        self.assertEqual(format_vqa_question("  a cube  "),
                         format_vqa_question("a cube"))

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            format_vqa_question("   ")


class TestScore(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(vqascore(0.87), 0.87)

    def test_bounds_rejected(self):
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                vqascore(bad)

    def test_edges(self):
        self.assertEqual(vqascore(0.0), 0.0)
        self.assertEqual(vqascore(1.0), 1.0)


class TestThreshold(unittest.TestCase):
    def test_default_is_point_nine(self):
        self.assertEqual(DEFAULT_THRESHOLD, 0.9)

    def test_above(self):
        self.assertTrue(meets_threshold(0.95))

    def test_at_threshold_stops(self):
        self.assertTrue(meets_threshold(0.9))

    def test_below(self):
        self.assertFalse(meets_threshold(0.8))

    def test_custom_threshold(self):
        self.assertTrue(meets_threshold(0.7, threshold=0.6))
        self.assertFalse(meets_threshold(0.5, threshold=0.6))

    def test_bad_threshold(self):
        with self.assertRaises(ValueError):
            meets_threshold(0.5, threshold=2.0)


class TestAggregate(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(aggregate_vqascore([0.2, 0.4, 0.6]), 0.4)

    def test_empty(self):
        with self.assertRaises(ValueError):
            aggregate_vqascore([])


class TestBestCandidate(unittest.TestCase):
    def test_pick_max(self):
        self.assertEqual(best_candidate([0.1, 0.9, 0.5]), (1, 0.9))

    def test_ties_earliest(self):
        self.assertEqual(best_candidate([0.7, 0.7, 0.3]), (0, 0.7))

    def test_empty(self):
        with self.assertRaises(ValueError):
            best_candidate([])


class TestTrajectory(unittest.TestCase):
    def test_stops_at_first_over_threshold(self):
        r = stopping_trajectory([0.4, 0.7, 0.92, 0.99])
        self.assertTrue(r["stopped"])
        self.assertEqual(r["stop_index"], 2)
        self.assertEqual(r["rounds"], 2)
        self.assertEqual(r["final_score"], 0.92)

    def test_direct_generation_already_passes(self):
        r = stopping_trajectory([0.95])
        self.assertTrue(r["stopped"])
        self.assertEqual(r["rounds"], 0)

    def test_never_stops(self):
        r = stopping_trajectory([0.1, 0.3, 0.5, 0.7])
        self.assertFalse(r["stopped"])
        self.assertIsNone(r["stop_index"])
        self.assertEqual(r["rounds"], 3)
        self.assertEqual(r["final_score"], 0.7)

    def test_custom_threshold(self):
        r = stopping_trajectory([0.6, 0.7], threshold=0.65)
        self.assertEqual(r["stop_index"], 1)

    def test_empty(self):
        with self.assertRaises(ValueError):
            stopping_trajectory([])


if __name__ == "__main__":
    unittest.main()
