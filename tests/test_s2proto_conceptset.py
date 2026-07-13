import unittest

from harnesscad.eval.bench.generative.s2proto_conceptset import (
    mean_pairwise_similarity, set_diversity, effective_concept_count,
    cross_modal_alignment, percentile_rank, percentile_value,
    rank_methods_by_diversity, conceptset_report,
)


def _id(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


class TestMeanPairwise(unittest.TestCase):
    def test_identical_set_high_similarity(self):
        m = [[1.0, 1.0], [1.0, 1.0]]
        self.assertEqual(mean_pairwise_similarity(m), 1.0)

    def test_orthogonal_set_zero(self):
        self.assertEqual(mean_pairwise_similarity(_id(4)), 0.0)

    def test_average_offdiagonal(self):
        m = [[1.0, 0.2, 0.4], [0.2, 1.0, 0.6], [0.4, 0.6, 1.0]]
        self.assertAlmostEqual(mean_pairwise_similarity(m), (0.2 + 0.4 + 0.6) / 3)

    def test_single_item_zero(self):
        self.assertEqual(mean_pairwise_similarity([[1.0]]), 0.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            mean_pairwise_similarity([])

    def test_non_square_raises(self):
        with self.assertRaises(ValueError):
            mean_pairwise_similarity([[1.0, 0.0]])

    def test_asymmetric_raises(self):
        with self.assertRaises(ValueError):
            mean_pairwise_similarity([[1.0, 0.2], [0.9, 1.0]])


class TestSetDiversity(unittest.TestCase):
    def test_inverse_of_similarity(self):
        m = [[1.0, 0.8], [0.8, 1.0]]
        self.assertAlmostEqual(set_diversity(m), 0.2)

    def test_diverse_higher_than_similar(self):
        similar = [[1.0, 0.9], [0.9, 1.0]]
        diverse = [[1.0, 0.1], [0.1, 1.0]]
        self.assertGreater(set_diversity(diverse), set_diversity(similar))

    def test_custom_max(self):
        # CLIP-style scores on a 0..100 scale
        m = [[100.0, 64.4], [64.4, 100.0]]
        self.assertAlmostEqual(set_diversity(m, max_similarity=100.0), 35.6)


class TestEffectiveCount(unittest.TestCase):
    def test_all_distinct(self):
        self.assertEqual(effective_concept_count(_id(4), threshold=0.5), 4)

    def test_all_duplicates_collapse(self):
        m = [[1.0, 0.99, 0.98], [0.99, 1.0, 0.97], [0.98, 0.97, 1.0]]
        self.assertEqual(effective_concept_count(m, threshold=0.5), 1)

    def test_partial(self):
        # 0 and 1 near-identical; 2 distinct from both
        m = [[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]]
        self.assertEqual(effective_concept_count(m, threshold=0.5), 2)


class TestCrossModal(unittest.TestCase):
    def test_average(self):
        cross = [[25.0, 27.0], [24.0, 28.0]]
        self.assertAlmostEqual(cross_modal_alignment(cross), (25 + 27 + 24 + 28) / 4)

    def test_rectangular_ok(self):
        cross = [[1.0, 2.0, 3.0]]
        self.assertAlmostEqual(cross_modal_alignment(cross), 2.0)

    def test_ragged_raises(self):
        with self.assertRaises(ValueError):
            cross_modal_alignment([[1.0, 2.0], [3.0]])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            cross_modal_alignment([])


class TestPercentiles(unittest.TestCase):
    def test_percentile_rank_bounds(self):
        dist = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile_rank(4.0, dist), 1.0)
        self.assertEqual(percentile_rank(0.0, dist), 0.0)
        self.assertEqual(percentile_rank(2.0, dist), 0.5)

    def test_percentile_rank_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile_rank(1.0, [])

    def test_percentile_value_extremes(self):
        dist = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(percentile_value(dist, 0), 10.0)
        self.assertEqual(percentile_value(dist, 100), 50.0)
        self.assertEqual(percentile_value(dist, 50), 30.0)

    def test_percentile_value_out_of_range(self):
        with self.assertRaises(ValueError):
            percentile_value([1.0], 150)


class TestRankMethods(unittest.TestCase):
    def test_s2proto_beats_baselines(self):
        methods = {
            "sketch_alone": [[1.0, 0.7], [0.7, 1.0]],
            "controlnet": [[1.0, 0.85], [0.85, 1.0]],
            "sketch2prototype": [[1.0, 0.3], [0.3, 1.0]],
        }
        ranking = rank_methods_by_diversity(methods)
        self.assertEqual(ranking[0][0], "sketch2prototype")
        self.assertEqual(ranking[-1][0], "controlnet")

    def test_tie_broken_by_name(self):
        methods = {
            "b": [[1.0, 0.5], [0.5, 1.0]],
            "a": [[1.0, 0.5], [0.5, 1.0]],
        }
        ranking = rank_methods_by_diversity(methods)
        self.assertEqual([n for n, _ in ranking], ["a", "b"])


class TestReport(unittest.TestCase):
    def test_report_keys(self):
        r = conceptset_report(_id(3), threshold=0.5)
        self.assertEqual(r["n"], 3)
        self.assertEqual(r["mean_pairwise_similarity"], 0.0)
        self.assertEqual(r["set_diversity"], 1.0)
        self.assertEqual(r["effective_concept_count"], 3)

    def test_report_without_threshold(self):
        r = conceptset_report(_id(2))
        self.assertNotIn("effective_concept_count", r)


if __name__ == "__main__":
    unittest.main()
