import unittest

from drawings.sympoint_query_grouping import (
    MASK_THRESHOLD,
    OBJECT_SCORE,
    OVERLAP_THRESHOLD,
    instance_inference,
    semantic_inference,
    semantic_labels,
    sigmoid,
    softmax,
)

BIG = 12.0   # sigmoid ~ 1
NEG = -12.0  # sigmoid ~ 0


class TestPrimitives(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(BIG), 0.999)
        self.assertLess(sigmoid(NEG), 0.001)

    def test_sigmoid_stable_for_large_negative(self):
        self.assertAlmostEqual(sigmoid(-1000.0), 0.0, places=9)

    def test_softmax_sums_to_one(self):
        p = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertGreater(p[2], p[1])

    def test_softmax_shift_invariant(self):
        a = softmax([1.0, 2.0])
        b = softmax([101.0, 102.0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_softmax_empty(self):
        with self.assertRaises(ValueError):
            softmax([])


class TestSemanticInference(unittest.TestCase):
    def test_mixture_of_queries(self):
        # 2 queries, 3 columns (2 real classes + no-object), 2 points
        cls = [[BIG, NEG, NEG], [NEG, BIG, NEG]]
        masks = [[BIG, NEG], [NEG, BIG]]
        semseg = semantic_inference(cls, masks)
        self.assertEqual(len(semseg), 2)
        self.assertEqual(len(semseg[0]), 2)
        self.assertEqual(semantic_labels(semseg), [0, 1])

    def test_no_object_column_dropped(self):
        cls = [[NEG, NEG, BIG]]
        masks = [[BIG]]
        semseg = semantic_inference(cls, masks)
        self.assertEqual(len(semseg[0]), 2)
        self.assertLess(sum(semseg[0]), 0.01)

    def test_ragged_inputs(self):
        with self.assertRaises(ValueError):
            semantic_inference([[1.0, 2.0], [1.0]], [[0.0], [0.0]])
        with self.assertRaises(ValueError):
            semantic_inference([[1.0, 2.0]], [[0.0], [0.0]])

    def test_needs_real_class(self):
        with self.assertRaises(ValueError):
            semantic_inference([[1.0]], [[0.0]])

    def test_labels_tie_lowest_class(self):
        self.assertEqual(semantic_labels([[0.5, 0.5, 0.5]]), [0])


class TestInstanceInference(unittest.TestCase):
    def test_two_disjoint_symbols(self):
        cls = [[BIG, NEG, NEG], [NEG, BIG, NEG]]
        masks = [[BIG, BIG, NEG, NEG], [NEG, NEG, BIG, BIG]]
        out = instance_inference(cls, masks)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["label"], 0)
        self.assertEqual(out[0]["points"], [0, 1])
        self.assertEqual(out[1]["points"], [2, 3])
        self.assertGreater(out[0]["score"], 0.99)

    def test_no_object_query_dropped(self):
        cls = [[NEG, NEG, BIG]]
        masks = [[BIG, BIG]]
        self.assertEqual(instance_inference(cls, masks), [])

    def test_low_score_query_dropped(self):
        # near-uniform class distribution over 20 classes -> max prob < 0.1
        cls = [[0.0] * 20 + [-50.0]]
        masks = [[BIG, BIG]]
        self.assertEqual(instance_inference(cls, masks), [])

    def test_winner_takes_all_makes_points_disjoint(self):
        # both queries claim all points; query 0 is more confident
        cls = [[BIG, NEG, NEG], [1.0, 0.9, NEG]]
        masks = [[BIG, BIG], [BIG, BIG]]
        out = instance_inference(cls, masks)
        owned = [p for inst in out for p in inst["points"]]
        self.assertEqual(sorted(owned), sorted(set(owned)))

    def test_fragment_query_dropped_by_overlap_rule(self):
        # query 1 claims both points but loses both to query 0 -> 0/2 < 0.8
        cls = [[BIG, NEG, NEG], [NEG, 1.0, NEG]]
        masks = [[BIG, BIG], [BIG, BIG]]
        out = instance_inference(cls, masks)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["label"], 0)
        self.assertEqual(out[0]["points"], [0, 1])

    def test_overlap_threshold_relaxable(self):
        cls = [[BIG, NEG, NEG], [NEG, 5.0, NEG]]
        masks = [[BIG, BIG, BIG], [NEG, BIG, BIG]]
        strict = instance_inference(cls, masks, overlap_threshold=1.0)
        loose = instance_inference(cls, masks, overlap_threshold=0.0)
        self.assertLessEqual(len(strict), len(loose))

    def test_mask_threshold_filters_points(self):
        cls = [[BIG, NEG, NEG]]
        masks = [[BIG, NEG]]
        out = instance_inference(cls, masks)
        self.assertEqual(out[0]["points"], [0])

    def test_empty_inputs(self):
        self.assertEqual(instance_inference([], []), [])
        self.assertEqual(instance_inference([[BIG, NEG]], [[]]), [])

    def test_constants(self):
        self.assertEqual(OBJECT_SCORE, 0.1)
        self.assertEqual(OVERLAP_THRESHOLD, 0.8)
        self.assertEqual(MASK_THRESHOLD, 0.5)

    def test_deterministic(self):
        cls = [[BIG, NEG, NEG], [NEG, BIG, NEG]]
        masks = [[BIG, NEG], [NEG, BIG]]
        self.assertEqual(instance_inference(cls, masks), instance_inference(cls, masks))


if __name__ == "__main__":
    unittest.main()
