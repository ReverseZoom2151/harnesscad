import unittest

from harnesscad.eval.bench.judges.engdesign_similarity_triplets import (
    enumerate_triplets, triplet_count, self_consistency, transitive_violations,
)


class TripletEnumerationTest(unittest.TestCase):
    def test_count_for_ten_designs(self):
        # Paper: 10 designs -> 360 anchored triplets.
        self.assertEqual(triplet_count(10), 360)
        self.assertEqual(len(enumerate_triplets(10)), 360)

    def test_each_design_is_anchor_36_times(self):
        trips = enumerate_triplets(10)
        anchors = [t[0] for t in trips]
        for d in range(10):
            self.assertEqual(anchors.count(d), 36)

    def test_small_n(self):
        self.assertEqual(enumerate_triplets(2), ())
        self.assertEqual(enumerate_triplets(3),
                         ((0, 1, 2), (1, 0, 2), (2, 0, 1)))


class SelfConsistencyTest(unittest.TestCase):
    def test_all_consistent(self):
        res = self_consistency({("a",): ("B", "B", "B"), ("b",): ("C", "C")})
        self.assertEqual(res["self_consistency"], 1.0)
        self.assertEqual(res["repeated"], 2)

    def test_half_consistent(self):
        res = self_consistency({("a",): ("B", "C"), ("b",): ("C", "C")})
        self.assertEqual(res["self_consistency"], 0.5)

    def test_single_answer_ignored(self):
        res = self_consistency({("a",): ("B",)})
        self.assertIsNone(res["self_consistency"])
        self.assertEqual(res["repeated"], 0)


class TransitiveViolationTest(unittest.TestCase):
    def test_consistent_metric_no_violation(self):
        # Similarity from a real 1-D embedding is always transitive.
        pos = {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}

        def chooser(anchor, x, y):
            return x if abs(pos[anchor] - pos[x]) < abs(pos[anchor] - pos[y]) else y

        res = transitive_violations(4, chooser)
        self.assertEqual(res["transitive_violations"], 0)

    def test_cyclic_triple_is_a_violation(self):
        # Force a strict cycle on triple {0,1,2}: 0->1, 1->2, 2->0.
        # anchor 0 picks 1 (dAB<dAC); anchor 1 picks 2 (dBC<dAB);
        # anchor 2 picks 0 (dAC<dBC) -> cycle.
        def chooser(anchor, x, y):
            table = {(0, 1, 2): 1, (1, 0, 2): 2, (2, 0, 1): 0}
            return table[(anchor, x, y)]

        res = transitive_violations(3, chooser)
        self.assertEqual(res["transitive_violations"], 1)
        self.assertEqual(res["violating_triples"], ((0, 1, 2),))

    def test_bad_chooser_raises(self):
        def chooser(anchor, x, y):
            return 99

        with self.assertRaises(ValueError):
            transitive_violations(3, chooser)


if __name__ == "__main__":
    unittest.main()
