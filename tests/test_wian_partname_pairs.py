import unittest

from harnesscad.eval.bench.data.partname_pairs import (
    NamePair,
    best_threshold,
    build_pairs,
    cooccurrence_table,
    evaluate_pairs,
    negative_pairs,
    pair_accuracy,
    pairs_to_csv,
    positive_pairs,
    roc_auc,
)

CORPUS = {
    "d1": {"body_names": ["wheel hub", "axle", "Part 1", "left wheel"]},
    "d2": {"body_names": ["motor mount", "propeller"]},
    "d3": {"body_names": ["battery tray", "lid"]},
    "d4": {"body_names": ["bracket"]},
    "d5": {"body_names": ["front panel", "rear panel"]},
}
IDS = ["d1", "d2", "d3", "d4", "d5"]


class TestCooccurrence(unittest.TestCase):
    def test_table_excludes_defaults_and_self(self):
        table = cooccurrence_table(CORPUS)
        self.assertNotIn("Part 1", table)
        self.assertEqual(table["axle"], {"wheel hub", "left wheel"})
        self.assertNotIn("axle", table["axle"])
        self.assertEqual(table["bracket"], set())

    def test_table_deterministic(self):
        self.assertEqual(cooccurrence_table(CORPUS), cooccurrence_table(CORPUS))


class TestPositivePairs(unittest.TestCase):
    def test_token_disjoint_and_one_per_document(self):
        pos = positive_pairs(CORPUS, IDS, seed=7)
        self.assertLessEqual(len(pos), 4)  # d4 has only one part
        for a, b in pos:
            self.assertTrue(
                set(a.lower().split()).isdisjoint(set(b.lower().split())), (a, b)
            )

    def test_no_pair_when_all_names_share_tokens(self):
        corpus = {"x": {"body_names": ["front panel", "rear panel"]}}
        # "panel" is shared, so no token-disjoint pair exists.
        self.assertEqual(positive_pairs(corpus, ["x"], seed=1), [])

    def test_single_part_document_skipped(self):
        self.assertEqual(positive_pairs(CORPUS, ["d4"], seed=1), [])

    def test_seed_determinism(self):
        self.assertEqual(
            positive_pairs(CORPUS, IDS, seed=3), positive_pairs(CORPUS, IDS, seed=3)
        )


class TestNegativePairs(unittest.TestCase):
    def test_negatives_never_cooccur(self):
        table = cooccurrence_table(CORPUS)
        pos = positive_pairs(CORPUS, IDS, seed=11)
        neg = negative_pairs(pos, table, seed=11)
        for a, b in neg:
            self.assertNotIn(b, table.get(a, set()))
            self.assertNotEqual(a, b)

    def test_empty_positives(self):
        self.assertEqual(negative_pairs([], {}, seed=1), [])


class TestBuildPairs(unittest.TestCase):
    def test_balanced_and_ordered(self):
        rows = build_pairs(CORPUS, IDS, seed=5)
        labels = [r.label for r in rows]
        self.assertEqual(labels.count(1), labels.count(0))
        # all positives first
        self.assertEqual(labels, sorted(labels, reverse=True))

    def test_deterministic(self):
        self.assertEqual(build_pairs(CORPUS, IDS, seed=5), build_pairs(CORPUS, IDS, seed=5))

    def test_csv_format(self):
        csv = pairs_to_csv([NamePair(1, "a b", 'c"d')])
        self.assertEqual(csv, '1,"a b","c""d"\n')
        self.assertEqual(pairs_to_csv([]), "")


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.4, 0.2, 0.6]
        self.assertAlmostEqual(pair_accuracy(labels, scores, threshold=0.5), 0.5)
        self.assertAlmostEqual(pair_accuracy(labels, scores, threshold=0.3), 0.75)

    def test_accuracy_length_mismatch(self):
        with self.assertRaises(ValueError):
            pair_accuracy([1], [0.1, 0.2])

    def test_best_threshold(self):
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.1, 0.2]
        cut, acc = best_threshold(labels, scores)
        self.assertAlmostEqual(acc, 1.0)
        self.assertLessEqual(cut, 0.8)

    def test_roc_auc_perfect_and_inverted(self):
        labels = [1, 1, 0, 0]
        self.assertAlmostEqual(roc_auc(labels, [0.9, 0.8, 0.1, 0.2]), 1.0)
        self.assertAlmostEqual(roc_auc(labels, [0.1, 0.2, 0.9, 0.8]), 0.0)

    def test_roc_auc_ties(self):
        self.assertAlmostEqual(roc_auc([1, 0], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(roc_auc([1, 1], [0.5, 0.9]), 0.5)  # degenerate

    def test_evaluate_pairs(self):
        pairs = [NamePair(1, "a", "b"), NamePair(0, "a", "c")]
        report = evaluate_pairs(pairs, [0.9, 0.1])
        self.assertEqual(report["n"], 2.0)
        self.assertEqual(report["positives"], 1.0)
        self.assertAlmostEqual(report["accuracy"], 1.0)
        self.assertAlmostEqual(report["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
