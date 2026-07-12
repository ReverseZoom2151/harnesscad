import unittest

from bench.wian_partname_pairs import NamePair, evaluate_pairs
from library.wian_partname_ppmi import (
    build_ppmi,
    cosine,
    nearest_tokens,
    token_cooccurrence,
)

# A small corpus with two clear domains: bicycles and drones.
CORPUS = {
    "b1": {"body_names": ["bicycle frame", "wheel spoke", "brake lever"]},
    "b2": {"body_names": ["bicycle frame", "brake lever", "chain guard"]},
    "b3": {"body_names": ["wheel spoke", "chain guard", "pedal crank"]},
    "b4": {"body_names": ["bicycle frame", "pedal crank", "wheel spoke"]},
    "d1": {"body_names": ["drone arm", "propeller blade", "battery tray"]},
    "d2": {"body_names": ["drone arm", "battery tray", "motor mount"]},
    "d3": {"body_names": ["propeller blade", "motor mount", "landing gear"]},
    "d4": {"body_names": ["drone arm", "landing gear", "propeller blade"]},
}
IDS = sorted(CORPUS)


class TestCosine(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(cosine({"a": 1.0}, {"a": 1.0}), 1.0)
        self.assertAlmostEqual(cosine({"a": 1.0}, {"b": 1.0}), 0.0)
        self.assertAlmostEqual(cosine({}, {"a": 1.0}), 0.0)
        self.assertAlmostEqual(cosine({"a": 1.0, "b": 1.0}, {"a": 1.0}), 1 / (2 ** 0.5))


class TestCooccurrence(unittest.TestCase):
    def test_symmetric_and_self_excluded(self):
        pairs, tokens, total = token_cooccurrence(CORPUS, IDS)
        self.assertGreater(total, 0)
        self.assertEqual(pairs[("bicycle", "wheel")], pairs[("wheel", "bicycle")])
        self.assertEqual(pairs[("wheel", "wheel")], 0)
        self.assertGreater(tokens["bicycle"], 0)

    def test_empty_corpus(self):
        pairs, tokens, total = token_cooccurrence({}, [])
        self.assertEqual(total, 0)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(len(tokens), 0)


class TestPPMIModel(unittest.TestCase):
    def setUp(self):
        self.model = build_ppmi(CORPUS, IDS)

    def test_vocabulary_built(self):
        self.assertIn("bicycle", self.model.vocabulary)
        self.assertIn("drone", self.model.vocabulary)

    def test_ppmi_non_negative(self):
        for vec in self.model.vectors.values():
            for weight in vec.values():
                self.assertGreater(weight, 0.0)

    def test_deterministic(self):
        again = build_ppmi(CORPUS, IDS)
        self.assertEqual(self.model.vectors, again.vectors)

    def test_within_domain_beats_cross_domain(self):
        same = self.model.pair_score("bicycle frame", "pedal crank")
        cross = self.model.pair_score("bicycle frame", "propeller blade")
        self.assertGreater(same, cross)

    def test_unknown_name_scores_zero(self):
        self.assertEqual(self.model.pair_score("zzz qqq", "bicycle frame"), 0.0)
        self.assertEqual(self.model.name_vector("zzz"), {})

    def test_set_vector_and_ranking(self):
        ranked = self.model.rank_candidates(
            ["drone arm", "battery tray"],
            ["pedal crank", "propeller blade", "bicycle frame"],
        )
        self.assertEqual(ranked[0], "propeller blade")
        self.assertEqual(len(ranked), 3)

    def test_set_vector_empty(self):
        self.assertEqual(self.model.set_vector(["zzz"]), {})

    def test_shift_prunes(self):
        shifted = build_ppmi(CORPUS, IDS, shift=2.0)
        total_shifted = sum(len(v) for v in shifted.vectors.values())
        total_plain = sum(len(v) for v in self.model.vectors.values())
        self.assertLess(total_shifted, total_plain)

    def test_min_count_prunes(self):
        pruned = build_ppmi(CORPUS, IDS, min_count=100)
        self.assertEqual(pruned.vectors, {})

    def test_score_pairs(self):
        scores = self.model.score_pairs([("drone arm", "motor mount")])
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0], 0.0)


class TestLexicalControl(unittest.TestCase):
    def setUp(self):
        self.model = build_ppmi(CORPUS, IDS)

    def test_tfidf_zero_on_disjoint_names(self):
        # The benchmark's pairs are token-disjoint by construction, so a
        # lexical scorer has no signal at all -- this is the control.
        self.assertEqual(self.model.tfidf_pair_score("drone arm", "battery tray"), 0.0)

    def test_tfidf_nonzero_on_shared_token(self):
        self.assertGreater(self.model.tfidf_pair_score("drone arm", "drone frame"), 0.0)


class TestOnBenchmark(unittest.TestCase):
    def test_ppmi_separates_positives_from_negatives(self):
        model = build_ppmi(CORPUS, IDS)
        pairs = [
            NamePair(1, "bicycle frame", "pedal crank"),
            NamePair(1, "drone arm", "motor mount"),
            NamePair(0, "bicycle frame", "propeller blade"),
            NamePair(0, "pedal crank", "battery tray"),
        ]
        scores = model.score_pairs([(p.a, p.b) for p in pairs])
        report = evaluate_pairs(pairs, scores)
        self.assertAlmostEqual(report["roc_auc"], 1.0)
        self.assertAlmostEqual(report["best_accuracy"], 1.0)


class TestNearestTokens(unittest.TestCase):
    def test_neighbours(self):
        model = build_ppmi(CORPUS, IDS)
        neighbours = [t for t, _ in nearest_tokens(model, "bicycle", k=4)]
        self.assertNotIn("bicycle", neighbours)
        self.assertTrue(set(neighbours) & {"pedal", "crank", "chain", "guard", "spoke"})

    def test_unknown_token(self):
        model = build_ppmi(CORPUS, IDS)
        self.assertEqual(nearest_tokens(model, "zzz"), [])


if __name__ == "__main__":
    unittest.main()
