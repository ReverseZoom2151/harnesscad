"""Tests for eval.bench.designqa_scorers (six DesignQA-family QA graders).

Every non-trivial expected value here is hand-derived in a comment, matching
the module's own selfcheck discipline. Trivial-looking scorers still get their
identity (=1.0) and disjoint (=0.0) endpoints pinned, and each partial case
carries the arithmetic that produces its value.
"""

import math
import unittest

from harnesscad.eval.bench.designqa_scorers import (
    NOANSWER,
    bleu2,
    character_f1,
    extract_yes_no,
    normalize_answer,
    rouge_l,
    rule_number_f1,
    score,
    SCORERS,
    token_f1,
    yesno_accuracy,
)


class TestNormalize(unittest.TestCase):
    def test_lowercase_punct_articles_whitespace(self):
        # lowercase, drop "the", delete comma/period, collapse whitespace.
        self.assertEqual(normalize_answer("The  Quick, Brown fox."),
                         "quick brown fox")

    def test_punctuation_deleted_not_spaced(self):
        # SQuAD deletes punctuation without inserting a space.
        self.assertEqual(normalize_answer("F.1.1"), "f11")

    def test_articles_only_as_whole_words(self):
        # "the" inside "theory" must survive.
        self.assertEqual(normalize_answer("theory"), "theory")


class TestTokenF1(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(token_f1("the quick brown fox",
                                  "the quick brown fox"), 1.0)

    def test_disjoint(self):
        self.assertEqual(token_f1("red green blue", "one two three"), 0.0)

    def test_partial_half(self):
        # article "the" dropped: [red, box] vs [blue, box]; shared {box};
        # P=1/2, R=1/2; F1 = 2*.25/1 = 0.5.
        self.assertAlmostEqual(token_f1("the red box", "the blue box"), 0.5)

    def test_both_empty_is_one(self):
        # both normalise to "" (pure articles) -> agree on emptiness -> 1.0.
        self.assertEqual(token_f1("the a an", "a the"), 1.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(token_f1("the", "brown fox"), 0.0)


class TestRuleNumberF1(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(rule_number_f1("T.1.1, T.2.2",
                                        ["T.1.1", "T.2.2"]), 1.0)

    def test_disjoint(self):
        self.assertEqual(rule_number_f1("A.1", ["Z.9"]), 0.0)

    def test_partial_extra_prediction(self):
        # [t.1.1, t.2.2, v.3] vs [t.1.1, t.2.2]: shared 2; P=2/3, R=1;
        # F1 = 2*(2/3)*1/(2/3+1) = 4/5 = 0.8.
        self.assertAlmostEqual(
            rule_number_f1("T.1.1, T.2.2, V.3", ["T.1.1", "T.2.2"]), 0.8)

    def test_dots_preserved_case_insensitive(self):
        # identifiers compared whole; case folded; whitespace stripped.
        self.assertEqual(rule_number_f1("t.1.1 , T.2.2", ["T.1.1", "t.2.2"]),
                         1.0)


class TestCharacterF1(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(character_f1("widget", "widget"), 1.0)

    def test_disjoint(self):
        self.assertEqual(character_f1("xyz", "qw"), 0.0)

    def test_synonym_max_picks_best(self):
        # "ts" is an exact synonym in "tractive system;ts" -> 1.0.
        self.assertEqual(character_f1("ts", "tractive system;ts"), 1.0)

    def test_partial_two_of_three_chars(self):
        # {c,a,t} vs {c,a,r}: shared 2; P=2/3, R=2/3; F1 = 2/3.
        self.assertAlmostEqual(character_f1("cat", "car"), 2.0 / 3.0)


class TestYesNo(unittest.TestCase):
    def test_extract_first_polar_word(self):
        self.assertEqual(extract_yes_no("No, wait, yes"), "no")

    def test_extract_none_is_sentinel(self):
        self.assertEqual(extract_yes_no("I cannot tell from the image"),
                         NOANSWER)

    def test_accuracy_match(self):
        self.assertEqual(yesno_accuracy("Yes, the bracket is present.",
                                        "yes"), 1.0)

    def test_accuracy_mismatch(self):
        self.assertEqual(yesno_accuracy("no", "yes"), 0.0)

    def test_noanswer_scores_zero_against_both_labels(self):
        # The sentinel is a PREDICTION-side "did not answer" marker. It scores
        # 0 against an answerable "yes" AND an answerable "no". This proves the
        # sentinel behaviour WITHOUT claiming any refusal ground truth exists
        # (DesignQA has none).
        self.assertEqual(yesno_accuracy("I cannot tell", "yes"), 0.0)
        self.assertEqual(yesno_accuracy("I cannot tell", "no"), 0.0)


class TestBleu2(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(bleu2("the quick brown fox",
                               "the quick brown fox"), 1.0)

    def test_disjoint(self):
        self.assertEqual(bleu2("alpha beta gamma", "one two three"), 0.0)

    def test_partial_sqrt_half(self):
        # p1 = 3/4 (a,b,c match, d does not), p2 = 2/3 ((a,b),(b,c) match,
        # (c,d) does not); c=4,r=4 -> BP=1; BLEU-2 = sqrt(3/4*2/3) = sqrt(1/2).
        self.assertAlmostEqual(bleu2("a b c d", "a b c e"), math.sqrt(0.5))

    def test_brevity_penalty(self):
        # pred "a b" vs ref "a b c d": p1=1, p2=1; c=2,r=4;
        # BP = exp(1 - 4/2) = exp(-1); BLEU-2 = exp(-1).
        self.assertAlmostEqual(bleu2("a b", "a b c d"), math.exp(-1.0))

    def test_single_token_has_no_bigram(self):
        self.assertEqual(bleu2("hello", "hello world foo"), 0.0)


class TestRougeL(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(rouge_l("the quick brown fox",
                                 "the quick brown fox"), 1.0)

    def test_disjoint(self):
        self.assertEqual(rouge_l("alpha beta", "one two"), 0.0)

    def test_partial_equal_lengths(self):
        # LCS("a b c e","a b c d") = "a b c" = 3; P=R=3/4; F1 = 0.75.
        self.assertAlmostEqual(rouge_l("a b c e", "a b c d"), 0.75)

    def test_partial_unequal_lengths(self):
        # LCS 3; P=3/3=1, R=3/5=0.6; F1 = 2*1*0.6/1.6 = 0.75.
        self.assertAlmostEqual(rouge_l("a b c", "a b c d e"), 0.75)

    def test_subsequence_not_substring(self):
        # "a x b y c" vs "a b c": LCS = "a b c" = 3 (gaps allowed);
        # P=3/5, R=1; F1 = 2*0.6*1/1.6 = 0.75.
        self.assertAlmostEqual(rouge_l("a x b y c", "a b c"), 0.75)


class TestDispatcher(unittest.TestCase):
    def test_all_six_present(self):
        self.assertEqual(
            set(SCORERS),
            {"retrieval", "compilation", "definition",
             "presence", "bleu2", "rouge_l"})

    def test_routes_correctly(self):
        self.assertAlmostEqual(score("retrieval", "the red box",
                                     "the blue box"), 0.5)
        self.assertAlmostEqual(
            score("compilation", "T.1.1, T.2.2, V.3", ["T.1.1", "T.2.2"]), 0.8)
        self.assertAlmostEqual(score("definition", "cat", "car"), 2.0 / 3.0)
        self.assertEqual(score("presence", "yes", "yes"), 1.0)
        self.assertAlmostEqual(score("bleu2", "a b c d", "a b c e"),
                               math.sqrt(0.5))
        self.assertAlmostEqual(score("rouge_l", "a b c e", "a b c d"), 0.75)

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            score("nonesuch", "a", "b")

    def test_range_bounded(self):
        for name, fn in SCORERS.items():
            if name == "compilation":
                v = fn("a, b, c, d", ["a", "b", "c", "e"])
            else:
                v = fn("a b c d", "a b c e")
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


class TestSelfcheck(unittest.TestCase):
    def test_selfcheck_exits_zero(self):
        from harnesscad.eval.bench.designqa_scorers import main
        self.assertEqual(main(["--selfcheck"]), 0)


if __name__ == "__main__":
    unittest.main()
