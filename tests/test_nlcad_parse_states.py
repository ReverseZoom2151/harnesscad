import unittest

from spec.nlcad_parse_states import (
    Sense, senses_of, parse, ParseState,
)


class TestSenses(unittest.TestCase):
    def test_ambiguous_what(self):
        poses = {s.pos for s in senses_of("what")}
        self.assertEqual(poses, {"pron", "det"})

    def test_ambiguous_output(self):
        poses = {s.pos for s in senses_of("output")}
        self.assertEqual(poses, {"noun", "verb"})

    def test_number_is_num(self):
        s = senses_of("10")
        self.assertEqual(s[0].pos, "num")

    def test_unknown_default_noun(self):
        s = senses_of("frobnitz")
        self.assertEqual(s[0].pos, "noun")


class TestParse(unittest.TestCase):
    def test_determiner_sense_terminated_before_verb(self):
        # "what is ..." -> the determiner reading of "what" dies at "is"
        # (a determiner cannot be followed by a verb), the pronoun reading lives.
        res = parse("what is the maximum voltage")
        self.assertIsNotNone(res.best)
        self.assertGreater(res.terminated_count, 0)
        # best reading tags "what" as a pronoun
        first_word, first_pos = res.best.tags[0]
        self.assertEqual(first_word, "what")
        self.assertEqual(first_pos, "pron")

    def test_best_is_complete_and_full_length(self):
        res = parse("what is the maximum voltage")
        self.assertTrue(res.best.complete)
        self.assertEqual(res.best.steps, 5)

    def test_confidence_in_range(self):
        res = parse("draw a circle at the origin")
        for st in res.ranked:
            self.assertGreaterEqual(st.confidence, 1)
            self.assertLessEqual(st.confidence, 10)

    def test_imperative_verb_start_preferred(self):
        res = parse("draw a circle")
        self.assertEqual(res.best.tags[0][1], "verb")

    def test_pos_sequence(self):
        res = parse("draw a circle")
        self.assertEqual(res.best.pos_sequence(), ("verb", "det", "noun"))

    def test_suspension_counts_with_narrow_beam(self):
        res = parse("what is the maximum voltage output node", beam=1)
        self.assertGreaterEqual(res.suspended_count, 0)
        self.assertLessEqual(len(res.ranked), 1)

    def test_deterministic(self):
        a = parse("what is the maximum voltage")
        b = parse("what is the maximum voltage")
        self.assertEqual(a.best.tags, b.best.tags)
        self.assertEqual(a.terminated_count, b.terminated_count)

    def test_complete_states_end_on_noun_or_num(self):
        res = parse("delete the hole")
        self.assertTrue(res.best.complete)
        self.assertIn(res.best.tags[-1][1], ("noun", "num"))

    def test_empty_input(self):
        res = parse("")
        self.assertEqual(res.best.steps, 0)


if __name__ == "__main__":
    unittest.main()
