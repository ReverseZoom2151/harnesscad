"""Underspecification: asking is scored right, hallucinating a number is scored wrong.

The classifier is the whole contribution here, so it is pinned on both controls:
a response that asks about the missing dimension must score ASKED; an op stream (or
a bare committed value) must score HALLUCINATED. If those two ever swap, the module
would reward the confident guesser -- the exact failure it exists to punish.
"""

from __future__ import annotations

import unittest

from harnesscad.core.cisp.ops import AddRectangle, Extrude, NewSketch
from harnesscad.eval.hardcorpus import ambiguous as amb


class TestAmbiguous(unittest.TestCase):

    def setUp(self):
        self.brief = amb.BRIEFS[0]        # plate with no thickness

    def test_structured_ask_is_asked(self):
        r = amb.Clarify("How thick should it be?", ("thickness",))
        self.assertEqual(amb.classify(self.brief, r), amb.Verdict.ASKED)

    def test_free_text_ask_is_asked(self):
        self.assertEqual(
            amb.classify(self.brief, "What thickness do you want for the plate?"),
            amb.Verdict.ASKED)

    def test_op_stream_is_hallucinated(self):
        ops = [NewSketch("XY"), AddRectangle("sk1", 0, 0, 80, 40), Extrude("sk1", 10)]
        self.assertEqual(amb.classify(self.brief, ops), amb.Verdict.HALLUCINATED)

    def test_committed_value_is_hallucinated(self):
        self.assertEqual(
            amb.classify(self.brief, "I'll use 10 mm thickness."),
            amb.Verdict.HALLUCINATED)

    def test_hedge_is_hedged(self):
        self.assertEqual(
            amb.classify(self.brief,
                         "I'll assume 8 mm thick unless you'd prefer otherwise?"),
            amb.Verdict.HEDGED)

    def test_asking_the_wrong_thing(self):
        self.assertEqual(
            amb.classify(self.brief, "What length would you like?"),
            amb.Verdict.ASKED_WRONG)

    def test_score_counts_and_rates(self):
        # all-ask -> ask_rate 1.0; all-hallucinate -> hallucination_rate 1.0.
        ask = amb.score([(b, amb.Clarify("?", b.missing)) for b in amb.BRIEFS])
        self.assertEqual(ask.ask_rate, 1.0)
        halluc = amb.score([(b, "I'll assume a default and proceed.")
                            for b in amb.BRIEFS])
        self.assertEqual(halluc.hallucination_rate, 1.0)

    def test_every_brief_removed_a_real_dimension(self):
        for b in amb.BRIEFS:
            self.assertTrue(b.missing)
            self.assertTrue(b.keywords)
            # the full text says more than the shown text (a dimension was removed).
            self.assertGreater(len(b.full_text), len(b.text))

    def test_caveats_are_stated(self):
        self.assertTrue(amb.CLASSIFIER_CAVEATS)


if __name__ == "__main__":
    unittest.main()
