"""Tests for the CAD grammar FSA, primitive pooling, and tokenizer benchmarks.

Rewritten from bare pytest-style module functions (never collected by
``python -m unittest``) into unittest.TestCase classes.
"""

import unittest

from bench.tokenizer_frontier import evaluate, frontier
from bench.tokenizer_split_audit import audit
from grammar_fsa import State, run
from quality.primitive_pooling import spans


_SEQ = ("line", "curve_end", "loop_end", "face_end", "sketch_end", "add", "pad")


class GrammarFSATest(unittest.TestCase):
    def test_well_formed_sequence_reaches_pad_state(self):
        self.assertIs(run(_SEQ)[0], State.PAD)

    def test_orphan_curve_end_kills_the_automaton(self):
        self.assertIs(run(("curve_end",))[0], State.DEAD)


class PrimitivePoolingTest(unittest.TestCase):
    def test_spans_cover_the_sequence_exactly_without_overflow(self):
        _, report = spans(_SEQ)
        self.assertTrue(report["exact_coverage"])
        self.assertFalse(report["overflow"])


class TokenizerFrontierTest(unittest.TestCase):
    def test_non_dominated_candidate_is_on_the_pareto_frontier(self):
        a = evaluate("a", (1, 2), (1,), (1, 2))
        b = evaluate("b", (1, 2), (1, 2), (1, 0))
        self.assertIn(a, frontier((a, b)))


class TokenizerSplitAuditTest(unittest.TestCase):
    def test_nested_split_flags_heldout_exposure(self):
        report = audit({1}, {1, 2})
        self.assertTrue(report["nested"])
        self.assertTrue(report["has_heldout_exposure"])


if __name__ == "__main__":
    unittest.main()
