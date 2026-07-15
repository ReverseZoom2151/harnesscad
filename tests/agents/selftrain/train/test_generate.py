"""Candidate generation has two disciplines that are not negotiable and are pure
python, so they are tested with no network and no GPU: reasoning models are refused
(their think-budget behaviour would be miscounted), and a ``<think>`` block is
stripped before the op parser ever sees it. Also: generation must never touch an
eval brief."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain.train import generate as G
from harnesscad.agents.selftrain.train import evaluate as E


class TestReasoningRefusal(unittest.TestCase):

    def test_denylist(self):
        for m in ("deepseek-r1:32b", "magistral:24b", "qwen3:30b"):
            self.assertTrue(G.is_reasoning_model(m), m)
        for m in ("qwen2.5-coder:7b", "qwen2.5-coder:32b", "codellama:7b"):
            self.assertFalse(G.is_reasoning_model(m), m)

    def test_sample_refuses_reasoning_model(self):
        with self.assertRaises(ValueError):
            G.sample_and_certify("deepseek-r1:32b")


class TestStripThink(unittest.TestCase):

    def test_removes_block(self):
        raw = "<think>I will reason a lot</think>\n[{\"op\":\"x\"}]"
        self.assertEqual(G.strip_think(raw).strip(), '[{"op":"x"}]')

    def test_multiline_and_case(self):
        raw = "<THINK>\nline1\nline2\n</THINK>[]"
        self.assertEqual(G.strip_think(raw).strip(), "[]")

    def test_noop_when_absent(self):
        self.assertEqual(G.strip_think("[]"), "[]")


class TestBriefIsolation(unittest.TestCase):

    def test_train_and_eval_briefs_are_disjoint(self):
        # The one contamination that would void the whole result: generating on a
        # brief the model is later evaluated on.
        train = set(G.TRAIN_BRIEFS)
        held = set(E.HELDOUT_BRIEFS)
        self.assertEqual(train & held, set())
        self.assertEqual(len(train), 12)
        self.assertEqual(len(held), 16)

    def test_together_they_are_the_whole_pressure_corpus(self):
        from harnesscad.eval.pressure import briefs as B
        allids = {x.id for x in B.BRIEFS}
        self.assertEqual(set(G.TRAIN_BRIEFS) | set(E.HELDOUT_BRIEFS), allids)


if __name__ == "__main__":
    unittest.main()
