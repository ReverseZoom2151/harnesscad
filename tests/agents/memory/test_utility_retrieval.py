"""Tests for agents.memory.utility_retrieval."""

import unittest

from harnesscad.agents.memory.utility_retrieval import (
    DualTrackMemory,
    MemoryEntry,
    jaccard_similarity,
    retrieval_score,
)


class SimilarityTest(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(jaccard_similarity("a b c", "a b c"), 1.0)

    def test_disjoint(self):
        self.assertEqual(jaccard_similarity("a b", "c d"), 0.0)

    def test_partial(self):
        self.assertAlmostEqual(jaccard_similarity("a b", "b c"), 1 / 3)


class EntryTest(unittest.TestCase):
    def test_update_moves_toward_reward(self):
        e = MemoryEntry("k", "case")
        e.update(1.0, lr=0.5)
        self.assertAlmostEqual(e.utility, 0.5)
        e.update(1.0, lr=0.5)
        self.assertAlmostEqual(e.utility, 0.75)
        self.assertEqual(e.count, 2)

    def test_bad_lr(self):
        with self.assertRaises(ValueError):
            MemoryEntry("k", "case").update(1.0, lr=0.0)


class RetrievalScoreTest(unittest.TestCase):
    def test_weighting(self):
        self.assertAlmostEqual(retrieval_score(1.0, 0.0, 0.7, 0.3), 0.7)
        self.assertAlmostEqual(retrieval_score(0.0, 1.0, 0.7, 0.3), 0.3)


class DualTrackMemoryTest(unittest.TestCase):
    def test_track_separation(self):
        m = DualTrackMemory()
        m.add("draw a slot", "case")
        m.add("slot skill", "skill")
        self.assertEqual(len(m.retrieve("slot", "case")), 1)
        self.assertEqual(len(m.retrieve("slot", "skill")), 1)

    def test_utility_reranks_over_similarity(self):
        # Two cases equally similar to the query; utility should break the tie
        # toward the one with successful execution history (the trap-avoidance).
        m = DualTrackMemory(alpha=0.5, beta=0.5)
        m.add("make a hole", "case")       # semantically similar but infeasible
        m.add("make a slot", "case")       # will accrue positive utility
        m.record_feedback("make a hole", "case", reward=-1.0)
        m.record_feedback("make a slot", "case", reward=1.0)
        top = m.retrieve("make a", "case", top_k=1)
        self.assertEqual(top[0][0].key, "make a slot")

    def test_feedback_unknown_raises(self):
        m = DualTrackMemory()
        with self.assertRaises(KeyError):
            m.record_feedback("nope", "case", 1.0)

    def test_deterministic(self):
        m = DualTrackMemory()
        m.add("a b", "skill")
        m.add("a c", "skill")
        self.assertEqual(
            [e.key for e, _ in m.retrieve("a", "skill")],
            [e.key for e, _ in m.retrieve("a", "skill")],
        )


if __name__ == "__main__":
    unittest.main()
