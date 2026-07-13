import unittest

from harnesscad.data.dataengine.reward.cadvf_visual_score import Component
from harnesscad.data.dataengine.preference.cadvf_preference_pipeline import (
    Candidate, build_preference_dataset, build_prompt_pair,
)


def box(cx, cy, cz, sx=1.0, sy=1.0, sz=1.0):
    return Component(lo=(cx - sx / 2, cy - sy / 2, cz - sz / 2),
                     hi=(cx + sx / 2, cy + sy / 2, cz + sz / 2))


def good(seq, n=2):
    """A well-formed, well-spaced object matching an expected count of n."""
    comps = [box(3 * i, 0, 0) for i in range(n)]
    return Candidate(sequence=seq, renderable=True, components=comps)


def clustered(seq, n=2):
    """A colliding object (all boxes on top of each other) -> low distribution."""
    comps = [box(0, 0, 0) for _ in range(n)]
    return Candidate(sequence=seq, renderable=True, components=comps)


class TestCandidate(unittest.TestCase):
    def test_components_coerced_to_tuple(self):
        c = Candidate("s", True, [box(0, 0, 0)])
        self.assertIsInstance(c.components, tuple)


class TestBuildPromptPair(unittest.TestCase):
    def test_best_vs_worst(self):
        cands = [good("A", 2), clustered("B", 2)]
        res = build_prompt_pair("p", 2, cands)
        self.assertIsNotNone(res.pair)
        ow, ol = res.pair
        self.assertEqual(ow["sequence"], "A")
        self.assertEqual(ol["sequence"], "B")
        self.assertGreater(ow["score"], ol["score"])

    def test_invalid_counted_and_dropped(self):
        cands = [good("A", 2), Candidate("bad", False)]
        res = build_prompt_pair("p", 2, cands)
        self.assertEqual(res.n_total, 2)
        self.assertEqual(res.n_invalid, 1)
        self.assertAlmostEqual(res.invalidity_ratio, 0.5)

    def test_low_quality_filtered(self):
        # An empty renderable object grades 0 -> below floor, dropped.
        empty = Candidate("empty", True, [])
        cands = [good("A", 2), empty]
        res = build_prompt_pair("p", 2, cands, quality_floor=1.0)
        self.assertEqual(res.n_low_quality, 1)
        # only one survivor -> no pair
        self.assertIsNone(res.pair)

    def test_no_pair_when_all_tie(self):
        cands = [good("A", 2), good("B", 2)]
        res = build_prompt_pair("p", 2, cands)
        self.assertIsNone(res.pair)  # equal scores, no preference signal

    def test_deterministic(self):
        cands = [good("A", 2), clustered("B", 2), good("C", 2)]
        a = build_prompt_pair("p", 2, cands)
        b = build_prompt_pair("p", 2, cands)
        self.assertEqual(a.pair, b.pair)

    def test_single_candidate_no_pair(self):
        res = build_prompt_pair("p", 2, [good("A", 2)])
        self.assertIsNone(res.pair)

    def test_invalidity_ratio_zero_when_empty(self):
        res = build_prompt_pair("p", 2, [])
        self.assertEqual(res.invalidity_ratio, 0.0)


class TestBuildPreferenceDataset(unittest.TestCase):
    def test_aggregate(self):
        batches = [
            ("p1", 2, [good("A", 2), clustered("B", 2)]),
            ("p2", 2, [good("C", 2), Candidate("bad", False)]),
        ]
        out = build_preference_dataset(batches)
        # p1 yields a pair; p2 has only one valid survivor -> no pair.
        self.assertEqual(out["counts"]["pairs"], 1)
        self.assertEqual(len(out["pairs"]), 1)
        self.assertEqual(out["pairs"][0]["prompt"], "p1")
        self.assertEqual(out["pairs"][0]["chosen"], "A")
        self.assertEqual(out["pairs"][0]["rejected"], "B")

    def test_invalidity_ratio_aggregated(self):
        batches = [
            ("p1", 2, [good("A", 2), Candidate("x", False)]),
            ("p2", 2, [Candidate("y", False), Candidate("z", False)]),
        ]
        out = build_preference_dataset(batches)
        # 3 invalid out of 4 sampled.
        self.assertAlmostEqual(out["invalidity_ratio"], 0.75)
        self.assertEqual(out["counts"]["sampled"], 4)
        self.assertEqual(out["counts"]["invalid"], 3)

    def test_chosen_score_geq_rejected(self):
        batches = [("p1", 2, [good("A", 2), clustered("B", 2)])]
        out = build_preference_dataset(batches)
        row = out["pairs"][0]
        self.assertGreater(row["chosen_score"], row["rejected_score"])

    def test_empty_batches(self):
        out = build_preference_dataset([])
        self.assertEqual(out["pairs"], [])
        self.assertEqual(out["invalidity_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
