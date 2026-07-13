"""Tests for editing.cadmorph_verify (CADMorph verification + priority queue)."""
import unittest

from harnesscad.domain.editing.candidate_verify import (
    CandidateQueue, ScoredCandidate, edit_distance, objective,
    score_candidates, select_best,
)


class EditDistanceTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(edit_distance(("a", "b", "c"), ("a", "b", "c")), 0)

    def test_single_substitution(self):
        self.assertEqual(edit_distance((1, 2, 3), (1, 9, 3)), 1)

    def test_insert_delete(self):
        self.assertEqual(edit_distance((1, 2), (1, 2, 3)), 1)
        self.assertEqual(edit_distance((1, 2, 3), (1, 3)), 1)

    def test_empty(self):
        self.assertEqual(edit_distance((), (1, 2)), 2)
        self.assertEqual(edit_distance((1, 2), ()), 2)


class ObjectiveTests(unittest.TestCase):
    def test_lambda_zero_is_geometry_only(self):
        self.assertEqual(objective(3.0, (1, 9, 3), (1, 2, 3), lam=0.0), 3.0)

    def test_structure_penalty_added(self):
        # geom 3 + lam 2 * editdist(1) = 5.
        self.assertEqual(objective(3.0, (1, 9, 3), (1, 2, 3), lam=2.0), 5.0)


class PriorityQueueTests(unittest.TestCase):
    def _sc(self, cand, dist, rnd=0, order=0):
        return ScoredCandidate(cand, dist, dist, rnd, order)

    def test_keeps_best_of_capacity(self):
        q = CandidateQueue(capacity=2)
        q.push(self._sc("a", 5.0))
        q.push(self._sc("b", 1.0))
        q.push(self._sc("c", 3.0))
        items = [sc.candidate for sc in q.items()]
        self.assertEqual(items, ["b", "c"])  # 5.0 evicted
        self.assertEqual(q.best().candidate, "b")

    def test_capacity_must_be_positive(self):
        with self.assertRaises(ValueError):
            CandidateQueue(capacity=0)

    def test_deterministic_tiebreak(self):
        q = CandidateQueue(capacity=3)
        q.push(self._sc("late", 1.0, rnd=2, order=0))
        q.push(self._sc("early", 1.0, rnd=0, order=1))
        # Equal score -> earlier round wins.
        self.assertEqual(q.best().candidate, "early")

    def test_cross_round_retention(self):
        # A good candidate from round 0 survives a bad round 1 (paper ablation B).
        q = CandidateQueue(capacity=2)
        select_best(["good"], lambda c: 1.0, original=("x",),
                    queue=q, round_index=0)
        best_round1 = select_best(["bad"], lambda c: 9.0, original=("x",),
                                  queue=q, round_index=1)
        self.assertEqual(best_round1.candidate, "good")


class SelectBestTests(unittest.TestCase):
    def test_picks_min_distance(self):
        best = select_best(["a", "b", "c"],
                           distance=lambda c: {"a": 3.0, "b": 1.0, "c": 2.0}[c],
                           original=("a",))
        self.assertEqual(best.candidate, "b")

    def test_structure_preservation_breaks_distance_tie(self):
        # Two candidates with equal geometric distance; lambda favours the one
        # closer to the original sequence (fewer edits) -> "smallest edit".
        original = (1, 2, 3)
        near = (1, 2, 4)   # edit distance 1
        far = (7, 8, 9)    # edit distance 3
        best = select_best([far, near], distance=lambda c: 2.0,
                           original=original, lam=1.0)
        self.assertEqual(best.candidate, near)

    def test_empty_without_queue_raises(self):
        with self.assertRaises(ValueError):
            select_best([], distance=lambda c: 0.0, original=())

    def test_scores_carry_distance_and_objective(self):
        scored = score_candidates([(1, 9, 3)], distance=lambda c: 4.0,
                                  original=(1, 2, 3), lam=1.0)
        self.assertEqual(scored[0].geom_distance, 4.0)
        self.assertEqual(scored[0].score, 5.0)  # 4 + 1*1


if __name__ == "__main__":
    unittest.main()
