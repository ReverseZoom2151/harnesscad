"""Tests for editing.cadmorph_loop (CADMorph plan-generate-verify orchestration)."""
import random
import unittest

from harnesscad.domain.editing.plan_verify_loop import CADMorphLoop
from harnesscad.domain.editing.edit_planning import MASK


# --------------------------------------------------------------------------- #
# A tiny deterministic "CAD world" standing in for the learned P2S / MPP models.
#
#   * a sequence is a tuple of integer segments;
#   * render(seq) = the sequence itself (identity "shape");
#   * distance = sum of squared per-position differences (equal length);
#   * contribution(seq, shape)[i] = (seq[i] - shape[i])^2 -> how much segment i
#     mismatches the shape, a clean stand-in for the cross-attention score.
# --------------------------------------------------------------------------- #
def render(seq):
    return tuple(seq)


def distance(a, b):
    return float(sum((x - y) ** 2 for x, y in zip(a, b)))


def contribution(seq, shape):
    return [float((s - t) ** 2) for s, t in zip(seq, shape)]


def make_generator(pool):
    """An MPP stand-in: fill each <mask> token with a sampled value from pool."""
    def generate(masked, n, rng):
        cands = []
        for _ in range(n):
            cand = []
            for tok in masked:
                if tok is MASK or tok == MASK:
                    cand.append(rng.choice(pool))
                else:
                    cand.append(tok)
            cands.append(tuple(cand))
        return cands
    return generate


class LoopConvergenceTests(unittest.TestCase):
    def test_edits_toward_target(self):
        original = (1, 2, 3)
        target = render((1, 9, 3))
        loop = CADMorphLoop(
            render, distance, contribution,
            make_generator([2, 9, 5]),
            n_candidates=6, max_rounds=10, tol=0.0)
        result = loop.run(original, target, seed=0)
        self.assertEqual(result.sequence, (1, 9, 3))
        self.assertEqual(result.distance, 0.0)
        self.assertTrue(result.converged)

    def test_deterministic_replay(self):
        original = (1, 2, 3)
        target = render((1, 9, 3))
        loop = CADMorphLoop(render, distance, contribution,
                            make_generator([2, 9, 5]), n_candidates=6)
        r1 = loop.run(original, target, seed=7)
        r2 = loop.run(original, target, seed=7)
        self.assertEqual(r1.sequence, r2.sequence)
        self.assertEqual([rd.best_distance for rd in r1.rounds],
                         [rd.best_distance for rd in r2.rounds])

    def test_no_regression_below_start(self):
        # A generator that only ever proposes worse candidates: the queue must
        # still return something no worse than the original (queue seeding).
        original = (1, 2, 3)
        target = render((1, 9, 3))
        loop = CADMorphLoop(render, distance, contribution,
                            make_generator([100]),  # always far from target
                            n_candidates=4, max_rounds=3)
        result = loop.run(original, target, seed=0)
        start = distance(render(original), target)
        self.assertLessEqual(result.distance, start)

    def test_already_matching_converges_immediately(self):
        original = (1, 9, 3)
        target = render((1, 9, 3))
        loop = CADMorphLoop(render, distance, contribution,
                            make_generator([2, 9, 5]), tol=0.0)
        result = loop.run(original, target, seed=0)
        self.assertEqual(result.sequence, (1, 9, 3))
        self.assertEqual(result.rounds, [])   # nothing to do
        self.assertTrue(result.converged)

    def test_structure_preservation_weight(self):
        # With lam>0 the loop still reaches the target here (single-segment
        # edit), and the recorded rounds show progress was accepted.
        original = (1, 2, 3)
        target = render((1, 9, 3))
        loop = CADMorphLoop(render, distance, contribution,
                            make_generator([2, 9, 5]),
                            n_candidates=6, lam=0.5, tol=0.0)
        result = loop.run(original, target, seed=1)
        self.assertEqual(result.sequence, (1, 9, 3))
        self.assertTrue(any(rd.accepted for rd in result.rounds))

    def test_invalid_params_raise(self):
        with self.assertRaises(ValueError):
            CADMorphLoop(render, distance, contribution,
                         make_generator([1]), n_candidates=0)
        with self.assertRaises(ValueError):
            CADMorphLoop(render, distance, contribution,
                         make_generator([1]), max_rounds=0)


if __name__ == "__main__":
    unittest.main()
