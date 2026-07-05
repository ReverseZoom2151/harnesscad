"""Tests for editing/cadreasoner_beam.py (geometry-guided stochastic beam)."""

import unittest

from bench.geometry_distance import symmetric_chamfer
from editing.cadreasoner_beam import BeamResult, run_geometry_beam


def _render(program):
    if program == "invalid":
        return None
    return [(float(program), 0.0, 0.0)]


def _score(target, render):
    return symmetric_chamfer(target, render)


# t=1 seeds: 0, 2, 4, 6, 8 (for n=5). Best seed is 8 (closest to target 10).
def _seed_gen(target, slot):
    return slot * 2.0


# children move a survivor toward the target: parent + slot (0..n-1).
def _child_gen(target, parent_program, parent_render, slot):
    return float(parent_program) + slot


class TestBeamConverges(unittest.TestCase):
    def _run(self, n=5, steps=2):
        target = [(10.0, 0.0, 0.0)]
        return run_geometry_beam(
            target, _seed_gen, _child_gen, _render, _score, n=n, steps=steps)

    def test_reaches_exact_target(self):
        res = self._run(n=5, steps=2)
        self.assertIsInstance(res, BeamResult)
        # From best seed 8, child slot 2 -> 10 -> score 0.
        self.assertAlmostEqual(res.best_score, 0.0)
        self.assertAlmostEqual(res.best_program, 10.0)

    def test_render_budget_matches_formula(self):
        res = self._run(n=5, steps=2)
        # all seeds valid -> N + (s-1) N^2 = 5 + 25 = 30
        self.assertEqual(res.expected_render_budget, 30)
        self.assertEqual(res.total_renders, 30)

    def test_survivors_capped_at_n(self):
        res = self._run(n=3, steps=3)
        for surv in res.survivors_per_step:
            self.assertLessEqual(len(surv), 3)

    def test_deterministic(self):
        self.assertEqual(self._run().to_dict(), self._run().to_dict())

    def test_single_step_is_seed_only(self):
        res = self._run(n=5, steps=1)
        self.assertEqual(res.total_renders, 5)
        self.assertEqual(len(res.survivors_per_step), 1)


class TestBeamInvalid(unittest.TestCase):
    def test_invalid_seeds_discarded(self):
        target = [(10.0, 0.0, 0.0)]

        def seed_gen(t, slot):
            return "invalid" if slot % 2 == 0 else slot * 2.0

        res = run_geometry_beam(
            target, seed_gen, _child_gen, _render, _score, n=4, steps=1)
        self.assertGreater(res.total_invalid, 0)
        # survivors are all valid
        for c in res.survivors_per_step[0]:
            self.assertTrue(c.valid)

    def test_all_invalid_yields_no_best(self):
        target = [(10.0, 0.0, 0.0)]
        res = run_geometry_beam(
            target, lambda t, s: "invalid", _child_gen, _render, _score,
            n=3, steps=2)
        self.assertIsNone(res.best)
        self.assertEqual(res.best_program, None)


class TestBeamValidation(unittest.TestCase):
    def test_rejects_bad_params(self):
        target = [(0.0, 0.0, 0.0)]
        with self.assertRaises(ValueError):
            run_geometry_beam(target, _seed_gen, _child_gen, _render, _score, n=0)
        with self.assertRaises(ValueError):
            run_geometry_beam(target, _seed_gen, _child_gen, _render, _score, steps=0)


if __name__ == "__main__":
    unittest.main()
