"""Tests for the EvoCAD evolutionary loop and variation operators."""
import random
import unittest

from harnesscad.agents.exploration.evolution import (
    average_rankings,
    ordering_to_ranks,
    rank_probabilities,
    select_parents,
    elite_indices,
    evolve,
    EvoResult,
)
from harnesscad.agents.exploration.variation import (
    CadOp,
    CadProgram,
    crossover,
    mutate,
    program_signature,
)


class TestRankPrimitives(unittest.TestCase):
    def test_average_rankings(self):
        avg = average_rankings([[0, 1, 2], [2, 1, 0]], 3)
        self.assertEqual(avg, [1.0, 1.0, 1.0])

    def test_average_rankings_length_mismatch(self):
        with self.assertRaises(ValueError):
            average_rankings([[0, 1]], 3)

    def test_average_rankings_empty(self):
        with self.assertRaises(ValueError):
            average_rankings([], 3)

    def test_ordering_to_ranks(self):
        # best-first order [2, 0, 1] -> index 2 rank0, index0 rank1, index1 rank2
        self.assertEqual(ordering_to_ranks([2, 0, 1], 3), [1.0, 2.0, 0.0])

    def test_ordering_to_ranks_not_permutation(self):
        with self.assertRaises(ValueError):
            ordering_to_ranks([0, 0, 1], 3)

    def test_rank_probabilities_monotone(self):
        probs = rank_probabilities([0.0, 1.0, 2.0], lam=0.5)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_rank_probabilities_lambda_zero_uniform(self):
        probs = rank_probabilities([0.0, 5.0, 9.0], lam=0.0)
        for p in probs:
            self.assertAlmostEqual(p, 1.0 / 3.0)

    def test_rank_probabilities_matches_equation(self):
        import math
        lam = 0.5
        ranks = [0.0, 1.0, 3.0]
        probs = rank_probabilities(ranks, lam)
        weights = [math.exp(-lam * r) for r in ranks]
        total = sum(weights)
        for p, w in zip(probs, weights):
            self.assertAlmostEqual(p, w / total)

    def test_rank_probabilities_negative_lambda(self):
        with self.assertRaises(ValueError):
            rank_probabilities([0.0, 1.0], lam=-0.1)


class TestSelection(unittest.TestCase):
    def test_select_parents_count_and_bounds(self):
        rng = random.Random(0)
        pairs = select_parents([0.7, 0.2, 0.1], 4, rng)
        self.assertEqual(len(pairs), 4)
        for a, b in pairs:
            self.assertIn(a, (0, 1, 2))
            self.assertIn(b, (0, 1, 2))

    def test_select_parents_distinct(self):
        rng = random.Random(3)
        pairs = select_parents([0.5, 0.5], 20, rng, distinct=True)
        for a, b in pairs:
            self.assertNotEqual(a, b)

    def test_select_parents_deterministic(self):
        a = select_parents([0.4, 0.4, 0.2], 6, random.Random(11))
        b = select_parents([0.4, 0.4, 0.2], 6, random.Random(11))
        self.assertEqual(a, b)

    def test_select_parents_favours_high_probability(self):
        rng = random.Random(7)
        pairs = select_parents([0.9, 0.05, 0.05], 200, rng, distinct=False)
        flat = [x for pair in pairs for x in pair]
        self.assertGreater(flat.count(0), flat.count(1) + flat.count(2))

    def test_elite_indices(self):
        self.assertEqual(elite_indices([2.0, 0.5, 1.0], 1), [1])
        self.assertEqual(elite_indices([2.0, 0.5, 1.0, 0.5], 2), [1, 3])


class TestVariation(unittest.TestCase):
    def _prog(self, *names):
        return CadProgram.of(*[CadOp.make(n, size=float(i + 1)) for i, n in enumerate(names)])

    def test_crossover_recombines(self):
        rng = random.Random(1)
        a = self._prog("sketch", "extrude", "hole")
        b = self._prog("rect", "cut")
        child = crossover(a, b, rng)
        self.assertIsInstance(child, CadProgram)
        self.assertGreaterEqual(len(child), 1)

    def test_crossover_empty_parent(self):
        rng = random.Random(1)
        a = self._prog("sketch")
        empty = CadProgram.of()
        self.assertEqual(crossover(a, empty, rng), a)
        self.assertEqual(crossover(empty, a, rng), a)

    def test_crossover_never_empty(self):
        rng = random.Random(5)
        a = self._prog("sketch", "extrude")
        b = self._prog("rect", "cut")
        for _ in range(50):
            self.assertGreaterEqual(len(crossover(a, b, rng)), 1)

    def test_mutate_changes_or_preserves_validity(self):
        rng = random.Random(2)
        prog = self._prog("sketch", "extrude", "hole")
        for _ in range(50):
            m = mutate(prog, rng)
            self.assertIsInstance(m, CadProgram)
            self.assertGreaterEqual(len(m), 1)

    def test_mutate_deterministic(self):
        p = self._prog("sketch", "extrude", "hole")
        self.assertEqual(
            program_signature(mutate(p, random.Random(9))),
            program_signature(mutate(p, random.Random(9))),
        )

    def test_signature_hashable(self):
        sig = program_signature(self._prog("a", "b"))
        self.assertEqual(hash(sig), hash(sig))


class TestEvolveLoop(unittest.TestCase):
    def setUp(self):
        # A target program length; fitness = distance from target length.
        self.target = 4

        def ranker(pop, gen, rep):
            # deterministic RLM stand-in: rank by |len - target|, ties by index.
            order = sorted(range(len(pop)), key=lambda i: (abs(len(pop[i]) - self.target), i))
            ranks = [0.0] * len(pop)
            for r, idx in enumerate(order):
                ranks[idx] = float(r)
            return ranks

        self.ranker = ranker
        self.pop = [
            CadProgram.of(*[CadOp.make("op", size=1.0)] * n) for n in (1, 2, 3, 5, 6, 8)
        ]

    def test_evolve_runs_and_preserves_size(self):
        res = evolve(self.pop, self.ranker, crossover, mutate,
                     generations=4, mutation_prob=0.5, lam=0.5,
                     num_elites=1, seed=0)
        self.assertIsInstance(res, EvoResult)
        self.assertEqual(len(res.final_population), len(self.pop))
        self.assertEqual(len(res.history), 4)

    def test_evolve_deterministic(self):
        r1 = evolve(self.pop, self.ranker, crossover, mutate, seed=42)
        r2 = evolve(self.pop, self.ranker, crossover, mutate, seed=42)
        self.assertEqual(r1.best_avg_rank, r2.best_avg_rank)
        self.assertEqual([program_signature(p) for p in r1.final_population],
                         [program_signature(p) for p in r2.final_population])

    def test_evolve_elitism_keeps_best_rank_zero(self):
        res = evolve(self.pop, self.ranker, crossover, mutate,
                     generations=3, num_elites=1, seed=1)
        # best avg rank in final population should be 0 (an individual at target)
        self.assertEqual(res.best_avg_rank, 0.0)

    def test_evolve_history_fields(self):
        res = evolve(self.pop, self.ranker, crossover, mutate, generations=2, seed=0)
        rec = res.history[0]
        self.assertEqual(len(rec.probabilities), len(self.pop))
        self.assertAlmostEqual(sum(rec.probabilities), 1.0)
        self.assertEqual(len(rec.elite_indices), 1)
        self.assertEqual(len(rec.parent_pairs), len(self.pop) - 1)

    def test_evolve_zero_generations(self):
        res = evolve(self.pop, self.ranker, crossover, mutate, generations=0, seed=0)
        self.assertEqual(res.history, [])
        self.assertEqual(len(res.final_population), len(self.pop))

    def test_evolve_no_elites(self):
        res = evolve(self.pop, self.ranker, crossover, mutate,
                     generations=2, num_elites=0, seed=0)
        rec = res.history[0]
        self.assertEqual(rec.elite_indices, ())
        self.assertEqual(len(rec.parent_pairs), len(self.pop))

    def test_evolve_empty_population_raises(self):
        with self.assertRaises(ValueError):
            evolve([], self.ranker, crossover, mutate)

    def test_evolve_bad_mutation_prob(self):
        with self.assertRaises(ValueError):
            evolve(self.pop, self.ranker, crossover, mutate, mutation_prob=1.5)

    def test_evolve_improves_or_holds_mean_rank(self):
        # elitism guarantees best never worsens across the run.
        res = evolve(self.pop, self.ranker, crossover, mutate,
                     generations=4, num_elites=1, seed=3)
        best_over_time = [min(r.avg_ranks) for r in res.history]
        self.assertTrue(all(b == 0.0 for b in best_over_time))


if __name__ == "__main__":
    unittest.main()
