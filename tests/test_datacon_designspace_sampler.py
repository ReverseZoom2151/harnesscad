"""Tests for exploration.datacon_designspace_sampler."""

import unittest

from exploration.datacon_designspace_sampler import (
    validate_space,
    uniform_sample,
    stratified_sample,
    grid_sample,
    coverage_of_samples,
    marginal_coverage,
)


def wheel_space():
    return {
        "spoke_style": ("categorical", ["five-spoke", "multispoke", "mesh", "minimalist"]),
        "rim_diameter": ("range", 14, 21),
        "spoke_count": ("int_range", 5, 12),
    }


class TestValidation(unittest.TestCase):
    def test_valid_space_ok(self):
        self.assertEqual(validate_space(wheel_space()), wheel_space())

    def test_rejects_non_dict(self):
        with self.assertRaises(ValueError):
            validate_space([("range", 1, 2)])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_space({})

    def test_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            validate_space({"x": ("weird", 1, 2)})

    def test_rejects_empty_categorical(self):
        with self.assertRaises(ValueError):
            validate_space({"x": ("categorical", [])})

    def test_rejects_bad_range(self):
        with self.assertRaises(ValueError):
            validate_space({"x": ("range", 5, 5)})

    def test_rejects_range_wrong_len(self):
        with self.assertRaises(ValueError):
            validate_space({"x": ("range", 5)})

    def test_rejects_non_integer_int_range(self):
        with self.assertRaises(ValueError):
            validate_space({"x": ("int_range", 1.5, 3.5)})


class TestUniform(unittest.TestCase):
    def test_count_and_domains(self):
        space = wheel_space()
        samples = uniform_sample(space, 20, seed=1)
        self.assertEqual(len(samples), 20)
        for row in samples:
            self.assertIn(row["spoke_style"], space["spoke_style"][1])
            self.assertTrue(14 <= row["rim_diameter"] <= 21)
            self.assertTrue(5 <= row["spoke_count"] <= 12)
            self.assertEqual(int(row["spoke_count"]), row["spoke_count"])

    def test_deterministic(self):
        space = wheel_space()
        a = uniform_sample(space, 15, seed=7)
        b = uniform_sample(space, 15, seed=7)
        self.assertEqual(a, b)

    def test_different_seed_differs(self):
        space = wheel_space()
        a = uniform_sample(space, 15, seed=1)
        b = uniform_sample(space, 15, seed=2)
        self.assertNotEqual(a, b)


class TestStratified(unittest.TestCase):
    def test_count(self):
        samples = stratified_sample(wheel_space(), 12, seed=3)
        self.assertEqual(len(samples), 12)

    def test_categorical_even_roundrobin_exact(self):
        space = wheel_space()
        n = 8
        samples = stratified_sample(space, n, seed=5)
        counts = {}
        for row in samples:
            counts[row["spoke_style"]] = counts.get(row["spoke_style"], 0) + 1
        self.assertEqual(set(counts.keys()), set(space["spoke_style"][1]))
        for c in space["spoke_style"][1]:
            self.assertEqual(counts[c], 2)

    def test_categorical_roundrobin_diff_by_one(self):
        space = wheel_space()
        n = 10
        samples = stratified_sample(space, n, seed=9)
        counts = {c: 0 for c in space["spoke_style"][1]}
        for row in samples:
            counts[row["spoke_style"]] += 1
        vals = list(counts.values())
        self.assertLessEqual(max(vals) - min(vals), 1)
        self.assertEqual(sum(vals), n)

    def test_continuous_covers_all_strata(self):
        space = wheel_space()
        n = 10
        samples = stratified_sample(space, n, seed=11)
        mc = marginal_coverage(samples, space)
        # bins used by marginal_coverage = min(n, 10) = 10, and stratified draws
        # one value per stratum, so every bin is covered.
        self.assertAlmostEqual(mc["rim_diameter"], 1.0)

    def test_deterministic(self):
        space = wheel_space()
        a = stratified_sample(space, 16, seed=42)
        b = stratified_sample(space, 16, seed=42)
        self.assertEqual(a, b)

    def test_stratified_marginal_ge_uniform(self):
        space = wheel_space()
        n = 12
        strat = stratified_sample(space, n, seed=4)
        uni = uniform_sample(space, n, seed=4)
        self.assertGreaterEqual(
            marginal_coverage(strat, space)["rim_diameter"],
            marginal_coverage(uni, space)["rim_diameter"],
        )


class TestGrid(unittest.TestCase):
    def test_size(self):
        space = wheel_space()
        per_dim = 3
        rows = grid_sample(space, per_dim)
        # 4 categorical choices * per_dim^2 continuous dims.
        # int_range 5..12 with 3 evenly spaced -> may dedupe but here 3 distinct.
        n_cat = len(space["spoke_style"][1])
        # rim_diameter -> 3 values, spoke_count -> 3 integer values (5, 8/9, 12)
        # count actual axis sizes by inspection: both continuous have 3 values.
        expected = n_cat * per_dim * per_dim
        self.assertEqual(len(rows), expected)

    def test_size_simple_all_continuous(self):
        space = {"a": ("range", 0, 1), "b": ("range", 0, 10)}
        rows = grid_sample(space, 4)
        self.assertEqual(len(rows), 16)

    def test_raises_on_explosion(self):
        space = {
            "a": ("range", 0, 1),
            "b": ("range", 0, 1),
            "c": ("range", 0, 1),
            "d": ("range", 0, 1),
        }
        with self.assertRaises(ValueError):
            grid_sample(space, 40)


class TestCoverage(unittest.TestCase):
    def test_in_unit_interval(self):
        space = wheel_space()
        samples = stratified_sample(space, 40, seed=2)
        cov = coverage_of_samples(samples, space, bins=4)
        self.assertGreaterEqual(cov, 0.0)
        self.assertLessEqual(cov, 1.0)

    def test_stratified_ge_degenerate(self):
        space = wheel_space()
        good = stratified_sample(space, 40, seed=2)
        degenerate = [
            {"spoke_style": "mesh", "rim_diameter": 17.0, "spoke_count": 8}
            for _ in range(40)
        ]
        cov_good = coverage_of_samples(good, space, bins=4)
        cov_bad = coverage_of_samples(degenerate, space, bins=4)
        self.assertGreaterEqual(cov_good, cov_bad)

    def test_marginal_range(self):
        space = wheel_space()
        samples = stratified_sample(space, 20, seed=8)
        mc = marginal_coverage(samples, space)
        for v in mc.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()
