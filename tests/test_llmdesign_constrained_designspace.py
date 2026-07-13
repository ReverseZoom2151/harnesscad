"""Tests for exploration.llmdesign_constrained_designspace.

Covers ParameterSpec validation, each constraint kind and operator, bounds and
integrality gating, is_valid / violations, deterministic constraint-respecting
sampling (including a tight space that must not hang), and the paper's Lego and
car design-space examples.
"""

import unittest

from harnesscad.agents.exploration.llmdesign_constrained_designspace import (
    ParameterSpec,
    Inequality,
    Divisible,
    DesignSpace,
)


def _lego_space():
    return DesignSpace(
        [
            ParameterSpec("brick_length", "integer", 3, 30),
            ParameterSpec("brick_width", "integer", 3, 30),
            ParameterSpec("brick_height", "integer", 1, 10),
        ],
        [
            Divisible("brick_length", 3),
            Divisible("brick_width", 3),
        ],
    )


def _car_space():
    return DesignSpace(
        [
            ParameterSpec("length", "continuous", 3.0, 6.0),
            ParameterSpec("width", "continuous", 1.0, 3.0),
            ParameterSpec("height", "continuous", 1.0, 2.5),
            ParameterSpec("wheel_radius", "continuous", 0.2, 1.0),
        ],
        [
            Inequality("width", "<", "length"),
            Inequality("width", ">", "height"),
            Inequality("wheel_radius", "<", "height"),
        ],
    )


class TestParameterSpec(unittest.TestCase):
    def test_low_greater_than_high_rejected(self):
        with self.assertRaises(ValueError):
            ParameterSpec("x", "continuous", 5.0, 1.0)

    def test_bad_kind_rejected(self):
        with self.assertRaises(ValueError):
            ParameterSpec("x", "categorical", 0, 1)

    def test_integer_bounds_must_be_integral(self):
        with self.assertRaises(ValueError):
            ParameterSpec("x", "integer", 0.5, 3)

    def test_contains_bounds_and_integrality(self):
        spec = ParameterSpec("x", "integer", 1, 10)
        self.assertTrue(spec.contains(1))
        self.assertTrue(spec.contains(10))
        self.assertFalse(spec.contains(0))
        self.assertFalse(spec.contains(11))
        self.assertFalse(spec.contains(3.5))
        self.assertFalse(spec.contains(True))


class TestInequality(unittest.TestCase):
    def test_param_param_all_operators(self):
        a = {"p": 2, "q": 5}
        self.assertTrue(Inequality("p", "<", "q").satisfied(a))
        self.assertTrue(Inequality("p", "<=", "q").satisfied(a))
        self.assertFalse(Inequality("p", ">", "q").satisfied(a))
        self.assertFalse(Inequality("p", ">=", "q").satisfied(a))
        eq = {"p": 5, "q": 5}
        self.assertTrue(Inequality("p", "<=", "q").satisfied(eq))
        self.assertTrue(Inequality("p", ">=", "q").satisfied(eq))
        self.assertFalse(Inequality("p", "<", "q").satisfied(eq))

    def test_param_const(self):
        a = {"wheel_radius": 0.4}
        self.assertTrue(Inequality("wheel_radius", "<", 1.0).satisfied(a))
        self.assertFalse(Inequality("wheel_radius", ">", 1.0).satisfied(a))

    def test_bad_operator_rejected(self):
        with self.assertRaises(ValueError):
            Inequality("a", "==", "b")

    def test_missing_parameter_is_unsatisfied(self):
        self.assertFalse(Inequality("a", "<", "b").satisfied({"a": 1}))

    def test_parameters_reported(self):
        self.assertEqual(Inequality("a", "<", "b").parameters(), {"a", "b"})
        self.assertEqual(Inequality("a", "<", 3).parameters(), {"a"})


class TestDivisible(unittest.TestCase):
    def test_divisible_true_false(self):
        d = Divisible("brick_length", 3)
        self.assertTrue(d.satisfied({"brick_length": 9}))
        self.assertFalse(d.satisfied({"brick_length": 10}))

    def test_non_integral_value_unsatisfied(self):
        self.assertFalse(Divisible("x", 3).satisfied({"x": 9.5}))

    def test_bad_modulus_rejected(self):
        with self.assertRaises(ValueError):
            Divisible("x", 0)
        with self.assertRaises(ValueError):
            Divisible("x", 2.5)

    def test_missing_parameter_unsatisfied(self):
        self.assertFalse(Divisible("x", 3).satisfied({}))


class TestDesignSpaceValidity(unittest.TestCase):
    def test_is_valid_true(self):
        space = _lego_space()
        self.assertTrue(space.is_valid(
            {"brick_length": 6, "brick_width": 9, "brick_height": 2}))

    def test_is_valid_false_divisibility(self):
        space = _lego_space()
        self.assertFalse(space.is_valid(
            {"brick_length": 7, "brick_width": 9, "brick_height": 2}))

    def test_bounds_checking(self):
        space = _lego_space()
        self.assertFalse(space.is_valid(
            {"brick_length": 33, "brick_width": 9, "brick_height": 2}))

    def test_integer_integrality_enforced(self):
        space = _lego_space()
        self.assertFalse(space.is_valid(
            {"brick_length": 6.5, "brick_width": 9, "brick_height": 2}))

    def test_violations_lists_right_failures(self):
        space = _lego_space()
        problems = space.violations(
            {"brick_length": 33, "brick_width": 8, "brick_height": 20})
        joined = " | ".join(problems)
        # brick_length: out of bounds (and not multiple of 3 -> reported too)
        self.assertIn("brick_length", joined)
        # brick_width 8 is in bounds but not divisible by 3
        self.assertTrue(any("brick_width" in p and "divis" in p for p in problems))
        # brick_height 20 out of bounds
        self.assertTrue(any("brick_height" in p and "out of bounds" in p
                            for p in problems))

    def test_violations_missing_parameter(self):
        space = _lego_space()
        problems = space.violations({"brick_length": 6, "brick_width": 9})
        self.assertTrue(any("missing parameter" in p and "brick_height" in p
                            for p in problems))

    def test_duplicate_parameter_rejected(self):
        with self.assertRaises(ValueError):
            DesignSpace([
                ParameterSpec("x", "integer", 0, 1),
                ParameterSpec("x", "integer", 0, 1),
            ])


class TestSampleValid(unittest.TestCase):
    def test_determinism_same_seed(self):
        space = _car_space()
        a = space.sample_valid(8, seed=0)
        b = space.sample_valid(8, seed=0)
        self.assertEqual(a, b)

    def test_all_valid_and_distinct(self):
        space = _car_space()
        samples = space.sample_valid(10, seed=0)
        self.assertTrue(len(samples) > 0)
        for row in samples:
            self.assertTrue(space.is_valid(row), row)
        keys = [tuple(sorted(r.items())) for r in samples]
        self.assertEqual(len(keys), len(set(keys)))

    def test_lego_sample_respects_divisibility(self):
        space = _lego_space()
        samples = space.sample_valid(15, seed=7)
        self.assertTrue(len(samples) > 0)
        for row in samples:
            self.assertEqual(row["brick_length"] % 3, 0)
            self.assertEqual(row["brick_width"] % 3, 0)
            self.assertTrue(3 <= row["brick_length"] <= 30)
            self.assertTrue(1 <= row["brick_height"] <= 10)

    def test_tight_space_does_not_hang(self):
        # Contradictory constraints -> empty space; must return quickly with
        # fewer than requested (here zero) rather than looping forever.
        space = DesignSpace(
            [ParameterSpec("a", "integer", 1, 10)],
            [Inequality("a", "<", 1), Inequality("a", ">", 5)],
        )
        samples = space.sample_valid(5, seed=0, max_attempts_per=50)
        self.assertEqual(samples, [])


class TestLegoEnumeration(unittest.TestCase):
    def test_full_lego_enumeration_divisibility(self):
        space = _lego_space()
        rows = space.enumerate_valid()
        self.assertTrue(len(rows) > 0)
        for row in rows:
            self.assertEqual(row["brick_length"] % 3, 0)
            self.assertEqual(row["brick_width"] % 3, 0)
            self.assertTrue(3 <= row["brick_length"] <= 30)
            self.assertTrue(3 <= row["brick_width"] <= 30)
            self.assertTrue(1 <= row["brick_height"] <= 10)
        # length multiples of 3 in [3,30]: 10; width: 10; height in [1,10]: 10.
        self.assertEqual(len(rows), 10 * 10 * 10)

    def test_small_window_hand_counted(self):
        space = DesignSpace(
            [
                ParameterSpec("brick_length", "integer", 3, 9),   # 3,6,9 -> 3
                ParameterSpec("brick_width", "integer", 3, 6),    # 3,6   -> 2
                ParameterSpec("brick_height", "integer", 1, 2),   # 1,2   -> 2
            ],
            [Divisible("brick_length", 3), Divisible("brick_width", 3)],
        )
        rows = space.enumerate_valid()
        self.assertEqual(len(rows), 3 * 2 * 2)


class TestCarExample(unittest.TestCase):
    def test_hand_made_valid(self):
        space = _car_space()
        self.assertTrue(space.is_valid(
            {"length": 5.0, "width": 2.0, "height": 1.5, "wheel_radius": 0.5}))

    def test_hand_made_invalid(self):
        space = _car_space()
        # width (2.0) not < length (1.8) -> invalid
        self.assertFalse(space.is_valid(
            {"length": 1.8, "width": 2.0, "height": 1.5, "wheel_radius": 0.5}))

    def test_sample_valid_reproducible_and_valid(self):
        space = _car_space()
        first = space.sample_valid(6, seed=0)
        second = space.sample_valid(6, seed=0)
        self.assertEqual(first, second)
        for row in first:
            self.assertTrue(space.is_valid(row))
            self.assertLess(row["width"], row["length"])
            self.assertGreater(row["width"], row["height"])
            self.assertLess(row["wheel_radius"], row["height"])


if __name__ == "__main__":
    unittest.main()
