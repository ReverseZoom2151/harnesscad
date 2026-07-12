"""Tests for exploration.blockdecomp_reward."""

import unittest

from geometry.blockdecomp_domain import Shape
from geometry.blockdecomp_cut import full_cut
from exploration.blockdecomp_reward import (
    aspect_term,
    no_effect_penalty,
    quad_term,
    reward,
    terminal_bonus,
    variance_term,
)


def _two_equal_squares():
    # 4x2 rectangle split at x=2 -> two 2x2 squares of equal area.
    r = Shape.from_rectangles([(0, 0, 4, 2)])
    return full_cut(r, "vertical", 2.0)


class TestComponents(unittest.TestCase):
    def test_aspect_term_squares_is_one(self):
        self.assertAlmostEqual(aspect_term(_two_equal_squares()), 1.0)

    def test_aspect_term_below_one_for_non_squares(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 1.0)  # 1x2 and 3x2 blocks
        self.assertLess(aspect_term(parts), 1.0)

    def test_variance_zero_for_equal_areas(self):
        self.assertAlmostEqual(variance_term(_two_equal_squares()), 0.0)

    def test_variance_positive_for_unequal(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 1.0)
        self.assertGreater(variance_term(parts), 0.0)

    def test_quad_term_all_quads(self):
        self.assertAlmostEqual(quad_term(_two_equal_squares()), 1.0)

    def test_penalty_single_part(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        parts = full_cut(r, "vertical", 0.0)  # ineffective, 1 part
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(no_effect_penalty(parts), 1.0)

    def test_penalty_zero_when_effective(self):
        self.assertAlmostEqual(no_effect_penalty(_two_equal_squares()), 0.0)


class TestReward(unittest.TestCase):
    def test_all_squares_equal_area_is_best_case(self):
        rc = reward(_two_equal_squares())
        # scale*(1 - 0 + 10*1) - 5*0 - 1 = 11/3 - 1
        self.assertAlmostEqual(rc.total, (1.0 / 3.0) * 11.0 - 1.0)
        self.assertAlmostEqual(rc.aspect, 1.0)
        self.assertAlmostEqual(rc.quad, 1.0)
        self.assertAlmostEqual(rc.variance, 0.0)
        self.assertAlmostEqual(rc.penalty, 0.0)

    def test_ineffective_cut_is_penalised(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        good = reward(_two_equal_squares())
        bad = reward(full_cut(r, "vertical", 0.0))
        self.assertLess(bad.total, good.total)
        self.assertAlmostEqual(bad.penalty, 1.0)

    def test_unequal_areas_reduce_reward(self):
        r = Shape.from_rectangles([(0, 0, 4, 2)])
        equal = reward(_two_equal_squares())
        unequal = reward(full_cut(r, "vertical", 1.0))
        self.assertLess(unequal.total, equal.total)

    def test_deterministic(self):
        self.assertEqual(reward(_two_equal_squares()), reward(_two_equal_squares()))


class TestTerminalBonus(unittest.TestCase):
    def test_bonus_when_complete(self):
        self.assertAlmostEqual(terminal_bonus(True), 10.0)

    def test_no_bonus_when_incomplete(self):
        self.assertAlmostEqual(terminal_bonus(False), 0.0)


if __name__ == "__main__":
    unittest.main()
