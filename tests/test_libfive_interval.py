"""Tests for numeric.libfive_interval.

The core contract of interval arithmetic: the computed interval must *enclose*
the true range.  We verify that by dense sampling for every operator and for the
IR evaluator over boxes.
"""

from __future__ import annotations

import math
import random
import unittest

from geometry import libfive_frep_ir as ir
from numeric import libfive_interval as iv
from numeric.libfive_interval import Interval


def _sample_range(fn, lo, hi, n=400):
    vals = [fn(lo + (hi - lo) * k / n) for k in range(n + 1)]
    return min(vals), max(vals)


class TestIntervalEncloses(unittest.TestCase):
    """Each interval op must contain the true range over the operand box."""

    def _check_unary(self, iv_fn, real_fn, lo, hi):
        result = iv_fn(Interval(lo, hi))
        tmin, tmax = _sample_range(real_fn, lo, hi)
        self.assertLessEqual(result.lo, tmin + 1e-9,
                             "lower bound not enclosing")
        self.assertGreaterEqual(result.hi, tmax - 1e-9,
                                "upper bound not enclosing")

    def test_square(self):
        for lo, hi in [(-2, 3), (1, 4), (-5, -2), (-1, 1)]:
            self._check_unary(lambda i: i.square(), lambda t: t * t, lo, hi)

    def test_sqrt(self):
        self._check_unary(lambda i: i.sqrt(), math.sqrt, 0.5, 9.0)

    def test_abs(self):
        for lo, hi in [(-2, 3), (1, 4), (-5, -2)]:
            self._check_unary(lambda i: i.abs(), abs, lo, hi)

    def test_sin(self):
        for lo, hi in [(0, 1), (0, 4), (-3, 3), (1, 1.5), (0, 7)]:
            self._check_unary(lambda i: i.sin(), math.sin, lo, hi)

    def test_cos(self):
        for lo, hi in [(0, 1), (0, 4), (-3, 3), (2, 2.5), (0, 7)]:
            self._check_unary(lambda i: i.cos(), math.cos, lo, hi)

    def test_exp(self):
        self._check_unary(lambda i: i.exp(), math.exp, -2.0, 2.0)

    def test_recip(self):
        self._check_unary(lambda i: i.recip(), lambda t: 1.0 / t, 0.5, 4.0)

    def test_mul_encloses(self):
        rng = random.Random(1234)
        for _ in range(50):
            alo, ahi = sorted((rng.uniform(-5, 5), rng.uniform(-5, 5)))
            blo, bhi = sorted((rng.uniform(-5, 5), rng.uniform(-5, 5)))
            result = Interval(alo, ahi) * Interval(blo, bhi)
            tmin, tmax = math.inf, -math.inf
            for i in range(11):
                for j in range(11):
                    a = alo + (ahi - alo) * i / 10
                    b = blo + (bhi - blo) * j / 10
                    tmin = min(tmin, a * b)
                    tmax = max(tmax, a * b)
            self.assertLessEqual(result.lo, tmin + 1e-9)
            self.assertGreaterEqual(result.hi, tmax - 1e-9)

    def test_div_straddling_zero_is_unbounded(self):
        result = Interval(1.0, 2.0) / Interval(-1.0, 1.0)
        self.assertTrue(result.maybe_nan)
        self.assertEqual(result.lo, -math.inf)


class TestIREvaluator(unittest.TestCase):
    def test_circle_interval_encloses_field(self):
        g = ir.Graph()
        c = ir.circle(g, 0.0, 0.0, 1.0)
        box_lo, box_hi = (0.2, 0.3, 0.0), (0.8, 0.9, 0.0)
        result = iv.eval_interval(c, box_lo, box_hi)
        # sample the true field over the box
        tmin, tmax = math.inf, -math.inf
        for i in range(21):
            for j in range(21):
                x = box_lo[0] + (box_hi[0] - box_lo[0]) * i / 20
                y = box_lo[1] + (box_hi[1] - box_lo[1]) * j / 20
                v = ir.eval_point(c, x, y)
                tmin, tmax = min(tmin, v), max(tmax, v)
        self.assertLessEqual(result.lo, tmin + 1e-9)
        self.assertGreaterEqual(result.hi, tmax - 1e-9)


class TestClassify(unittest.TestCase):
    def test_pruning_decisions(self):
        g = ir.Graph()
        c = ir.circle(g, 0.0, 0.0, 1.0)
        # box far outside
        self.assertEqual(iv.classify(iv.eval_interval(c, (5, 5, 0), (6, 6, 0))),
                         iv.EMPTY)
        # box fully inside (near centre)
        self.assertEqual(
            iv.classify(iv.eval_interval(c, (-0.1, -0.1, 0), (0.1, 0.1, 0))),
            iv.FILLED)
        # box straddling the circle boundary
        self.assertEqual(
            iv.classify(iv.eval_interval(c, (0.5, -0.5, 0), (1.5, 0.5, 0))),
            iv.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
