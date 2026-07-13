"""Tests for geometry.libfive_frep_ir (the f-rep opcode graph IR)."""

from __future__ import annotations

import math
import unittest

from harnesscad.domain.geometry.sdf import libfive_frep_ir as ir


class TestBuildAndEval(unittest.TestCase):
    def test_leaves(self):
        g = ir.Graph()
        self.assertEqual(ir.eval_point(g.x(), 3.0, 5.0, 7.0), 3.0)
        self.assertEqual(ir.eval_point(g.y(), 3.0, 5.0, 7.0), 5.0)
        self.assertEqual(ir.eval_point(g.z(), 3.0, 5.0, 7.0), 7.0)
        self.assertEqual(ir.eval_point(g.constant(2.5), 0, 0, 0), 2.5)

    def test_arithmetic_matches_python(self):
        g = ir.Graph()
        expr = (g.x() * g.x() + g.y() * 3.0 - 1.0) / 2.0
        for (x, y) in [(1.0, 2.0), (-3.0, 4.0), (0.5, -0.5)]:
            expected = (x * x + y * 3.0 - 1.0) / 2.0
            self.assertAlmostEqual(ir.eval_point(expr, x, y), expected, places=12)

    def test_unary_ops(self):
        g = ir.Graph()
        expr = g.sqrt(g.square(g.x()) + g.square(g.y()))
        self.assertAlmostEqual(ir.eval_point(expr, 3.0, 4.0), 5.0, places=12)

    def test_circle_field_sign(self):
        g = ir.Graph()
        c = ir.circle(g, 0.0, 0.0, 1.0)
        self.assertLess(ir.eval_point(c, 0.0, 0.0), 0.0)   # centre inside
        self.assertGreater(ir.eval_point(c, 2.0, 0.0), 0.0)  # outside
        self.assertAlmostEqual(ir.eval_point(c, 1.0, 0.0), 0.0, places=12)  # on

    def test_make_callable_matches_eval_point(self):
        g = ir.Graph()
        expr = ir.sphere(g, 1.0, 2.0, 3.0, 2.0)
        f = ir.make_callable(expr)
        for p in [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 4.0, 4.0)]:
            self.assertAlmostEqual(f(*p), ir.eval_point(expr, *p), places=12)


class TestCSE(unittest.TestCase):
    def test_identical_subexpr_shared(self):
        g = ir.Graph()
        x = g.x()
        a = g.square(x)
        b = g.square(x)
        self.assertIs(a, b)  # hash-consed to the same node

    def test_repeated_subexpr_counted_once(self):
        g = ir.Graph()
        x = g.x()
        s = g.square(x)
        _ = s + s  # f + f
        # nodes: x, square(x), add  -> exactly 3 distinct nodes
        expr = s + s
        order = ir._post_order(expr)
        self.assertEqual(len(order), 3)

    def test_commutative_canonicalisation(self):
        g = ir.Graph()
        x, y = g.x(), g.y()
        self.assertIs(x + y, y + x)     # add is commutative
        self.assertIs(g.min(x, y), g.min(y, x))

    def test_noncommutative_not_merged(self):
        g = ir.Graph()
        x, y = g.x(), g.y()
        self.assertIsNot(x - y, y - x)


class TestConstantFolding(unittest.TestCase):
    def test_binary_fold(self):
        g = ir.Graph()
        n = g.constant(2.0) + g.constant(3.0)
        self.assertEqual(n.op, "const")
        self.assertEqual(n.value, 5.0)

    def test_unary_fold(self):
        g = ir.Graph()
        n = g.sqrt(g.constant(9.0))
        self.assertEqual(n.op, "const")
        self.assertAlmostEqual(n.value, 3.0, places=12)

    def test_fold_disabled(self):
        g = ir.Graph(fold=False)
        n = g.constant(2.0) + g.constant(3.0)
        self.assertEqual(n.op, "add")


class TestPrinting(unittest.TestCase):
    def test_infix(self):
        g = ir.Graph()
        expr = g.max(g.x() - 1.0, g.sqrt(g.y()))
        self.assertEqual(ir.to_infix(expr), "max((x - 1), sqrt(y))")

    def test_sexpr(self):
        g = ir.Graph()
        expr = g.max(g.x() - 1.0, g.sqrt(g.y()))
        self.assertEqual(ir.to_sexpr(expr), "(max (- x 1) (sqrt y))")


class TestCSG(unittest.TestCase):
    def test_union_is_min(self):
        g = ir.Graph()
        a = ir.circle(g, -0.5, 0.0, 1.0)
        b = ir.circle(g, 0.5, 0.0, 1.0)
        u = ir.union(g, a, b)
        # a point inside a but outside b is inside the union
        self.assertLess(ir.eval_point(u, -1.2, 0.0), 0.0)

    def test_difference(self):
        g = ir.Graph()
        big = ir.circle(g, 0.0, 0.0, 1.0)
        small = ir.circle(g, 0.0, 0.0, 0.5)
        d = ir.difference(g, big, small)
        self.assertLess(ir.eval_point(d, 0.75, 0.0), 0.0)   # in the ring
        self.assertGreater(ir.eval_point(d, 0.0, 0.0), 0.0)  # in the hole


if __name__ == "__main__":
    unittest.main()
