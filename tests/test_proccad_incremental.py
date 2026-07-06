"""Tests for procedural.proccad_incremental (incremental re-evaluation)."""

import unittest

from procedural.proccad_incremental import ProceduralGraph


def _build():
    #   a   b   c
    #    \ /    |
    #    sum    sq
    #      \   /
    #      total
    g = ProceduralGraph()
    g.add_input("a", 1.0).add_input("b", 2.0).add_input("c", 3.0)
    g.add_compute("sum", ["a", "b"], lambda d: d["a"] + d["b"])
    g.add_compute("sq", ["c"], lambda d: d["c"] * d["c"])
    g.add_compute("total", ["sum", "sq"], lambda d: d["sum"] + d["sq"])
    return g


class EvaluationTest(unittest.TestCase):
    def test_initial_evaluate_computes_all(self):
        g = _build()
        vals, n = g.evaluate()
        self.assertEqual(n, 6)  # 3 inputs + 3 compute
        self.assertAlmostEqual(vals["total"], (1 + 2) + 9)

    def test_second_evaluate_recomputes_nothing(self):
        g = _build()
        g.evaluate()
        _, n = g.evaluate()
        self.assertEqual(n, 0)

    def test_edit_only_dirties_downstream_cone(self):
        g = _build()
        g.evaluate()
        dirtied = g.set_input("c", 4.0)
        # editing c invalidates c, sq, total -- NOT a, b, sum
        self.assertEqual(dirtied, {"c", "sq", "total"})

    def test_edit_recomputes_only_dirty(self):
        g = _build()
        g.evaluate()
        g.set_input("c", 4.0)
        _, n = g.evaluate()
        self.assertEqual(n, 3)  # c, sq, total

    def test_edit_propagates_value(self):
        g = _build()
        g.value("total")
        g.set_input("a", 10.0)
        self.assertAlmostEqual(g.value("total"), (10 + 2) + 9)

    def test_noop_edit_dirties_nothing(self):
        g = _build()
        g.evaluate()
        dirtied = g.set_input("a", 1.0)  # unchanged
        self.assertEqual(dirtied, set())
        _, n = g.evaluate()
        self.assertEqual(n, 0)

    def test_independent_branch_not_recomputed(self):
        g = _build()
        g.evaluate()
        g.set_input("a", 5.0)  # affects sum, total; sq untouched
        dirtied = g.set_input("a", 5.0) or {"a"}
        # re-run: only a, sum, total recompute
        g.set_input("a", 6.0)
        _, n = g.evaluate()
        self.assertEqual(n, 3)


class ConstructionTest(unittest.TestCase):
    def test_unknown_dependency_rejected(self):
        g = ProceduralGraph()
        with self.assertRaises(ValueError):
            g.add_compute("x", ["missing"], lambda d: 0)

    def test_duplicate_node_rejected(self):
        g = ProceduralGraph()
        g.add_input("a", 1)
        with self.assertRaises(ValueError):
            g.add_input("a", 2)

    def test_set_non_input_rejected(self):
        g = _build()
        with self.assertRaises(ValueError):
            g.set_input("sum", 0.0)

    def test_deterministic(self):
        self.assertEqual(_build().value("total"), _build().value("total"))


if __name__ == "__main__":
    unittest.main()
