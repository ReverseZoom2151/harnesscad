"""Tests for procedural.shapegramm_markov_grammar (grammar + Markov selection)."""

import unittest

from procedural.shapegramm_markov_grammar import Rule, MarkovGrammar


def _grammar():
    # A -> B (via r_ab1) or A -> C (via r_ab2)
    # B -> x (via r_bx)   C -> y (via r_cy)
    rules = [
        Rule("r_ab1", "A", ("B",), weight=1.0),
        Rule("r_ab2", "A", ("C",), weight=1.0),
        Rule("r_bx", "B", ("x",)),
        Rule("r_cy", "C", ("y",)),
    ]
    return rules


class ExpansionTest(unittest.TestCase):
    def test_terminals_only_remain(self):
        g = MarkovGrammar(_grammar(), terminals={"x", "y"})
        terms, trace, diag = g.expand("A", seed=1)
        for sym, _ in terms:
            self.assertIn(sym, {"x", "y"})
        self.assertEqual(diag, ())

    def test_determinism_same_seed(self):
        g = MarkovGrammar(_grammar(), terminals={"x", "y"})
        a = g.expand("A", seed=7)
        b = g.expand("A", seed=7)
        self.assertEqual(a, b)

    def test_trace_records_parent_rule(self):
        g = MarkovGrammar(_grammar(), terminals={"x", "y"})
        _, trace, _ = g.expand("A", seed=0)
        # first application expands A with parent None
        self.assertEqual(trace[0][0], "A")
        self.assertIsNone(trace[0][2])
        # the child expansion carries the A-rule as its parent
        self.assertEqual(trace[1][2], trace[0][1])


class MarkovSelectionTest(unittest.TestCase):
    def test_transition_table_constrains_choice(self):
        # An outer rule R that produces two A's; the Markov table forces the
        # A reached via r_root to always take r_ab2 (-> C -> y).
        rules = _grammar() + [Rule("r_root", "S", ("A",))]
        transitions = {"r_root": {"r_ab2": 1.0}}
        g = MarkovGrammar(rules, terminals={"x", "y"}, transitions=transitions)
        for seed in range(20):
            terms, _, _ = g.expand("S", seed=seed)
            syms = [s for s, _ in terms]
            self.assertEqual(syms, ["y"])  # never reaches x

    def test_no_table_falls_back_to_context_free_weights(self):
        # Heavily bias r_ab1 by base weight; parent None has no table.
        rules = [
            Rule("r_ab1", "A", ("B",), weight=1000.0),
            Rule("r_ab2", "A", ("C",), weight=1.0),
            Rule("r_bx", "B", ("x",)),
            Rule("r_cy", "C", ("y",)),
        ]
        g = MarkovGrammar(rules, terminals={"x", "y"})
        counts = {"x": 0, "y": 0}
        for seed in range(200):
            terms, _, _ = g.expand("A", seed=seed)
            counts[terms[0][0]] += 1
        self.assertGreater(counts["x"], counts["y"])

    def test_different_parents_pick_differently(self):
        # Same predecessor A expands differently based on how it was reached.
        rules = _grammar() + [
            Rule("r_left", "L", ("A",)),
            Rule("r_right", "R", ("A",)),
        ]
        transitions = {
            "r_left": {"r_ab1": 1.0},   # L's A -> B -> x
            "r_right": {"r_ab2": 1.0},  # R's A -> C -> y
        }
        g = MarkovGrammar(rules, terminals={"x", "y"}, transitions=transitions)
        left, _, _ = g.expand("L", seed=3)
        right, _, _ = g.expand("R", seed=3)
        self.assertEqual([s for s, _ in left], ["x"])
        self.assertEqual([s for s, _ in right], ["y"])


class DiagnosticsTest(unittest.TestCase):
    def test_blocked_when_table_excludes_all(self):
        rules = _grammar() + [Rule("r_root", "S", ("A",))]
        transitions = {"r_root": {"r_nonexistent": 1.0}}
        g = MarkovGrammar(rules, terminals={"x", "y"}, transitions=transitions)
        terms, _, diag = g.expand("S", seed=0)
        self.assertEqual(terms, ())
        self.assertIn("blocked:A", diag)

    def test_depth_budget(self):
        rules = [Rule("r_loop", "A", ("A",))]
        g = MarkovGrammar(rules, terminals=set())
        _, _, diag = g.expand("A", seed=0, max_depth=5)
        self.assertTrue(any(d.startswith("depth_budget") for d in diag))

    def test_duplicate_rule_id_rejected(self):
        rules = [Rule("dup", "A", ("x",)), Rule("dup", "B", ("y",))]
        with self.assertRaises(ValueError):
            MarkovGrammar(rules, terminals={"x", "y"})

    def test_transition_matrix_normalizes(self):
        rules = _grammar() + [Rule("r_root", "S", ("A",))]
        transitions = {"r_root": {"r_ab1": 3.0, "r_ab2": 1.0}}
        g = MarkovGrammar(rules, terminals={"x", "y"}, transitions=transitions)
        m = g.transition_matrix("r_root")
        self.assertAlmostEqual(m["r_ab1"], 0.75)
        self.assertAlmostEqual(m["r_ab2"], 0.25)
        self.assertEqual(g.transition_matrix("unknown"), {})


if __name__ == "__main__":
    unittest.main()
