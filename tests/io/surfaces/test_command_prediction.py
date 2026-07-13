import math
import unittest

from harnesscad.io.surfaces.command_prediction import (
    BayesianCommandPredictor,
    Prediction,
    WorkflowGraph,
)


class TestWorkflowGraph(unittest.TestCase):
    def test_counts_transitions(self):
        g = WorkflowGraph()
        g.add_sequence(["new-sketch", "circle", "extrude"])
        g.add_sequence(["new-sketch", "rectangle", "extrude"])
        self.assertEqual(g.transition_count("new-sketch", "circle"), 1)
        self.assertEqual(g.transition_count("new-sketch", "rectangle"), 1)
        self.assertEqual(g.context_total("new-sketch"), 2)
        self.assertEqual(g.transition_count("START", "new-sketch"), 2)

    def test_vocabulary_sorted(self):
        g = WorkflowGraph()
        g.add_sequence(["extrude", "circle", "new-sketch"])
        self.assertEqual(g.vocabulary, ("circle", "extrude", "new-sketch"))

    def test_edges_deterministic(self):
        g = WorkflowGraph()
        g.add_sequence(["a", "b"])
        g.add_sequence(["a", "c"])
        self.assertEqual(
            g.edges(),
            (("START", "a", 2), ("a", "b", 1), ("a", "c", 1)),
        )

    def test_rejects_empty_command(self):
        with self.assertRaises(ValueError):
            WorkflowGraph().add_sequence(["ok", ""])


class TestBayesianPredictor(unittest.TestCase):
    def setUp(self):
        self.g = WorkflowGraph()
        # circle follows new-sketch 3x, rectangle 1x
        for _ in range(3):
            self.g.add_sequence(["new-sketch", "circle", "extrude"])
        self.g.add_sequence(["new-sketch", "rectangle", "extrude"])
        self.p = BayesianCommandPredictor(self.g, alpha=1.0)

    def test_probabilities_sum_to_one(self):
        dist = self.p.distribution("new-sketch")
        self.assertAlmostEqual(sum(pr.probability for pr in dist), 1.0)

    def test_add_alpha_formula(self):
        # vocab = {circle, extrude, new-sketch, rectangle} -> V=4
        # n(new-sketch, circle)=3, N=4, alpha=1 -> (3+1)/(4+4)=0.5
        self.assertAlmostEqual(self.p.probability("new-sketch", "circle"), 0.5)
        # rectangle: (1+1)/(4+4)=0.25
        self.assertAlmostEqual(self.p.probability("new-sketch", "rectangle"), 0.25)
        # unseen-in-context extrude: (0+1)/(4+4)=0.125
        self.assertAlmostEqual(self.p.probability("new-sketch", "extrude"), 0.125)

    def test_predict_next_ranking(self):
        preds = self.p.predict_next(["new-sketch"], top_k=1)
        self.assertEqual(preds[0].command, "circle")

    def test_most_likely_first_command(self):
        # START -> new-sketch always
        self.assertEqual(self.p.most_likely([]), "new-sketch")

    def test_unseen_context_is_uniform_prior(self):
        dist = self.p.distribution("never-seen-ctx")
        probs = {pr.command: pr.probability for pr in dist}
        # uniform over 4 vocab items
        for v in probs.values():
            self.assertAlmostEqual(v, 0.25)

    def test_unknown_command_probability_zero(self):
        self.assertEqual(self.p.probability("new-sketch", "not-a-command"), 0.0)

    def test_lexical_tie_break(self):
        g = WorkflowGraph()
        g.add_sequence(["x", "b"])
        g.add_sequence(["x", "a"])  # a and b both count 1 -> tie
        p = BayesianCommandPredictor(g)
        preds = p.predict_next(["x"])
        # equal probability -> lexical order 'a' before 'b'
        self.assertEqual(preds[0].command, "a")

    def test_alpha_must_be_positive(self):
        with self.assertRaises(ValueError):
            BayesianCommandPredictor(self.g, alpha=0.0)

    def test_empty_graph_returns_empty(self):
        p = BayesianCommandPredictor(WorkflowGraph())
        self.assertEqual(p.predict_next([]), ())
        self.assertIsNone(p.most_likely([]))


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.g = WorkflowGraph()
        for _ in range(3):
            self.g.add_sequence(["a", "b", "c"])
        self.p = BayesianCommandPredictor(self.g, alpha=1.0)

    def test_log_likelihood_matches_manual(self):
        ll = self.p.sequence_log_likelihood(["a", "b"])
        expected = math.log(self.p.probability("START", "a")) + math.log(
            self.p.probability("a", "b")
        )
        self.assertAlmostEqual(ll, expected)

    def test_perplexity_positive(self):
        pp = self.p.perplexity(["a", "b", "c"])
        self.assertGreater(pp, 0.0)
        self.assertNotEqual(pp, float("inf"))

    def test_perplexity_empty_is_one(self):
        self.assertEqual(self.p.perplexity([]), 1.0)

    def test_determinism(self):
        self.assertEqual(
            self.p.distribution("a"), self.p.distribution("a")
        )


if __name__ == "__main__":
    unittest.main()
