import unittest

from agent.tool_knowledge import (
    ToolExample,
    ToolKnowledgeCard,
    ToolKnowledgeCatalog,
    default_cisp_cards,
)
from research.role_ablation import compare_role_ablation


class ToolKnowledgeCardTests(unittest.TestCase):
    def test_requires_question_for_every_context_field(self):
        with self.assertRaisesRegex(ValueError, "diameter"):
            ToolKnowledgeCard("hole", "Cut a hole", ("diameter",))

    def test_example_requires_explanation(self):
        with self.assertRaises(ValueError):
            ToolExample("", {}, "result")

    def test_conceptualization_asks_only_for_missing_context(self):
        card = ToolKnowledgeCatalog(default_cisp_cards()).get("extrude")
        concept = card.conceptualize({"sketch": "s1"})
        self.assertFalse(concept.ready)
        self.assertEqual(concept.missing, ("distance",))
        self.assertEqual(len(concept.questions), 1)
        self.assertEqual(concept.prerequisites, ("new_sketch",))

    def test_complete_context_is_ready(self):
        card = ToolKnowledgeCatalog(default_cisp_cards()).get("hole")
        concept = card.conceptualize({"diameter": 6.6, "location": (0, 0)})
        self.assertTrue(concept.ready)
        self.assertEqual(concept.questions, ())


class ToolKnowledgeCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ToolKnowledgeCatalog(default_cisp_cards())

    def test_retrieves_minimal_relevant_tools(self):
        cards = self.catalog.retrieve("drill a fastener hole in the plate", limit=2)
        self.assertEqual(tuple(card.name for card in cards), ("hole", "extrude"))

    def test_result_is_stable_and_bounded(self):
        first = self.catalog.retrieve("round edge radius on solid", limit=1)
        second = self.catalog.retrieve("round edge radius on solid", limit=1)
        self.assertEqual(first, second)
        self.assertEqual(first[0].name, "fillet")

    def test_no_overlap_returns_no_irrelevant_tools(self):
        self.assertEqual(self.catalog.retrieve("paint it blue"), ())

    def test_required_tool_precedes_scored_tools(self):
        cards = self.catalog.retrieve("drill a hole", limit=2, required_tools=("new_sketch",))
        self.assertEqual(cards[0].name, "new_sketch")
        self.assertEqual(cards[1].name, "hole")

    def test_unknown_required_tool_fails(self):
        with self.assertRaises(KeyError):
            self.catalog.retrieve("anything", required_tools=("missing",))

    def test_dispatch_collects_questions(self):
        plan = self.catalog.dispatch(
            "extrude a plate", {"sketch": "s1"}, required_tools=("extrude",), limit=1
        )
        self.assertEqual(plan.ready_tools, ())
        self.assertIn("distance", plan.questions[0].lower())

    def test_duplicate_registration_fails(self):
        card = default_cisp_cards()[0]
        catalog = ToolKnowledgeCatalog((card,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            catalog.register(card)

    def test_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            self.catalog.retrieve("hole", limit=0)


class RoleAblationTests(unittest.TestCase):
    def test_reports_sorted_metric_deltas(self):
        result = compare_role_ablation(
            {"success": .8, "validity": .9},
            {"success": .7, "validity": .9},
            "reviewer",
        )
        self.assertEqual(result.removed_role, "reviewer")
        self.assertAlmostEqual(result.deltas["success"], -.1)
        self.assertEqual(result.deltas["validity"], 0.0)
        self.assertTrue(result.harmful)

    def test_rejects_incomparable_runs(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_role_ablation({"success": 1}, {"cost": 1}, "planner")


if __name__ == "__main__":
    unittest.main()
