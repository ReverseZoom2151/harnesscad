import unittest

from harnesscad.eval.bench.protocols.engdesign_taxonomy import (
    benchmark_taxonomy, experiment_index, total_max_score, stage_max_scores,
    aggregate_scorecard, experiment_leaders, model_win_counts,
)


class TaxonomyTest(unittest.TestCase):
    def test_total_max_matches_paper(self):
        # Table 20 maximum total is 1113.
        self.assertEqual(total_max_score(), 1113)

    def test_four_stages(self):
        self.assertEqual(len(benchmark_taxonomy()), 4)

    def test_experiment_index(self):
        idx = experiment_index()
        self.assertEqual(idx["Topology optimization"][1], 90)
        self.assertEqual(idx["Crack/defect inspection"][1], 345)

    def test_stage_max_scores_sum(self):
        self.assertEqual(sum(stage_max_scores().values()), 1113)


class ScorecardTest(unittest.TestCase):
    def setUp(self):
        # Subset of Table 20 for GPT-4V vs LLaVA (None = N/A for LLaVA).
        self.models = {
            "GPT-4V": {
                "Design description: with text": 30,
                "Design description: no text": 16,
                "Topology optimization": 68,
                "Design for additive manufacturing": -22,
                "Textbook questions": 51,
                "Spatial reasoning: rotation": 18,
            },
            "LLaVA 1.6 34B": {
                "Design description: with text": 26,
                "Design description: no text": 14,
                "Topology optimization": 43,
                "Design for additive manufacturing": None,
                "Textbook questions": 261,
                "Spatial reasoning: rotation": None,
            },
        }

    def test_totals(self):
        card = aggregate_scorecard(self.models)
        # GPT-4V: 30+16+68-22+51+18 = 161
        self.assertEqual(card["models"]["GPT-4V"]["total"], 161)
        # LLaVA (drops the two None experiments): 26+14+43+261 = 344
        self.assertEqual(card["models"]["LLaVA 1.6 34B"]["total"], 344)

    def test_applicable_max_ignores_none(self):
        card = aggregate_scorecard(self.models)
        # LLaVA skips additive(90) and rotation(100).
        self.assertEqual(card["models"]["LLaVA 1.6 34B"]["applicable_max"],
                         30 + 30 + 90 + 135)

    def test_stage_totals(self):
        card = aggregate_scorecard(self.models)
        st = card["models"]["GPT-4V"]["stage_totals"]
        self.assertEqual(st["Conceptual Design"], 46)

    def test_unknown_experiment_raises(self):
        with self.assertRaises(KeyError):
            aggregate_scorecard({"m": {"nonexistent": 1}})

    def test_leaders_and_wins(self):
        leaders = experiment_leaders(self.models)
        self.assertEqual(leaders["Textbook questions"]["leaders"],
                         ("LLaVA 1.6 34B",))
        self.assertEqual(leaders["Design description: with text"]["leaders"],
                         ("GPT-4V",))
        wins = model_win_counts(self.models)
        # GPT-4V leads with-text, no-text, topology, additive (LLaVA N/A),
        # and rotation (LLaVA N/A); LLaVA only leads textbook.
        self.assertEqual(wins["GPT-4V"], 5)
        self.assertEqual(wins["LLaVA 1.6 34B"], 1)


if __name__ == "__main__":
    unittest.main()
