"""The recipe is a declaration. It must not import a trainer, and it must not
let anyone start at the expensive end."""

from __future__ import annotations

import unittest

from harnesscad.agents.selftrain import recipe


class TestRecipe(unittest.TestCase):

    def test_the_order_is_rft_kto_dpo_grpo(self):
        self.assertEqual([s.name for s in recipe.RECIPE],
                         ["RFT / STaR", "KTO", "DPO (Robust)", "GRPO"])
        self.assertEqual([s.order for s in recipe.RECIPE], [1, 2, 3, 4])

    def test_grpo_is_last_and_says_do_not_start_here(self):
        grpo = recipe.RECIPE[-1]
        self.assertIn("DO NOT START HERE", grpo.stop_condition)
        # And it is 300x the cost of the first three combined.
        cheap = recipe.total_cost(through=3)
        self.assertGreater(grpo.usd, 100 * cheap["usd"] / 3)

    def test_every_stage_has_a_stop_condition(self):
        for s in recipe.RECIPE:
            self.assertTrue(s.stop_condition, s.name)
            self.assertTrue(s.rationale, s.name)

    def test_stages_1_to_3_are_a_few_gpu_hours(self):
        c = recipe.total_cost(through=3)
        self.assertEqual(c["stages"], 3.0)
        self.assertLessEqual(c["gpu_hours"], 8.0)
        self.assertLessEqual(c["usd"], 50.0)

    def test_the_prerequisites_are_all_cheaper_than_a_gpu_hour(self):
        # Four of them, and every one is a lesson from the pressure run.
        self.assertEqual(len(recipe.PREREQUISITES), 4)
        joined = " ".join(recipe.PREREQUISITES)
        self.assertIn("false_positive_rate == 0", joined)
        self.assertIn("HELD-OUT", joined)
        self.assertIn("Best-of-N", joined)

    def test_it_imports_no_trainer(self):
        # pyproject.toml says dependencies = [] and this module must not change
        # that. A recipe that needs torch to be READ is not a recipe.
        import sys
        before = set(sys.modules)
        recipe.format_recipe()
        new = set(sys.modules) - before
        for banned in ("torch", "trl", "peft", "transformers"):
            self.assertNotIn(banned, new)

    def test_format_recipe_renders(self):
        text = recipe.format_recipe()
        self.assertIn("STAGE 1", text)
        self.assertIn("QLoRA", text)


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
