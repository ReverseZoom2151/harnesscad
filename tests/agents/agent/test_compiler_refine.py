import unittest

from harnesscad.agents.agent.compiler_refine import (
    build_refine_prompt,
    run_refine_loop,
)
from harnesscad.eval.judge.compiler_review import review_sequence

BAD = [{"type": "extrude", "depth": 1.0}, {"type": "end"}]  # extrude with no sketch
GOOD = [
    {"type": "sketch", "loops": [{"points": [[0, 0], [1, 0], [1, 1], [0, 1]]}]},
    {"type": "extrude", "depth": 1.0, "boolean": "union"},
    {"type": "end"},
]


class CompilerRefineTests(unittest.TestCase):
    def test_refines_bad_to_good(self):
        outputs = iter([BAD, GOOD])

        def generate(prompt):
            return next(outputs)

        res = run_refine_loop("make a box", generate, max_iters=1)
        self.assertTrue(res.ok)
        self.assertEqual(res.iters, 1)
        self.assertEqual(len(res.history), 2)
        self.assertEqual(list(res.sequence), GOOD)

    def test_vanilla_zero_iters_returns_first(self):
        def generate(prompt):
            return BAD

        res = run_refine_loop("x", generate, max_iters=0)
        self.assertFalse(res.ok)
        self.assertEqual(res.iters, 0)
        self.assertEqual(len(res.history), 1)

    def test_first_pass_success_no_refine(self):
        def generate(prompt):
            return GOOD

        res = run_refine_loop("x", generate, max_iters=3)
        self.assertTrue(res.ok)
        self.assertEqual(res.iters, 0)

    def test_refine_prompt_contains_feedback(self):
        review = review_sequence(BAD)
        prompt = build_refine_prompt("base task", review)
        self.assertIn("base task", prompt)
        self.assertIn("Compiler feedback", prompt)

    def test_generate_receives_feedback_prompt(self):
        seen = []
        outputs = iter([BAD, GOOD])

        def generate(prompt):
            seen.append(prompt)
            return next(outputs)

        run_refine_loop("make a box", generate, max_iters=1)
        self.assertIn("Compiler feedback", seen[1])


if __name__ == "__main__":
    unittest.main()
