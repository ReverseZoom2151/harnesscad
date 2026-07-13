import unittest

from harnesscad.agents.generation.verify_loop import (
    repair_until_compiles, filter_feedback, answer_accuracy, run_cadcodeverify,
)


class TestRepairLoop(unittest.TestCase):
    def test_compiles_first_try(self):
        r = repair_until_compiles("code", lambda c: (True, ""), lambda c, e: c, max_iters=3)
        self.assertTrue(r["compiled"])
        self.assertEqual(r["repair_attempts"], 0)

    def test_recovers_after_two_fixes(self):
        state = {"n": 0}

        def comp(c):
            return (state["n"] >= 2, "err")

        def fix(c, e):
            state["n"] += 1
            return c + "#"

        r = repair_until_compiles("x", comp, fix, max_iters=5)
        self.assertTrue(r["compiled"])
        self.assertEqual(r["repair_attempts"], 2)
        self.assertEqual(r["code"], "x##")

    def test_gives_up(self):
        r = repair_until_compiles("x", lambda c: (False, "e"), lambda c, e: c, max_iters=2)
        self.assertFalse(r["compiled"])
        self.assertEqual(r["repair_attempts"], 2)
        self.assertEqual(r["last_error"], "e")

    def test_bad_iters(self):
        with self.assertRaises(ValueError):
            repair_until_compiles("x", lambda c: (True, ""), lambda c, e: c, max_iters=0)


class TestFilterFeedback(unittest.TestCase):
    def test_drops_yes(self):
        f = filter_feedback(["q1", "q2", "q3"], ["Yes", "No", "Unclear"])
        self.assertTrue(f["needs_refinement"])
        self.assertEqual([q for q, _ in f["unresolved"]], ["q2", "q3"])
        self.assertEqual(f["num_yes"], 1)

    def test_all_yes_no_refinement(self):
        f = filter_feedback(["q1", "q2"], ["yes", "YES"])
        self.assertFalse(f["needs_refinement"])
        self.assertEqual(f["unresolved"], [])

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            filter_feedback(["q1"], ["Yes", "No"])

    def test_bad_label(self):
        with self.assertRaises(ValueError):
            filter_feedback(["q1"], ["maybe"])


class TestAnswerAccuracy(unittest.TestCase):
    def test_accuracy_over_labeled(self):
        # 3 labeled (Yes/No), 1 unclear. 2 of 3 correct.
        acc = answer_accuracy(["Yes", "No", "No", "Unclear"],
                              [True, True, False, False])
        self.assertEqual(acc["total"], 4)
        self.assertEqual(acc["labeled"], 3)
        self.assertAlmostEqual(acc["accuracy_over_labeled"], 2 / 3)
        self.assertAlmostEqual(acc["unclear_fraction"], 0.25)
        self.assertAlmostEqual(acc["incorrect_fraction"], 0.25)

    def test_all_unclear(self):
        acc = answer_accuracy(["Unclear", "Unclear"], [False, False])
        self.assertIsNone(acc["accuracy_over_labeled"])
        self.assertEqual(acc["unclear_fraction"], 1.0)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            answer_accuracy(["Yes"], [True, False])


class TestRunLoop(unittest.TestCase):
    def test_stops_when_all_yes(self):
        r = run_cadcodeverify(
            "code", "desc",
            question_fn=lambda d: ["q1"],
            answer_fn=lambda d, q: ["Yes"],
            feedback_fn=lambda pairs: "fb",
            refine_fn=lambda c, d, f: c + "!",
            max_refinements=2,
        )
        self.assertEqual(r["rounds"], 1)
        self.assertFalse(r["history"][0]["refined"])
        self.assertEqual(r["code"], "code")

    def test_refines_until_limit(self):
        r = run_cadcodeverify(
            "code", "desc",
            question_fn=lambda d: ["q1"],
            answer_fn=lambda d, q: ["No"],
            feedback_fn=lambda pairs: "fix it",
            refine_fn=lambda c, d, f: c + "!",
            max_refinements=2,
        )
        self.assertEqual(r["rounds"], 2)
        self.assertEqual(r["code"], "code!!")
        self.assertTrue(all(h["refined"] for h in r["history"]))

    def test_negative_refinements(self):
        with self.assertRaises(ValueError):
            run_cadcodeverify("c", "d", question_fn=lambda d: [],
                              answer_fn=lambda d, q: [], feedback_fn=lambda p: "",
                              refine_fn=lambda c, d, f: c, max_refinements=-1)


if __name__ == "__main__":
    unittest.main()
