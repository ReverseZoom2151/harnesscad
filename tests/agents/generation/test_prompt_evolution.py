import unittest

from harnesscad.agents.generation.prompt_evolution import (
    REQUIRED_SCAFFOLD, build_initial_prompt, combine_terminal_log,
    refine_prompt, constraint_from_error, evolve, EvolutionResult)


class BuildInitialPromptTests(unittest.TestCase):
    def test_includes_scaffold_and_description(self):
        p = build_initial_prompt("a 50mm cube at origin")
        self.assertIn("a 50mm cube at origin", p)
        for req in REQUIRED_SCAFFOLD:
            self.assertIn(req, p)

    def test_constraints_listed(self):
        p = build_initial_prompt("box", ["fully parametric"])
        self.assertIn("fully parametric", p)

    def test_empty_description_rejected(self):
        with self.assertRaises(ValueError):
            build_initial_prompt("   ")

    def test_deterministic(self):
        self.assertEqual(build_initial_prompt("cube"), build_initial_prompt("cube"))


class TerminalLogTests(unittest.TestCase):
    def test_combines_both_streams(self):
        log = combine_terminal_log("made box", "Null shape")
        self.assertIn("STDOUT", log)
        self.assertIn("STDERR", log)
        self.assertIn("Null shape", log)

    def test_empty_streams(self):
        self.assertEqual(combine_terminal_log("", "  "), "")


class ConstraintFromErrorTests(unittest.TestCase):
    def test_unsupported_api(self):
        c = constraint_from_error("module 'Part' has no attribute 'makeGear'")
        self.assertIn("unsupported", c.lower())

    def test_null_shape(self):
        self.assertIn("non-null", constraint_from_error("Null shape").lower())

    def test_overconstraint(self):
        self.assertIn("overconstrain", constraint_from_error("overconstrained sketch").lower())

    def test_syntax(self):
        self.assertIn("valid", constraint_from_error("SyntaxError: invalid syntax").lower())

    def test_fallback(self):
        self.assertTrue(constraint_from_error("weird thing"))


class RefinePromptTests(unittest.TestCase):
    def test_contains_accumulated(self):
        p = refine_prompt("PI", "cube", "old script", "STDERR:\nNull shape",
                          ["c1", "c2"])
        self.assertIn("old script", p)
        self.assertIn("c1", p)
        self.assertIn("c2", p)


class EvolveTests(unittest.TestCase):
    def test_first_attempt_success(self):
        res = evolve("cube", lambda p: "SCRIPT", lambda s: ("ok", ""),
                     max_retries=3)
        self.assertIsInstance(res, EvolutionResult)
        self.assertTrue(res.converged)
        self.assertEqual(res.iterations, 0)
        self.assertEqual(len(res.steps), 1)

    def test_converges_after_refinements(self):
        # Fail twice, then succeed once the constraint has been accumulated.
        calls = {"n": 0}

        def gen(prompt):
            return f"script_{calls['n']}"

        def exe(script):
            calls["n"] += 1
            if calls["n"] < 3:
                return ("partial", "Null shape")
            return ("done", "")

        res = evolve("frame", gen, exe, max_retries=5)
        self.assertTrue(res.converged)
        self.assertEqual(res.iterations, 2)
        self.assertIn("ensure every operation yields a non-null shape "
                      "before the next step", res.constraints)

    def test_graceful_failure_at_max(self):
        res = evolve("gear", lambda p: "s",
                     lambda s: ("", "has no attribute 'makeGear'"),
                     max_retries=2)
        self.assertFalse(res.converged)
        self.assertEqual(res.iterations, 2)
        self.assertTrue(any("graceful failure" in m for m in res.log))

    def test_accumulated_constraints_are_unique(self):
        res = evolve("x", lambda p: "s", lambda s: ("", "Null shape"),
                     max_retries=4)
        self.assertEqual(len(res.constraints), len(set(res.constraints)))

    def test_negative_retries_rejected(self):
        with self.assertRaises(ValueError):
            evolve("x", lambda p: "s", lambda s: ("", ""), max_retries=-1)

    def test_deterministic(self):
        a = evolve("cube", lambda p: "s", lambda s: ("", ""), max_retries=2)
        b = evolve("cube", lambda p: "s", lambda s: ("", ""), max_retries=2)
        self.assertEqual(a.converged, b.converged)
        self.assertEqual(a.iterations, b.iterations)


if __name__ == "__main__":
    unittest.main()
