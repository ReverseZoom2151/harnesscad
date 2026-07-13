import unittest

from harnesscad.agents.generation.cadsmith_dual_loop import (
    ExecResult, ValidationResult, run_inner_loop, run_dual_loop, Stop,
)


class TestInnerLoop(unittest.TestCase):
    def test_first_run_succeeds(self):
        rec = run_inner_loop("code", lambda c: ExecResult(True), lambda c, t, a: c)
        self.assertTrue(rec.resolved)
        self.assertEqual(rec.retries, 0)

    def test_recovers_after_retries(self):
        # Fails twice, succeeds on third run.
        calls = {"n": 0}

        def ex(code):
            calls["n"] += 1
            return ExecResult(calls["n"] >= 3, traceback="boom")

        rec = run_inner_loop("code", ex, lambda c, t, a: c + "!")
        self.assertTrue(rec.resolved)
        self.assertEqual(rec.retries, 2)

    def test_gives_up_after_max_retries(self):
        rec = run_inner_loop("code",
                             lambda c: ExecResult(False, traceback="boom"),
                             lambda c, t, a: c, max_retries=3)
        self.assertFalse(rec.resolved)
        self.assertEqual(len(rec.attempts), 4)   # 1 initial + 3 retries

    def test_error_refiner_receives_attempt_index(self):
        seen = []

        def ref(code, tb, attempt):
            seen.append(attempt)
            return code

        run_inner_loop("code", lambda c: ExecResult(False, traceback="x"), ref,
                       max_retries=2)
        self.assertEqual(seen, [0, 1])


class TestDualLoop(unittest.TestCase):
    def test_validated_first_iteration(self):
        res = run_dual_loop(
            "code",
            lambda c: ExecResult(True),
            lambda c, t, a: c,
            lambda c, e: ValidationResult(True),
            lambda c, v, i: c,
        )
        self.assertIs(res.stop, Stop.VALIDATED)
        self.assertTrue(res.passed)
        self.assertEqual(res.outer_count, 1)

    def test_exec_failure_short_circuits(self):
        res = run_dual_loop(
            "code",
            lambda c: ExecResult(False, traceback="boom"),
            lambda c, t, a: c,
            lambda c, e: ValidationResult(True),
            lambda c, v, i: c,
            max_inner_retries=1,
        )
        self.assertIs(res.stop, Stop.EXEC_FAILURE)
        self.assertIsNone(res.iterations[0].validation)

    def test_geometric_refinement_converges(self):
        # Validator passes only once the code has been refined twice.
        def validate(code, exec_res):
            return ValidationResult(code == "code!!", feedback="fix",
                                    issue_code="bbox")

        res = run_dual_loop(
            "code",
            lambda c: ExecResult(True),
            lambda c, t, a: c,
            validate,
            lambda c, v, i: c + "!",
            max_outer=5,
        )
        self.assertIs(res.stop, Stop.VALIDATED)
        self.assertEqual(res.outer_count, 3)
        self.assertEqual(res.final_code, "code!!")

    def test_max_outer_exhausted(self):
        res = run_dual_loop(
            "code",
            lambda c: ExecResult(True),
            lambda c, t, a: c,
            lambda c, e: ValidationResult(False, issue_code="bbox"),
            lambda c, v, i: c + "!",
            max_outer=5,
        )
        self.assertIs(res.stop, Stop.MAX_OUTER)
        self.assertEqual(res.outer_count, 5)

    def test_refiner_receives_iteration_index(self):
        seen = []

        def ref(code, val, i):
            seen.append(i)
            return code

        run_dual_loop(
            "code",
            lambda c: ExecResult(True),
            lambda c, t, a: c,
            lambda c, e: ValidationResult(False, issue_code="x"),
            ref,
            max_outer=3,
        )
        # Refiner runs on iterations 0 and 1 (not after the last iteration).
        self.assertEqual(seen, [0, 1])

    def test_bad_max_outer(self):
        with self.assertRaises(ValueError):
            run_dual_loop("c", lambda c: ExecResult(True), lambda c, t, a: c,
                          lambda c, e: ValidationResult(True),
                          lambda c, v, i: c, max_outer=0)


if __name__ == "__main__":
    unittest.main()
