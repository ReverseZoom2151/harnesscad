"""Tests for bench.cadtb_exec (string-based CADTEST execution harness)."""

import unittest

from bench.cadtb_exec import (
    ASSERTION_EXEC_PREAMBLE,
    BlockResult,
    CadTestOutcome,
    build_replay_script,
    execute_cadtest,
    extract_model_var_name,
    make_check,
    run_cadtest_block,
    strip_export_calls,
)


MODEL = (
    "a = 3\n"
    "widget = a + 2\n"
    'cq.exporters.export(widget, "out.stl")\n'
)


class ExtractModelVarTest(unittest.TestCase):
    def test_finds_exported_variable(self):
        self.assertEqual(extract_model_var_name(MODEL), "widget")

    def test_default_when_no_export(self):
        self.assertEqual(extract_model_var_name("x = 1\n"), "final_result")

    def test_custom_default(self):
        self.assertEqual(
            extract_model_var_name("x = 1\n", default="model"), "model")

    def test_ignores_non_cq_export(self):
        code = 'other.exporters.export(widget, "f")\n'
        self.assertEqual(extract_model_var_name(code), "final_result")

    def test_ignores_non_name_first_arg(self):
        code = 'cq.exporters.export(build(), "f")\n'
        self.assertEqual(extract_model_var_name(code), "final_result")

    def test_syntax_error_propagates(self):
        with self.assertRaises(SyntaxError):
            extract_model_var_name("def (:\n")


class StripExportTest(unittest.TestCase):
    def test_removes_export_line_only(self):
        stripped = strip_export_calls(MODEL)
        self.assertIn("widget = a + 2", stripped)
        self.assertNotIn("exporters.export", stripped)

    def test_no_export_unchanged_content(self):
        src = "a = 1\nb = 2"
        self.assertEqual(strip_export_calls(src), src)


class ExecuteCadTestTest(unittest.TestCase):
    def test_pass_records_message(self):
        env = {"final_result": 5}
        out = execute_cadtest(
            'check(final_result == 5, "five ok", "not five")', env,
            cadtest_id=7)
        self.assertTrue(out.passed)
        self.assertEqual(out.status, "pass")
        self.assertEqual(out.message, "five ok")
        self.assertIsNone(out.exception)
        self.assertEqual(out.cadtest_id, 7)

    def test_fail_records_fail_message(self):
        env = {"final_result": 4}
        out = execute_cadtest(
            'check(final_result == 5, "ok", "expected 5")', env)
        self.assertFalse(out.passed)
        self.assertEqual(out.message, "expected 5")
        self.assertEqual(out.exception, "AssertionError")

    def test_runtime_error_tagged(self):
        out = execute_cadtest("check(undefined_name, 'a', 'b')", {})
        self.assertFalse(out.passed)
        self.assertEqual(out.exception, "NameError")
        self.assertIn("NameError", out.message)

    def test_math_available_in_preamble(self):
        out = execute_cadtest(
            "check(math.isclose(1.0, 1.0), 'close', 'far')", {})
        self.assertTrue(out.passed)

    def test_preamble_defines_check(self):
        self.assertIn("def check(", ASSERTION_EXEC_PREAMBLE)


class MakeCheckTest(unittest.TestCase):
    def test_records_pass_message(self):
        rec = {}
        check = make_check(rec)
        check(True, "good", "bad")
        self.assertEqual(rec["pass_msg"], "good")

    def test_raises_on_false(self):
        check = make_check({})
        with self.assertRaises(AssertionError):
            check(False, "good", "bad")


class RunBlockTest(unittest.TestCase):
    def test_runs_model_and_cadtests(self):
        tests = [
            'check(final_result == 5, "eq5", "ne5")',
            'check(final_result > 10, "big", "too small")',
        ]
        res = run_cadtest_block(MODEL, tests)
        self.assertIsInstance(res, BlockResult)
        self.assertIsNone(res.model_error)
        self.assertFalse(res.model_failed)
        self.assertEqual(res.num_passed, 1)
        self.assertFalse(res.passed_all)
        self.assertTrue(res.outcomes[0].passed)
        self.assertFalse(res.outcomes[1].passed)

    def test_passed_all(self):
        res = run_cadtest_block(MODEL, ['check(final_result == 5, "a", "b")'])
        self.assertTrue(res.passed_all)

    def test_dict_rows_carry_ids(self):
        rows = [
            {"cadtest_id": 42, "cadtest_code": 'check(final_result == 5, "a", "b")'},
        ]
        res = run_cadtest_block(MODEL, rows)
        self.assertEqual(res.outcomes[0].cadtest_id, 42)
        self.assertTrue(res.outcomes[0].passed)

    def test_model_exec_error_fails_all(self):
        bad = "raise ValueError('boom')\ncq.exporters.export(x, 'f')\n"
        res = run_cadtest_block(bad, ["check(True, 'a', 'b')"])
        self.assertTrue(res.model_failed)
        self.assertEqual(len(res.outcomes), 1)
        self.assertFalse(res.outcomes[0].passed)
        self.assertEqual(res.outcomes[0].exception, "ModelExecError")

    def test_missing_variable_fails_all(self):
        # export references a name that the sanitized body never defines.
        src = "a = 1\ncq.exporters.export(missing, 'f')\n"
        res = run_cadtest_block(src, ["check(True, 'a', 'b')"])
        self.assertTrue(res.model_failed)
        self.assertIn("missing", res.model_error)

    def test_parse_error_fails_all(self):
        res = run_cadtest_block("def (:\n", ["check(True, 'a', 'b')"])
        self.assertTrue(res.model_failed)
        self.assertIn("parse error", res.model_error)

    def test_base_env_not_mutated(self):
        base = {"seed": 1}
        run_cadtest_block(MODEL, ['check(final_result == 5, "a", "b")'],
                          base_env=base)
        self.assertEqual(base, {"seed": 1})

    def test_stdout_captured_not_propagated(self):
        src = "print('hello')\nwidget = 1\ncq.exporters.export(widget, 'f')\n"
        res = run_cadtest_block(src, ['check(final_result == 1, "a", "b")'])
        self.assertIn("hello", res.stdout)
        self.assertTrue(res.passed_all)

    def test_determinism(self):
        tests = ['check(final_result == 5, "a", "b")']
        r1 = run_cadtest_block(MODEL, tests)
        r2 = run_cadtest_block(MODEL, tests)
        self.assertEqual(
            [o.status for o in r1.outcomes],
            [o.status for o in r2.outcomes])


class ReplayScriptTest(unittest.TestCase):
    def test_script_compiles(self):
        rows = [
            {"cadtest_id": 1, "cadtest_type": "topology",
             "cadtest_code": 'check(final_result == 5, "a", "b")'},
        ]
        script = build_replay_script(MODEL, rows)
        compile(script, "<replay>", "exec")

    def test_script_runs_and_tracks_pass(self):
        rows = [
            {"cadtest_id": 1, "cadtest_type": "topology",
             "cadtest_code": "assert final_result == 5"},
        ]
        script = build_replay_script(MODEL, rows)
        env = {}
        exec(compile(script, "<replay>", "exec"), env)
        tracker = env["cadtest_results_tracker"]
        self.assertEqual(tracker["total_test"], 1)
        self.assertEqual(tracker["passed"], [1])
        self.assertEqual(tracker["failed"], [])
        self.assertFalse(tracker["model_compile_error"])
        self.assertEqual(tracker["categories"]["topology"]["passed"], [1])

    def test_script_tracks_fail(self):
        rows = [{"cadtest_id": 2, "cadtest_type": "volumetric",
                 "cadtest_code": "assert final_result == 999"}]
        script = build_replay_script(MODEL, rows)
        env = {}
        exec(compile(script, "<replay>", "exec"), env)
        tracker = env["cadtest_results_tracker"]
        self.assertEqual(tracker["failed"], [2])
        self.assertEqual(tracker["passed"], [])

    def test_script_model_error_marks_all_failed(self):
        bad = "raise ValueError('x')\ncq.exporters.export(w, 'f')\n"
        rows = [{"cadtest_id": 3, "cadtest_type": "topology",
                 "cadtest_code": "assert True"}]
        script = build_replay_script(bad, rows)
        env = {}
        exec(compile(script, "<replay>", "exec"), env)
        tracker = env["cadtest_results_tracker"]
        self.assertTrue(tracker["model_compile_error"])
        self.assertEqual(tracker["failed"], [3])

    def test_string_cadtests_default_category(self):
        script = build_replay_script(MODEL, ["assert final_result == 5"])
        env = {}
        exec(compile(script, "<replay>", "exec"), env)
        tracker = env["cadtest_results_tracker"]
        self.assertIn("uncategorized", tracker["categories"])


if __name__ == "__main__":
    unittest.main()
