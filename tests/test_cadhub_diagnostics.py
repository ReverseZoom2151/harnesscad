"""Tests for programs.cadhub_diagnostics (multi-language diagnostic parser)."""

import unittest

from harnesscad.domain.programs.validate.cadhub_diagnostics import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    UnknownDialect,
    annotate_source,
    first_error,
    is_success,
    parse,
    scrub_paths,
    sorted_diagnostics,
    summarize,
)


class TestScrubPaths(unittest.TestCase):
    def test_tmp_path_replaced_by_basename(self):
        text = "ERROR: syntax error in file \"/tmp/aX9k/main.scad\", line 3"
        self.assertEqual(
            scrub_paths(text),
            'ERROR: syntax error in file "main.scad", line 3',
        )

    def test_unquoted_and_windows_paths(self):
        self.assertEqual(scrub_paths("in file /tmp/abc/main.scad, line 2"), "in file main.scad, line 2")
        self.assertIn("main.py", scrub_paths('File "C:\\tmp\\z\\main.py", line 1'))

    def test_idempotent(self):
        once = scrub_paths("in file /tmp/a/main.curv:1(2)")
        self.assertEqual(scrub_paths(once), once)


class TestOpenScad(unittest.TestCase):
    def test_error_with_location(self):
        text = (
            "Compiling design (CSG Tree generation)...\n"
            "ERROR: Parser error: syntax error in file /tmp/a1/main.scad, line 12\n"
        )
        diags = parse("openscad", text)
        self.assertEqual(len(diags), 1)
        d = diags[0]
        self.assertEqual(d.severity, SEVERITY_ERROR)
        self.assertEqual(d.file, "main.scad")
        self.assertEqual(d.line, 12)
        self.assertEqual(d.message, "Parser error: syntax error")
        self.assertEqual(d.location(), "main.scad:12")

    def test_warning_and_success(self):
        text = "WARNING: Ignoring unknown module 'foo' in file main.scad, line 5\n"
        diags = parse("openscad", text)
        self.assertEqual(diags[0].severity, SEVERITY_WARNING)
        self.assertEqual(diags[0].line, 5)
        self.assertTrue(is_success(diags))

    def test_no_location(self):
        diags = parse("openscad", "ERROR: Current top level object is empty.\n")
        self.assertEqual(diags[0].line, None)
        self.assertEqual(diags[0].location(), "?")
        self.assertFalse(is_success(diags))


class TestCadQuery(unittest.TestCase):
    def test_runtime_traceback(self):
        text = (
            "Traceback (most recent call last):\n"
            '  File "/var/task/cq-cli.py", line 90, in <module>\n'
            "    exec(code)\n"
            '  File "/tmp/aB/main.py", line 5, in <module>\n'
            "    result = cq.Workplane().box(w, h, d)\n"
            "NameError: name 'd' is not defined\n"
        )
        diags = parse("cadquery", text)
        self.assertEqual(len(diags), 1)
        d = diags[0]
        self.assertEqual(d.file, "main.py")
        self.assertEqual(d.line, 5)
        self.assertEqual(d.severity, SEVERITY_ERROR)
        self.assertIn("NameError", d.message)

    def test_syntax_error_column(self):
        text = (
            '  File "/tmp/aB/main.py", line 3\n'
            "    result = cq.Workplane(\n"
            "                        ^\n"
            "SyntaxError: unexpected EOF while parsing\n"
        )
        d = parse("cadquery", text)[0]
        self.assertEqual((d.line, d.column), (3, 25))
        self.assertEqual(d.location(), "main.py:3:25")

    def test_clean_output(self):
        self.assertEqual(parse("cadquery", "wrote /tmp/a/out.stl\n"), [])


class TestJsCad(unittest.TestCase):
    def test_error_with_stack_frame(self):
        text = (
            "ReferenceError: cube is not defined\n"
            "    at main (jscad_script:7:11)\n"
            "    at Object.eval (worker.js:22:3)\n"
        )
        d = parse("jscad", text)[0]
        self.assertEqual(d.message, "ReferenceError: cube is not defined")
        self.assertEqual((d.file, d.line, d.column), ("jscad_script", 7, 11))

    def test_bare_frame_only(self):
        d = parse("jscad", "at jscad_script:2:4\n")[0]
        self.assertEqual(d.line, 2)
        self.assertEqual(d.severity, SEVERITY_ERROR)

    def test_empty(self):
        self.assertEqual(parse("jscad", ""), [])


class TestCurv(unittest.TestCase):
    def test_error_with_location(self):
        text = (
            'ERROR: a: not defined\n'
            'at file "/tmp/aQ/main.curv":3(5)-3(6)\n'
            "  3| a >> colour red;\n"
        )
        d = parse("curv", text)[0]
        self.assertEqual(d.message, "a: not defined")
        self.assertEqual((d.file, d.line, d.column), ("main.curv", 3, 5))

    def test_two_messages(self):
        text = (
            "WARNING: deprecated\n"
            'at file "main.curv":1(1)\n'
            "ERROR: syntax error\n"
            'at file "main.curv":9(2)\n'
        )
        diags = parse("curv", text)
        self.assertEqual(len(diags), 2)
        self.assertEqual(diags[0].severity, SEVERITY_WARNING)
        self.assertEqual(diags[1].line, 9)


class TestFacade(unittest.TestCase):
    def test_unknown_dialect(self):
        with self.assertRaises(UnknownDialect):
            parse("solidworks", "boom")

    def test_summary_and_sorting(self):
        text = (
            "WARNING: slow in file main.scad, line 2\n"
            "ERROR: bad in file main.scad, line 40\n"
            "ERROR: worse in file main.scad, line 8\n"
        )
        diags = parse("openscad", text)
        ordered = sorted_diagnostics(diags)
        self.assertEqual([d.line for d in ordered], [8, 40, 2])
        summary = summarize(diags)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["warnings"], 1)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["first_error"]["line"], 8)
        self.assertEqual(first_error(ordered).message, "worse")

    def test_summary_clean(self):
        summary = summarize(parse("openscad", "Compiling...\n"))
        self.assertTrue(summary["ok"])
        self.assertIsNone(summary["first_error"])


class TestAnnotate(unittest.TestCase):
    def test_caret_and_context(self):
        source = "a = 1;\nb = ;\nc = 3;\n"
        d = parse("openscad", "ERROR: syntax error in file main.scad, line 2")[0]
        d = type(d)(
            language=d.language,
            severity=d.severity,
            message=d.message,
            file=d.file,
            line=d.line,
            column=5,
        )
        text = annotate_source(source, d)
        self.assertIn("2| b = ;", text)
        self.assertIn("^", text)
        self.assertEqual(len(text.splitlines()), 4)

    def test_out_of_range(self):
        d = parse("openscad", "ERROR: e in file main.scad, line 99")[0]
        self.assertEqual(annotate_source("a=1;\n", d), "")


if __name__ == "__main__":
    unittest.main()
