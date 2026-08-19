"""The scad_grammar dialect gate: refuse RapCAD instead of mis-diagnosing it.

This module implements OpenSCAD only (transliterated from RapCAD's
``doc/openscad.bnf``). RapCAD's own, larger grammar (``doc/rapcad.bnf``) was
never ported, so ``.rcad`` input must be refused by name rather than reported as
a heap of invented syntax errors.

Every snippet here is written from scratch; no RapCAD (GPL-3) file content is
reproduced or vendored.
"""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from harnesscad.domain.programs.validate.scad_grammar import (
    RAPCAD_MARKERS,
    Rule,
    detect_rapcad,
    main,
    validate,
)

PM = chr(0x00B1)


RAPCAD_ONLY = [
    ("return", "function f(x) { return x * 2; }"),
    ("const", "const width = 10;"),
    ("param", "param depth = 4;"),
    ("namespace ::", "x = lib::helper(1);"),
    ("append ~=", "xs ~= [1, 2];"),
    ("exponent ^", "x = 2 ^ 8;"),
    ("cross **", "n = a ** b;"),
    ("componentwise .*", "v = a .* b;"),
    ("componentwise ./", "v = a ./ b;"),
    ("tolerance operator", "d = 10 " + PM + " 0.1;"),
    ("interval literal", "d = 10[0.1, 0.2];"),
    ("doc comment", "/** @param r radius */\nmodule ring(r) { }"),
]

NOT_RAPCAD = [
    ("relative include path", "include <./lib/util.scad>\ncube(1);"),
    ("relative use path", "use <../shared/gears.scad>\ncube(1);"),
    ("markers inside a string", 'echo("return const param :: ^ ~= .*");'),
    ("markers in a line comment", "// return const ^ ::\ncube(1);"),
    ("markers in a block comment", "/* param ~= .* */\ncube(1);"),
    ("identifiers containing markers", "constant = 1;\nreturns = 2;"),
    ("ordinary indexing", "x = v[0];"),
    ("plain openscad", "translate([1, 0, 0]) cube([1, 2, 3], center = true);"),
]


class ScadGrammarDialectTest(unittest.TestCase):
    def test_rapcad_only_constructs_are_refused_not_mis_diagnosed(self):
        for label, source in RAPCAD_ONLY:
            with self.subTest(label=label):
                result = validate(source)
                self.assertFalse(result.ok)
                # exactly one diagnostic, and it names the dialect -- not a pile
                # of invented statement-level syntax errors
                self.assertEqual(len(result.diagnostics), 1)
                diag = result.diagnostics[0]
                self.assertEqual(diag.rule, Rule.DIALECT.value)
                self.assertIn("RapCAD", diag.found)
                self.assertIn("rapcad_language", diag.expected)

    def test_lookalikes_do_not_trip_the_dialect_gate(self):
        for label, source in NOT_RAPCAD:
            with self.subTest(label=label):
                self.assertIsNone(detect_rapcad(source))
                result = validate(source)
                self.assertFalse(
                    any(d.rule == Rule.DIALECT.value for d in result.diagnostics))

    def test_rcad_extension_alone_is_enough(self):
        # text that is perfectly good OpenSCAD, but the file says it is RapCAD
        self.assertTrue(validate("cube(1);").ok)
        refused = validate("cube(1);", path="parts/bracket.rcad")
        self.assertFalse(refused.ok)
        self.assertEqual(refused.diagnostics[0].rule, Rule.DIALECT.value)

    def test_extension_matching_is_case_insensitive_and_suffix_anchored(self):
        self.assertEqual(
            validate("cube(1);", path="A.RCAD").diagnostics[0].rule,
            Rule.DIALECT.value)
        # ".rcad" in the middle of a name is not the extension
        self.assertTrue(validate("cube(1);", path="rcad_examples/a.scad").ok)

    def test_path_defaults_to_none_so_the_old_call_still_works(self):
        self.assertTrue(validate("cube(1);").ok)

    def test_refusal_reports_the_earliest_marker_position(self):
        source = "cube(1);\nx = 2 ^ 8;\ny = a ** b;\n"
        diag = detect_rapcad(source)
        self.assertIsNotNone(diag)
        self.assertEqual(diag.line, 2)
        self.assertIn("^", diag.found)

    def test_detect_rapcad_returns_none_for_clean_openscad(self):
        self.assertIsNone(
            detect_rapcad("module a(n = 3) { for (i = [0 : n]) cube(i); }"))

    def test_marker_table_is_non_empty_and_described(self):
        self.assertTrue(RAPCAD_MARKERS)
        for text, what in RAPCAD_MARKERS:
            self.assertTrue(text)
            self.assertIn("RapCAD", what)

    def test_unterminated_string_or_comment_does_not_hang_the_scanner(self):
        self.assertIsNone(detect_rapcad('echo("unterminated'))
        self.assertIsNone(detect_rapcad("/* unterminated const"))
        # a doc comment is still caught even when unterminated
        self.assertIsNotNone(detect_rapcad("/** unterminated"))

    def test_selfcheck_exits_zero(self):
        self.assertEqual(main(["--selfcheck"]), 0)

    def test_cli_refuses_an_rcad_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "bracket.rcad"
            target.write_text("cube(1);\n", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main([str(target)]), 1)
            self.assertIn("RapCAD", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
