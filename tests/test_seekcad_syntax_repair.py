import unittest

from harnesscad.domain.programs.seekcad_syntax_repair import (
    balance_parentheses,
    normalise_identifier_case,
    repair,
)


class TestBalanceParentheses(unittest.TestCase):
    def test_append_missing_close(self):
        out, fixes = balance_parentheses("f((a)")
        self.assertEqual(out, "f((a))")
        self.assertEqual(fixes, 1)

    def test_trim_extra_close(self):
        out, fixes = balance_parentheses("f(a))")
        self.assertEqual(out, "f(a)")
        self.assertEqual(fixes, 1)

    def test_balanced_untouched(self):
        out, fixes = balance_parentheses("Extrude(Sketch(p), 5)")
        self.assertEqual(out, "Extrude(Sketch(p), 5)")
        self.assertEqual(fixes, 0)

    def test_ignores_parens_in_strings(self):
        out, fixes = balance_parentheses('name = ")("')
        self.assertEqual(out, 'name = ")("')
        self.assertEqual(fixes, 0)

    def test_per_line(self):
        out, fixes = balance_parentheses("a(1\nb(2)")
        self.assertEqual(out, "a(1)\nb(2)")
        self.assertEqual(fixes, 1)


class TestNormaliseCase(unittest.TestCase):
    def test_fix_later_miscase(self):
        code = "myShape = Extrude(s, 3)\nresult = myshape.union(o)"
        out, fixes, renamed = normalise_identifier_case(code)
        self.assertIn("myShape.union", out)
        self.assertEqual(fixes, 1)
        self.assertEqual(renamed, [("myshape", "myShape")])

    def test_assignment_defines_canonical(self):
        code = "Loop = 1\nx = LOOP + loop"
        out, fixes, _ = normalise_identifier_case(code)
        self.assertEqual(out, "Loop = 1\nx = Loop + Loop")
        self.assertEqual(fixes, 2)

    def test_no_false_positive_on_correct(self):
        code = "sketch = Sketch(p)\ny = sketch.addProfile(f)"
        out, fixes, _ = normalise_identifier_case(code)
        self.assertEqual(out, code)
        self.assertEqual(fixes, 0)

    def test_unknown_identifier_untouched(self):
        code = "a = 1\nb = Something.Else"
        out, fixes, _ = normalise_identifier_case(code)
        self.assertEqual(out, code)
        self.assertEqual(fixes, 0)

    def test_string_contents_ignored(self):
        code = 'tag = 1\nx = "TAG in a string"'
        out, fixes, _ = normalise_identifier_case(code)
        self.assertEqual(out, code)
        self.assertEqual(fixes, 0)


class TestRepair(unittest.TestCase):
    def test_combined(self):
        code = "myVar = Extrude(s, 2\nz = myvar.Fillet(1)"
        rep = repair(code)
        self.assertTrue(rep.changed)
        self.assertEqual(rep.case_fixes, 1)
        self.assertEqual(rep.paren_fixes, 1)
        self.assertIn("myVar.Fillet(1)", rep.code)
        self.assertIn("Extrude(s, 2)", rep.code)

    def test_clean_code_unchanged(self):
        code = "s = Sketch(p)\nm = Extrude(s, 5)"
        rep = repair(code)
        self.assertFalse(rep.changed)
        self.assertEqual(rep.code, code)

    def test_report_fields(self):
        rep = repair("a = 1\nb = A)")
        self.assertEqual(rep.renamed, [("A", "a")])
        self.assertEqual(rep.paren_fixes, 1)


if __name__ == "__main__":
    unittest.main()
