"""Tests for programs.t2cq_validity (ast-based CadQuery-code validity checks)."""

import unittest

from harnesscad.domain.programs.t2cq_validity import check_code, invalid_rate, is_valid

_GOOD = (
    "import cadquery as cq\n"
    'part_1 = cq.Workplane("XY").moveTo(0.0, 0.0).lineTo(0.75, 0.0).close().extrude(0.5)\n'
    "result = part_1\n"
)


class TestValid(unittest.TestCase):
    def test_good_code_passes(self):
        self.assertEqual(check_code(_GOOD), [])
        self.assertTrue(is_valid(_GOOD))

    def test_circle_extrude_valid(self):
        code = "import cadquery as cq\nr = cq.Workplane('XY').circle(0.375).extrude(0.1)\n"
        self.assertTrue(is_valid(code))

    def test_boolean_and_transform_valid(self):
        code = (
            "import cadquery as cq\n"
            "a = cq.Workplane('XY').box(1, 1, 1)\n"
            "b = a.union(a).translate((0, 1, 0))\n"
        )
        self.assertTrue(is_valid(code))


class TestSyntax(unittest.TestCase):
    def test_syntax_error_reported(self):
        issues = check_code("import cadquery as cq\nx = cq.Workplane('XY'.box(\n")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, "syntax")

    def test_syntax_error_makes_invalid(self):
        self.assertFalse(is_valid("def (:\n"))


class TestApi(unittest.TestCase):
    def test_unknown_method(self):
        code = "import cadquery as cq\nx = cq.Workplane('XY').frobnicate(3)\n"
        issues = check_code(code)
        self.assertTrue(any(i.code == "unknown_api" for i in issues))
        self.assertFalse(is_valid(code))

    def test_bad_arity(self):
        code = "import cadquery as cq\nx = cq.Workplane('XY').circle(1, 2, 3)\n"
        issues = check_code(code)
        self.assertTrue(any(i.code == "arity" for i in issues))

    def test_arity_line_number(self):
        code = "import cadquery as cq\nx = cq.Workplane('XY').circle(1, 2)\n"
        issues = [i for i in check_code(code) if i.code == "arity"]
        self.assertEqual(issues[0].line, 2)

    def test_exporters_export_allowed(self):
        code = (
            "import cadquery as cq\n"
            "r = cq.Workplane('XY').box(1, 1, 1)\n"
            "cq.exporters.export(r, 'out.stl')\n"
        )
        self.assertTrue(is_valid(code))


class TestWorkplane(unittest.TestCase):
    def test_unknown_plane(self):
        code = "import cadquery as cq\nx = cq.Workplane('QQ').box(1, 1, 1)\n"
        self.assertTrue(any(i.code == "workplane_plane" for i in check_code(code)))

    def test_workplane_too_many_args(self):
        code = "import cadquery as cq\nx = cq.Workplane('XY', 'ZZ').box(1, 1, 1)\n"
        self.assertTrue(any(i.code == "workplane_arity" for i in check_code(code)))

    def test_workplane_no_arg_ok(self):
        code = "import cadquery as cq\nx = cq.Workplane().box(1, 1, 1)\n"
        self.assertTrue(is_valid(code))


class TestImportAndRate(unittest.TestCase):
    def test_missing_import(self):
        code = "x = cq.Workplane('XY').box(1, 1, 1)\n"
        self.assertTrue(any(i.code == "missing_import" for i in check_code(code)))

    def test_from_import_accepted(self):
        code = "from cadquery import Workplane\nx = Workplane\n"
        self.assertFalse(any(i.code == "missing_import" for i in check_code(code)))

    def test_invalid_rate(self):
        bad = "import cadquery as cq\nx = cq.Workplane('XY').nope(1)\n"
        rate = invalid_rate([_GOOD, _GOOD, bad, bad])
        self.assertAlmostEqual(rate, 0.5)

    def test_invalid_rate_empty(self):
        self.assertEqual(invalid_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
