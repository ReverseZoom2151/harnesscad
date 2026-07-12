"""Tests for programs.t2cq_analysis (execution-free static analysis)."""

import unittest

from programs.t2cq_analysis import analyze, is_safe

_CLEAN = (
    "import cadquery as cq\n"
    "part_1 = cq.Workplane('XY').box(1.0, 1.0, 1.0)\n"
    "part_2 = part_1.union(part_1).translate((0, 1, 0))\n"
)


class TestClean(unittest.TestCase):
    def test_clean_program_ok(self):
        report = analyze(_CLEAN)
        self.assertTrue(report.ok)
        self.assertEqual(report.errors, [])
        self.assertTrue(is_safe(_CLEAN))


class TestSafety(unittest.TestCase):
    def test_forbidden_name(self):
        code = "import cadquery as cq\nx = eval('1+1')\n"
        report = analyze(code)
        self.assertTrue(report.has_code("unsafe"))
        self.assertFalse(report.ok)

    def test_forbidden_attribute(self):
        code = "import cadquery as cq\nimport os\nx = os.system('ls')\n"
        # os is forbidden both as import target name and attribute usage.
        self.assertFalse(is_safe(code))

    def test_subprocess_flagged(self):
        code = "import cadquery as cq\ny = subprocess\n"
        self.assertTrue(analyze(code).has_code("unsafe"))


class TestWorkplaneRooting(unittest.TestCase):
    def test_missing_workplane_root(self):
        # A geometry chain rooted at a bare undefined name -> undefined_var error.
        code = "import cadquery as cq\nx = foo.box(1, 1, 1)\n"
        report = analyze(code)
        self.assertFalse(report.ok)
        self.assertTrue(report.has_code("undefined_var"))

    def test_defined_var_root_ok(self):
        code = (
            "import cadquery as cq\n"
            "a = cq.Workplane('XY').box(1, 1, 1)\n"
            "b = a.fillet(0.1)\n"
        )
        self.assertTrue(analyze(code).ok)

    def test_undefined_boolean_operand(self):
        code = (
            "import cadquery as cq\n"
            "a = cq.Workplane('XY').box(1, 1, 1)\n"
            "b = a.union(ghost)\n"
        )
        report = analyze(code)
        self.assertTrue(report.has_code("undefined_var"))


class TestDegenerateArc(unittest.TestCase):
    def test_degenerate_arc_warns(self):
        # Reproduces the Appendix A.4 failure case: tiny, near-colinear arc points.
        code = (
            "import cadquery as cq\n"
            "s = 0.75\n"
            "part_1 = (cq.Workplane('XY')\n"
            "  .moveTo(0.0, 0.0)\n"
            "  .lineTo(0.0156, 0.0)\n"
            "  .threePointArc((0.0078, 0.0078), (0.0, 0.0156))\n"
            "  .close()\n"
            "  .extrude(0.04))\n"
        )
        report = analyze(code)
        self.assertTrue(report.has_code("degenerate_arc"))
        # It is a warning, not an error -- code still "ok" (executable-ish).
        self.assertTrue(all(f.severity == "warning" for f in report.findings
                            if f.code == "degenerate_arc"))

    def test_healthy_arc_no_warning(self):
        code = (
            "import cadquery as cq\n"
            "part_1 = (cq.Workplane('XY')\n"
            "  .moveTo(0.0, 0.0)\n"
            "  .lineTo(1.0, 0.0)\n"
            "  .threePointArc((0.5, 0.5), (0.0, 1.0))\n"
            "  .close()\n"
            "  .extrude(0.5))\n"
        )
        self.assertFalse(analyze(code).has_code("degenerate_arc"))


class TestSyntax(unittest.TestCase):
    def test_syntax_error(self):
        report = analyze("x = cq.Workplane('XY'.box(\n")
        self.assertTrue(report.has_code("syntax"))
        self.assertFalse(report.ok)


if __name__ == "__main__":
    unittest.main()
