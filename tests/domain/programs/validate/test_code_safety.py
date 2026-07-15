"""Tests for the Text23D static CAD code-safety gate."""

import unittest

from harnesscad.domain.programs.validate import code_safety as cs


SAFE = """
import cadquery as cq
import math

def build_model():
    return cq.Workplane("XY").box(10, 10, 10)
"""


class SafeCodeTest(unittest.TestCase):
    def test_safe_passes(self):
        report = cs.check_cad_code(SAFE)
        self.assertTrue(report.ok, report.codes())

    def test_assert_does_not_raise(self):
        cs.assert_cad_code_safe(SAFE)


class UnsafeCodeTest(unittest.TestCase):
    def test_syntax_error(self):
        report = cs.check_cad_code("def build_model(:\n  pass")
        self.assertFalse(report.ok)
        self.assertIn("syntax", report.codes())

    def test_missing_entrypoint(self):
        report = cs.check_cad_code("import cadquery\nx = 1\n")
        self.assertIn("missing_entrypoint", report.codes())

    def test_blocked_import(self):
        code = "import os\ndef build_model():\n  return None\n"
        report = cs.check_cad_code(code)
        self.assertFalse(report.ok)
        self.assertIn("import_not_allowed", report.codes())

    def test_blocked_name(self):
        code = "import cadquery\ndef build_model():\n  return subprocess\n"
        report = cs.check_cad_code(code)
        self.assertIn("blocked_name", report.codes())

    def test_blocked_call(self):
        code = "import cadquery\ndef build_model():\n  return eval('1+1')\n"
        report = cs.check_cad_code(code)
        self.assertIn("blocked_call", report.codes())

    def test_blocked_module_call(self):
        code = "import cadquery\ndef build_model():\n  os.system('rm -rf /')\n"
        report = cs.check_cad_code(code)
        # os triggers both blocked_name and blocked_call.
        self.assertIn("blocked_call", report.codes())

    def test_async_forbidden(self):
        code = "import cadquery\nasync def build_model():\n  return None\n"
        report = cs.check_cad_code(code)
        self.assertIn("async_forbidden", report.codes())

    def test_assert_raises(self):
        with self.assertRaises(cs.CodeSafetyError):
            cs.assert_cad_code_safe("import os\ndef build_model():\n  pass\n")


class KernelTest(unittest.TestCase):
    def test_freecad_allows_part(self):
        code = "import Part\ndef build_model():\n  return Part\n"
        report = cs.check_cad_code(code, kernel="freecad")
        self.assertTrue(report.ok, report.codes())

    def test_freecad_rejects_cadquery(self):
        code = "import cadquery\ndef build_model():\n  return None\n"
        report = cs.check_cad_code(code, kernel="freecad")
        self.assertIn("import_not_allowed", report.codes())

    def test_build123d_allows(self):
        code = "from build123d import Box\ndef build_model():\n  return Box(1, 1, 1)\n"
        report = cs.check_cad_code(code, kernel="build123d")
        self.assertTrue(report.ok, report.codes())

    def test_no_required_def(self):
        report = cs.check_cad_code("import cadquery\nx = 1\n", required_def=None)
        self.assertTrue(report.ok)


if __name__ == "__main__":
    unittest.main()
