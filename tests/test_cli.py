"""Tests for the CLI — invokes main() in-process on the sample ops file."""

import io
import os
import unittest
from contextlib import redirect_stdout

from harnesscad.core import cli


EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "ops_plate.json",
)


class TestApplyCommand(unittest.TestCase):
    def test_apply_sample_exits_zero_and_ok(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["apply", EXAMPLE])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("ok:       True", out)
        self.assertIn("digest:", out)

    def test_missing_file_exits_two(self):
        code = cli.main(["apply", "does-not-exist.json"])
        self.assertEqual(code, 2)


class TestDemoCommand(unittest.TestCase):
    def test_demo_exits_zero_and_reports_solid(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(["demo"])
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn("ok:       True", out)
        self.assertIn("solid_present", out)


if __name__ == "__main__":
    unittest.main()
