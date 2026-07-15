"""Tests for domain.reconstruction.sequences.guidecad_prompt_template."""

import unittest

from harnesscad.domain.reconstruction.sequences.guidecad_prompt_template import (
    COMMANDS,
    PARAMETERS,
    Arc,
    Circle,
    Command,
    EOS,
    Extrusion,
    Line,
    SOL,
    command_line,
    command_lines,
    render_prompt,
)


class VocabTest(unittest.TestCase):
    def test_six_commands(self):
        self.assertEqual(len(COMMANDS), 6)

    def test_sixteen_parameters(self):
        self.assertEqual(len(PARAMETERS), 16)

    def test_bad_command_rejected(self):
        with self.assertRaises(ValueError):
            Command("Bogus")

    def test_bad_param_rejected(self):
        with self.assertRaises(ValueError):
            Command("Line", {"zzz": 1.0})


class CommandLineTest(unittest.TestCase):
    def test_sol_and_eos(self):
        self.assertEqual(command_line(SOL()), "<SOL>")
        self.assertEqual(command_line(EOS()), "<EOS>")

    def test_line_canonical_order(self):
        self.assertEqual(command_line(Line(3, 4)), "Line, x=3, y=4")

    def test_circle(self):
        self.assertEqual(command_line(Circle(1, 2, 5)), "Circle, x=1, y=2, r=5")

    def test_command_lines_count(self):
        seq = [SOL(), Line(1, 1), Line(2, 2), EOS()]
        self.assertEqual(len(command_lines(seq)), 4)


class RenderPromptTest(unittest.TestCase):
    def setUp(self):
        self.seq = [
            SOL(),
            Line(0, 10),
            Arc(10, 10, 90, 1),
            Circle(5, 5, 2),
            Extrusion(0, 0, 1, 0, 0, 0, 1.0, 5.0, 0.0, 0, 2),
            EOS(),
        ]

    def test_numbered(self):
        out = render_prompt(self.seq, numbered=True)
        self.assertTrue(out.startswith("1. Start a new closed loop."))
        self.assertEqual(len(out.splitlines()), 6)

    def test_paragraph(self):
        out = render_prompt(self.seq, numbered=False)
        self.assertNotIn("\n", out)
        self.assertIn("radius 2", out)

    def test_arc_ccw_wording(self):
        self.assertIn("counter-clockwise", render_prompt([Arc(1, 1, 45, 1)]))
        self.assertIn("clockwise", render_prompt([Arc(1, 1, 45, 0)]))

    def test_extrusion_merge_name(self):
        out = render_prompt([Extrusion(0, 0, 1, 0, 0, 0, 1, 5, 0, 0, 2)])
        self.assertIn("cutting", out)  # w=2 -> cutting

    def test_determinism(self):
        self.assertEqual(render_prompt(self.seq), render_prompt(self.seq))


if __name__ == "__main__":
    unittest.main()
