"""Tests for GeoCAD geometry-constrained local-edit prompts."""

import unittest

from harnesscad.domain.reconstruction.translate.hierarchy_text import (
    curve, loop, face, sketch, extrusion, se, model, LOOP_MASK, infill, tokenize,
)
from harnesscad.data.dataengine.annotation import edit_prompts as gp


def _two_loop_model():
    lp0 = loop(curve("line", 0, 0), curve("line", 1, 0), curve("line", 1, 1))
    lp1 = loop(curve("line", 2, 2), curve("line", 3, 2), curve("line", 3, 3))
    f = face(lp0, lp1)
    return model(se(sketch(f), extrusion("add", 1, 2, 3)))


class Stage2Test(unittest.TestCase):
    def test_instruction_in_prompt(self):
        m = _two_loop_model()
        p = gp.build_stage2_prompt(m, 0, 0, 1, "a right triangle")
        self.assertIn("a right triangle", p.prompt)
        self.assertEqual(p.mask_token, LOOP_MASK)
        self.assertIn(LOOP_MASK, p.prompt)

    def test_answer_roundtrips(self):
        m = _two_loop_model()
        p = gp.build_stage2_prompt(m, 0, 0, 0, "a triangle")
        # The masked CAD text + answer should reconstruct the original tokens.
        instr = tuple(p.prompt.split("CAD model: ")[1].split("\n")[0].split())
        rebuilt = infill(instr, tuple(p.answer.split()), (LOOP_MASK,))
        self.assertEqual(rebuilt, tokenize(m))

    def test_answer_is_masked_loop(self):
        m = _two_loop_model()
        p = gp.build_stage2_prompt(m, 0, 0, 1, "a square")
        # The second loop's first curve starts at (2,2).
        self.assertIn("2", p.answer.split())


class Stage1Test(unittest.TestCase):
    def test_stage1_prompt(self):
        parts = ["line 0 0 <curve_end> <loop_end>",
                 "line 5 5 <curve_end> <loop_end>"]
        p = gp.build_stage1_prompt(parts, "a right trapezoid")
        self.assertIn("a right trapezoid", p.prompt)
        self.assertIn("2", p.prompt)  # count
        self.assertEqual(len(p.answers), 2)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            gp.build_stage1_prompt([], "a square")


if __name__ == "__main__":
    unittest.main()
