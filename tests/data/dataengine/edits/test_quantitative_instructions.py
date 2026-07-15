"""Tests for data.dataengine.edits.quantitative_instructions."""

import unittest

from harnesscad.data.dataengine.edits.quantitative_instructions import (
    RefinementSession,
    apply_instruction,
    parse_instruction,
)


class ParseTest(unittest.TestCase):
    def test_reduce_by(self):
        inst = parse_instruction("reduce the radius by 6mm")
        self.assertEqual(inst.edit_type, "modify")
        self.assertEqual(inst.target, "radius")
        self.assertEqual(inst.operation, "reduce")
        self.assertEqual(inst.amount, 6.0)
        self.assertEqual(inst.unit, "mm")
        self.assertFalse(inst.qualitative)

    def test_increase_to_is_set(self):
        inst = parse_instruction("increase the height to 20mm")
        self.assertEqual(inst.operation, "set")
        self.assertEqual(inst.amount, 20.0)

    def test_qualitative(self):
        inst = parse_instruction("make the base thicker")
        self.assertTrue(inst.qualitative)
        self.assertEqual(inst.target, "base")
        self.assertIsNone(inst.amount)

    def test_add(self):
        self.assertEqual(parse_instruction("add a hole").edit_type, "add")

    def test_delete(self):
        self.assertEqual(parse_instruction("remove the fillet").edit_type, "delete")


class ApplyTest(unittest.TestCase):
    def test_reduce(self):
        out = apply_instruction({"radius": 20.0}, parse_instruction("reduce the radius by 6mm"))
        self.assertEqual(out["radius"], 14.0)

    def test_set(self):
        out = apply_instruction({"height": 5.0}, parse_instruction("set the height to 12"))
        self.assertEqual(out["height"], 12.0)

    def test_qualitative_not_applicable(self):
        with self.assertRaises(ValueError):
            apply_instruction({"base": 1.0}, parse_instruction("make the base thicker"))

    def test_unknown_target(self):
        with self.assertRaises(ValueError):
            apply_instruction({"height": 1.0}, parse_instruction("reduce the radius by 2mm"))


class SessionTest(unittest.TestCase):
    def test_progressive(self):
        s = RefinementSession({"radius": 20.0, "height": 10.0})
        s.refine("reduce the radius by 5mm")
        s.refine("increase the height by 3mm")
        self.assertEqual(s.params["radius"], 15.0)
        self.assertEqual(s.params["height"], 13.0)
        self.assertEqual(s.num_steps(), 2)


if __name__ == "__main__":
    unittest.main()
