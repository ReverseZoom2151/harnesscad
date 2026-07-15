"""Tests for the CadAgent deterministic code-repair rules."""

import unittest

from harnesscad.agents.agent import code_repair_rules as crr


class PrecheckTest(unittest.TestCase):
    def test_valid_code_passes(self):
        ok, msg = crr.precheck_syntax("x = 1 + 2\n")
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_syntax_error_points_at_line(self):
        ok, msg = crr.precheck_syntax("def f(:\n    pass\n")
        self.assertFalse(ok)
        self.assertIn("SyntaxError", msg)


class MissingPartPrefixTest(unittest.TestCase):
    def test_makebox_autofix(self):
        code = "b = makeBox(10, 10, 10)\n"
        s = crr.apply_first_repair("NameError", "name 'makeBox' is not defined", code)
        self.assertIsNotNone(s)
        self.assertTrue(s.has_autofix)
        self.assertIn("Part.makeBox(", s.fixed_code)


class BooleanResultDroppedTest(unittest.TestCase):
    def test_cut_result_assigned(self):
        code = "body.cut(hole)\n"
        s = crr.apply_first_repair(
            "AttributeError", "'NoneType' object has no attribute 'Shape'", code
        )
        self.assertIsNotNone(s)
        self.assertEqual(s.rule_id, "boolean_result_dropped")
        self.assertIn("body = body.cut(hole)", s.fixed_code)


class InplaceTranslateTest(unittest.TestCase):
    def test_translate_assignment_removed(self):
        code = "shape = shape.translate(FreeCAD.Vector(1, 2, 3))\n"
        s = crr.apply_first_repair(
            "AttributeError", "'NoneType' object has no attribute 'x'", code
        )
        self.assertIsNotNone(s)
        self.assertEqual(s.rule_id, "inplace_translate_assigned")
        self.assertNotIn("shape = shape.translate", s.fixed_code)


class TranslateArityTest(unittest.TestCase):
    def test_cq_translate_wrapped_in_tuple(self):
        code = "res = cq.Workplane('XY').box(1,1,1).translate(1, 2, 3)\n"
        s = crr.apply_first_repair("TypeError", "translate() takes 2 positional arguments", code)
        self.assertIsNotNone(s)
        self.assertIn("translate((1, 2, 3))", s.fixed_code)

    def test_freecad_translate_wrapped_in_vector(self):
        code = "shape.translate(1, 2, 3)\n"
        s = crr.apply_first_repair("TypeError", "translate() takes 2 positional arguments", code)
        self.assertIsNotNone(s)
        self.assertIn("FreeCAD.Vector(1, 2, 3)", s.fixed_code)


class HintOnlyTest(unittest.TestCase):
    def test_extrude_before_2d_is_hint_only(self):
        s = crr.apply_first_repair(
            "ValueError", "No pending wires present for extrude", "cq.Workplane('XY').extrude(5)"
        )
        self.assertIsNotNone(s)
        self.assertFalse(s.has_autofix)
        self.assertIn("extrude", s.hint)

    def test_make_ellipse_missing(self):
        s = crr.apply_first_repair("AttributeError", "module 'Part' has no attribute 'makeEllipse'", "Part.makeEllipse()")
        self.assertIsNotNone(s)
        self.assertEqual(s.rule_id, "make_ellipse_missing")


class NoMatchTest(unittest.TestCase):
    def test_unknown_error_returns_none(self):
        s = crr.apply_first_repair("KeyError", "'totally_unknown'", "x = {}['totally_unknown']")
        self.assertIsNone(s)

    def test_autofix_preferred_over_hint(self):
        # makeBox autofix should sort before any hint-only rule.
        code = "b = makeBox(1,1,1)\n"
        suggestions = crr.suggest_repair("NameError", "name 'makeBox' is not defined", code)
        self.assertTrue(suggestions[0].has_autofix)


if __name__ == "__main__":
    unittest.main()
