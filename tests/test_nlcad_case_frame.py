import unittest

from harnesscad.domain.spec.nlcad_case_frame import (
    Nominal, classify, parse_command, verb_frame, known_verbs,
)


class TestClassify(unittest.TestCase):
    def test_shape_noun(self):
        n = classify("circle")
        self.assertEqual(n.feature, "shape")
        self.assertEqual(n.value, "circle")

    def test_shape_synonym(self):
        self.assertEqual(classify("rect").value, "rectangle")
        self.assertEqual(classify("block").value, "box")

    def test_dimension_word(self):
        self.assertEqual(classify("radius").feature, "dimension-name")

    def test_number(self):
        n = classify("5")
        self.assertEqual(n.feature, "quantity")
        self.assertEqual(n.value, 5.0)

    def test_location_word(self):
        self.assertEqual(classify("origin").feature, "location")

    def test_unknown(self):
        self.assertIsNone(classify("banana"))


class TestVerbFrames(unittest.TestCase):
    def test_known_verbs_nonempty(self):
        self.assertIn("draw", known_verbs())
        self.assertIn("move", known_verbs())

    def test_frame_action(self):
        self.assertEqual(verb_frame("create").action, "create")
        self.assertEqual(verb_frame("erase").action, "delete")
        self.assertEqual(verb_frame("shift").action, "translate")

    def test_unknown_verb(self):
        self.assertIsNone(verb_frame("frobnicate"))


class TestParseCommand(unittest.TestCase):
    def test_draw_circle_radius_location(self):
        cmd = parse_command("draw a circle of radius 5 at the origin")
        self.assertEqual(cmd.action, "create")
        self.assertEqual(cmd.obj, "circle")
        self.assertEqual(cmd.dimensions, {"radius": 5.0})
        self.assertEqual(cmd.location, "origin")
        self.assertTrue(cmd.complete)

    def test_coordinate_location(self):
        cmd = parse_command("create a square of side 10 at (3, 4)")
        self.assertEqual(cmd.obj, "square")
        self.assertEqual(cmd.dimensions, {"side": 10.0})
        self.assertEqual(cmd.location, (3.0, 4.0))

    def test_multiple_dimensions(self):
        cmd = parse_command("draw a rectangle of width 20 height 8")
        self.assertEqual(cmd.dimensions, {"width": 20.0, "height": 8.0})

    def test_move_to_target(self):
        cmd = parse_command("move the circle to (10, 0)")
        self.assertEqual(cmd.action, "translate")
        self.assertEqual(cmd.obj, "circle")
        self.assertEqual(cmd.target, (10.0, 0.0))

    def test_delete(self):
        cmd = parse_command("delete the hole")
        self.assertEqual(cmd.action, "delete")
        self.assertEqual(cmd.obj, "hole")

    def test_missing_object(self):
        cmd = parse_command("draw at the origin")
        self.assertIsNone(cmd.obj)
        self.assertIn("object", cmd.missing)
        self.assertFalse(cmd.complete)

    def test_no_verb_returns_none(self):
        self.assertIsNone(parse_command("the big red circle"))

    def test_determiners_ignored(self):
        a = parse_command("draw a circle of radius 5")
        b = parse_command("draw circle radius 5")
        self.assertEqual(a.obj, b.obj)
        self.assertEqual(a.dimensions, b.dimensions)

    def test_to_dict(self):
        cmd = parse_command("draw a circle of radius 5 at origin")
        d = cmd.to_dict()
        self.assertEqual(d["action"], "create")
        self.assertEqual(d["object"], "circle")
        self.assertEqual(d["dimensions"], {"radius": 5.0})

    def test_deterministic(self):
        t = "draw a rectangle of width 20 height 8 at (1, 2)"
        self.assertEqual(parse_command(t).to_dict(), parse_command(t).to_dict())


if __name__ == "__main__":
    unittest.main()
