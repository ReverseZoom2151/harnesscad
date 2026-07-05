import unittest

from cisp.ops import AddCircle, Extrude, Hole, Mate, NewSketch, SetParam
from surfaces.commands import (
    CommandParseError,
    CommandSurface,
    HelpIntent,
    Mode,
    ModeIntent,
    OpIntent,
    QueryIntent,
    UndoIntent,
)


class CommandSurfaceTests(unittest.TestCase):
    def test_explicit_initial_mode_and_discovery(self):
        surface = CommandSurface()
        self.assertEqual(Mode.SKETCH, surface.state.mode)
        self.assertIn("circle", surface.available_commands())
        self.assertNotIn("extrude", surface.available_commands())

    def test_new_sketch_defaults_plane(self):
        intent = CommandSurface().parse("new-sketch")
        self.assertEqual(OpIntent(NewSketch("XY")), intent)

    def test_parses_quoted_tokens_and_numbers(self):
        intent = CommandSurface().parse('circle "front sketch" 1 2.5 4')
        self.assertEqual(AddCircle("front sketch", 1, 2.5, 4), intent.op)

    def test_mode_transition_is_explicit(self):
        surface = CommandSurface()
        intent = surface.parse("mode feature")
        self.assertEqual(ModeIntent(Mode.FEATURE), intent)
        self.assertEqual(Mode.FEATURE, surface.state.mode)
        self.assertIn("extrude", surface.available_commands())

    def test_feature_commands_return_typed_ops(self):
        surface = CommandSurface()
        surface.parse("mode feature")
        self.assertEqual(Extrude("sk1", 10), surface.parse("extrude sk1 10").op)
        hole = surface.parse("hole top 1 2 6 8").op
        self.assertEqual(Hole("top", 1, 2, 6, 8, False), hole)

    def test_set_param_coerces_scalar(self):
        surface = CommandSurface()
        surface.parse("mode feature")
        self.assertEqual(
            SetParam(3, "distance", 12.5),
            surface.parse("set-param 3 distance 12.5").op,
        )

    def test_assembly_mate(self):
        surface = CommandSurface()
        surface.parse("mode assembly")
        self.assertEqual(
            Mate("revolute", "i1", "i2"),
            surface.parse("mate revolute i1 i2").op,
        )

    def test_query_and_undo_are_non_op_intents(self):
        surface = CommandSurface()
        self.assertEqual(UndoIntent(2), surface.parse("undo 2"))
        surface.parse("mode query")
        self.assertEqual(QueryIntent("summary"), surface.parse("query summary"))

    def test_help_is_accessible_and_contextual(self):
        surface = CommandSurface()
        help_intent = surface.parse("help")
        self.assertIsInstance(help_intent, HelpIntent)
        self.assertIn("Current mode: sketch", help_intent.text)
        self.assertIn("circle SKETCH CX CY R", help_intent.text)
        self.assertIn("Add a circle", surface.parse("help circle").text)

    def test_typo_has_machine_and_screen_reader_friendly_suggestion(self):
        with self.assertRaises(CommandParseError) as caught:
            CommandSurface().parse("circel sk1 0 0 4")
        self.assertEqual("circel", caught.exception.token)
        self.assertIn("circle", caught.exception.suggestions)
        self.assertIn("Did you mean: circle?", caught.exception.accessible_message())

    def test_wrong_mode_explains_required_mode(self):
        with self.assertRaises(CommandParseError) as caught:
            CommandSurface().parse("extrude sk1 10")
        self.assertIn("switch to feature mode", caught.exception.accessible_message())

    def test_malformed_input_is_rejected_without_execution(self):
        surface = CommandSurface()
        for text in ('circle sk1 0 0', 'circle sk1 x 0 4', '"unterminated'):
            with self.subTest(text=text), self.assertRaises(CommandParseError):
                surface.parse(text)

    def test_invalid_mode_and_undo(self):
        surface = CommandSurface()
        with self.assertRaises(CommandParseError) as caught:
            surface.parse("mode featre")
        self.assertIn("feature", caught.exception.suggestions)
        with self.assertRaises(CommandParseError):
            surface.parse("undo 0")

    def test_parser_is_deterministic(self):
        left = CommandSurface().parse("rectangle sk1 0 0 10 20")
        right = CommandSurface().parse("rectangle sk1 0 0 10 20")
        self.assertEqual(left, right)
        self.assertIsInstance(left, OpIntent)


if __name__ == "__main__":
    unittest.main()
