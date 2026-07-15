"""Tests for the ShowUI interleaved action + grounding formats."""

import os
import tempfile
import unittest

from harnesscad.io.cua.action_stream import (
    ACTION_SPACE, Action, ActionFormatError, GroundingSample, click,
    from_pixel_click, input_text, load_grounding, save_grounding,
)


class TestActionValidate(unittest.TestCase):
    def test_click_needs_position(self):
        with self.assertRaises(ActionFormatError):
            Action("CLICK").validate()

    def test_click_ok(self):
        a = click(0.5, 0.25)
        self.assertEqual(a.action, "CLICK")

    def test_input_needs_value(self):
        with self.assertRaises(ActionFormatError):
            Action("INPUT", position=(0.5, 0.5)).validate()

    def test_position_must_be_fraction(self):
        with self.assertRaises(ActionFormatError):
            Action("CLICK", position=(500, 20)).validate()

    def test_unknown_action_rejected(self):
        with self.assertRaises(ActionFormatError):
            Action("TELEPORT", position=(0.1, 0.1)).validate()

    def test_enter_takes_no_position(self):
        with self.assertRaises(ActionFormatError):
            Action("ENTER", position=(0.1, 0.1)).validate()
        Action("ENTER").validate()

    def test_select_text_pair(self):
        a = Action("SELECT_TEXT", position=((0.1, 0.1), (0.2, 0.3))).validate()
        self.assertEqual(len(a.position), 2)

    def test_scroll_direction(self):
        Action("SCROLL", value="down").validate()


class TestActionStreamRoundtrip(unittest.TestCase):
    def test_roundtrip(self):
        a = input_text(0.5, 0.5, "37.5")
        line = a.to_stream()
        b = Action.parse_stream(line)
        self.assertEqual(a, b)

    def test_pair_roundtrip(self):
        a = Action("SELECT_TEXT", position=((0.1, 0.2), (0.3, 0.4)))
        self.assertEqual(Action.from_dict(a.to_dict()), a)

    def test_parse_bad_json(self):
        with self.assertRaises(ActionFormatError):
            Action.parse_stream("not json")


class TestPixelInterop(unittest.TestCase):
    def test_to_pixels(self):
        a = click(0.5, 0.25)
        self.assertEqual(a.to_pixels(1000, 400), (500, 100))

    def test_from_pixel_click(self):
        a = from_pixel_click(500, 100, 1000, 400)
        self.assertAlmostEqual(a.position[0], 0.5)
        self.assertAlmostEqual(a.position[1], 0.25)

    def test_from_pixel_rejects_zero_size(self):
        with self.assertRaises(ValueError):
            from_pixel_click(1, 1, 0, 10)

    def test_pair_to_pixels(self):
        a = Action("SELECT_TEXT", position=((0.0, 0.0), (1.0, 1.0)))
        self.assertEqual(a.to_pixels(100, 200), ((0, 0), (100, 200)))


class TestGrounding(unittest.TestCase):
    def test_hit(self):
        s = GroundingSample("the top face", (0.5, 0.5), bbox=(0.4, 0.4, 0.6, 0.6))
        self.assertTrue(s.hit(0.5, 0.5))
        self.assertFalse(s.hit(0.9, 0.9))

    def test_hit_without_bbox_raises(self):
        s = GroundingSample("x", (0.5, 0.5))
        with self.assertRaises(ValueError):
            s.hit(0.5, 0.5)

    def test_point_pixels(self):
        s = GroundingSample("x", (0.5, 0.25))
        self.assertEqual(s.point_pixels(1000, 400), (500, 100))

    def test_jsonl_roundtrip(self):
        samples = [GroundingSample("a", (0.1, 0.2), bbox=(0.0, 0.0, 0.3, 0.3)),
                   GroundingSample("b", (0.7, 0.8))]
        d = tempfile.mkdtemp()
        path = os.path.join(d, "g.jsonl")
        save_grounding(path, samples)
        loaded = load_grounding(path)
        self.assertEqual(loaded, samples)


class TestActionSpace(unittest.TestCase):
    def test_every_action_has_spec(self):
        for name, spec in ACTION_SPACE.items():
            self.assertIn("position", spec)
            self.assertIn("value", spec)


if __name__ == "__main__":
    unittest.main()
