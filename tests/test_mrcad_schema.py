"""Tests for editing.mrcad_schema."""

import unittest

from harnesscad.domain.editing.mrcad_schema import (
    Curve,
    Design,
    DeletePoint,
    EDIT_VOCABULARY,
    Instruction,
    MakeCurve,
    Message,
    MoveCurve,
    MovePoint,
    RemoveCurve,
    arc,
    circle,
    line,
    parse_instruction,
    tokenize,
)


class CurveTest(unittest.TestCase):
    def test_arity_enforced(self):
        with self.assertRaises(ValueError):
            Curve("line", ((0, 0),))
        with self.assertRaises(ValueError):
            Curve("arc", ((0, 0), (1, 1)))

    def test_unknown_kind(self):
        with self.assertRaises(ValueError):
            Curve("spline", ((0, 0), (1, 1)))

    def test_points_coerced_to_float(self):
        c = line((0, 0), (2, 3))
        self.assertEqual(c.points, ((0.0, 0.0), (2.0, 3.0)))
        self.assertTrue(all(isinstance(v, float) for p in c.points for v in p))

    def test_value_equality_and_hash(self):
        self.assertEqual(circle((0, 0), (4, 0)), circle((0, 0), (4, 0)))
        self.assertEqual(hash(line((0, 0), (1, 1))), hash(line((0, 0), (1, 1))))

    def test_has_point(self):
        c = arc((0, 0), (1, 1), (2, 0))
        self.assertTrue(c.has_point((1, 1)))
        self.assertFalse(c.has_point((5, 5)))

    def test_translate(self):
        c = line((0, 0), (2, 0)).translate((1, 1))
        self.assertEqual(c, line((1, 1), (3, 1)))

    def test_replace_point_only_matching(self):
        c = line((0, 0), (2, 0)).replace_point((0, 0), (9, 9))
        self.assertEqual(c, line((9, 9), (2, 0)))
        # no-op when point absent
        self.assertEqual(c.replace_point((5, 5), (0, 0)), c)


class DesignTest(unittest.TestCase):
    def test_dedup_on_construction(self):
        d = Design((line((0, 0), (1, 0)), line((0, 0), (1, 0))))
        self.assertEqual(len(d), 1)

    def test_add_remove_immutable(self):
        d0 = Design.empty()
        d1 = d0.add(line((0, 0), (1, 0)))
        self.assertEqual(len(d0), 0)
        self.assertEqual(len(d1), 1)
        d2 = d1.remove(line((0, 0), (1, 0)))
        self.assertEqual(len(d2), 0)

    def test_equality_order_independent(self):
        a = line((0, 0), (1, 0))
        b = circle((0, 0), (2, 0))
        self.assertEqual(Design((a, b)), Design((b, a)))

    def test_points_unique(self):
        d = Design((line((0, 0), (1, 0)), line((0, 0), (2, 2))))
        self.assertEqual(d.points(), ((0.0, 0.0), (1.0, 0.0), (2.0, 2.0)))


class ActionVocabularyTest(unittest.TestCase):
    def test_ops_carry_names(self):
        self.assertEqual(MakeCurve(line((0, 0), (1, 0))).op, "make_curve")
        self.assertEqual(RemoveCurve(line((0, 0), (1, 0))).op, "remove_curve")
        self.assertEqual(MoveCurve(line((0, 0), (1, 0)), (1, 1)).op, "move_curve")
        self.assertEqual(MovePoint((0, 0), (1, 1)).op, "move_point")
        self.assertEqual(DeletePoint((0, 0)).op, "delete_point")

    def test_vocabulary_complete(self):
        self.assertEqual(
            set(EDIT_VOCABULARY),
            {"make_curve", "remove_curve", "move_curve", "move_point", "delete_point"},
        )


class MessageTest(unittest.TestCase):
    def test_empty_modality(self):
        self.assertEqual(Message().modality(), "empty")

    def test_text_only(self):
        m = Message(text="make it straight")
        self.assertEqual(m.modality(), "text")
        self.assertTrue(m.has_text)
        self.assertFalse(m.has_drawing)

    def test_drawing_only(self):
        m = Message(strokes=(((0, 0), (1, 1)),))
        self.assertEqual(m.modality(), "drawing")

    def test_multimodal(self):
        m = Message(text="here", strokes=(((0, 0), (1, 0)),))
        self.assertEqual(m.modality(), "multimodal")

    def test_whitespace_text_is_not_text(self):
        self.assertEqual(Message(text="   ").modality(), "empty")

    def test_stroke_count_ignores_empty(self):
        m = Message(strokes=(((0, 0), (1, 0)), (), ((2, 2), (3, 3))))
        self.assertEqual(m.stroke_count(), 2)

    def test_ink_length(self):
        m = Message(strokes=(((0, 0), (3, 4)),))
        self.assertAlmostEqual(m.ink(), 5.0)


class ParserTest(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Move the-line, please!"), ("move", "the", "line", "please"))

    def test_imperative_refinement_instruction(self):
        m = Message(text="move the circle up", strokes=(((0, 0), (1, 1)),))
        ins = parse_instruction(m)
        self.assertIsInstance(ins, Instruction)
        self.assertEqual(ins.root_word, "move")
        self.assertTrue(ins.is_imperative)
        self.assertTrue(ins.is_refinement_like)
        self.assertEqual(ins.modality, "multimodal")
        self.assertEqual(ins.stroke_count, 1)

    def test_generation_like_no_verb_head(self):
        # A generation instruction: mostly drawing, noun-headed / no verb.
        m = Message(text="a flower with petals", strokes=(((0, 0), (1, 1)),))
        ins = parse_instruction(m)
        self.assertEqual(ins.root_word, "a")
        self.assertFalse(ins.is_imperative)
        self.assertFalse(ins.is_refinement_like)

    def test_empty_text_root(self):
        ins = parse_instruction(Message(strokes=(((0, 0), (1, 1)),)))
        self.assertEqual(ins.root_word, "")
        self.assertFalse(ins.is_imperative)


if __name__ == "__main__":
    unittest.main()
