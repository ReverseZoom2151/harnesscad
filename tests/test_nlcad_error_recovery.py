import unittest

from harnesscad.domain.spec.command_recovery import (
    insert_missing_determiners, insert_missing_prepositions,
    recover, parse_with_recovery,
)
from harnesscad.domain.spec.case_frame import _tokenize


class TestDeterminerRecovery(unittest.TestCase):
    def test_insert_article_after_verb(self):
        out, reps = insert_missing_determiners(_tokenize("draw circle"))
        self.assertEqual(out, ["draw", "a", "circle"])
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].kind, "determiner")

    def test_no_insert_when_present(self):
        out, reps = insert_missing_determiners(_tokenize("draw a circle"))
        self.assertEqual(out, ["draw", "a", "circle"])
        self.assertEqual(reps, [])


class TestPrepositionRecovery(unittest.TestCase):
    def test_restore_at_before_location(self):
        out, reps = insert_missing_prepositions(_tokenize("draw circle origin"))
        self.assertIn("at", out)
        self.assertEqual(out[-2:], ["at", "origin"])
        self.assertEqual(reps[0].kind, "preposition")

    def test_restore_at_before_coordinate(self):
        out, reps = insert_missing_prepositions(_tokenize("draw circle (3, 4)"))
        self.assertEqual(reps[0].inserted, "at")
        # 'at' inserted just before the '('
        idx = out.index("(")
        self.assertEqual(out[idx - 1], "at")

    def test_motion_verb_uses_to(self):
        out, reps = insert_missing_prepositions(
            _tokenize("move circle (10, 0)"), action="translate")
        self.assertEqual(reps[0].inserted, "to")

    def test_no_insert_when_governed(self):
        out, reps = insert_missing_prepositions(_tokenize("draw circle at origin"))
        self.assertEqual(reps, [])


class TestRecover(unittest.TestCase):
    def test_full_recovery_pipeline(self):
        rec = recover("draw circle origin")
        self.assertEqual(rec.repaired_text, "draw a circle at origin")
        kinds = {r.kind for r in rec.repairs}
        self.assertEqual(kinds, {"determiner", "preposition"})

    def test_unknown_word_collected(self):
        rec = recover("draw a squircle")
        self.assertIn("squircle", rec.unknown_words)
        self.assertTrue(rec.needs_replacement)

    def test_unknown_word_replacement(self):
        rec = recover("draw a squircle", replacements={"squircle": "circle"})
        self.assertFalse(rec.needs_replacement)
        self.assertIn("circle", rec.repaired_tokens)
        self.assertTrue(any(r.kind == "replacement" for r in rec.repairs))

    def test_deterministic(self):
        a = recover("draw circle origin").repaired_text
        b = recover("draw circle origin").repaired_text
        self.assertEqual(a, b)


class TestParseWithRecovery(unittest.TestCase):
    def test_terse_command_parses_after_recovery(self):
        cmd, rec = parse_with_recovery("draw circle radius 5 origin")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.action, "create")
        self.assertEqual(cmd.obj, "circle")
        self.assertEqual(cmd.dimensions, {"radius": 5.0})
        self.assertEqual(cmd.location, "origin")
        self.assertTrue(cmd.complete)

    def test_replacement_enables_parse(self):
        cmd, rec = parse_with_recovery(
            "make a squircle of radius 3",
            replacements={"squircle": "circle"})
        self.assertEqual(cmd.obj, "circle")
        self.assertEqual(cmd.dimensions, {"radius": 3.0})

    def test_motion_terse_recovery(self):
        cmd, rec = parse_with_recovery("move circle (10, 0)")
        self.assertEqual(cmd.action, "translate")
        self.assertEqual(cmd.target, (10.0, 0.0))


if __name__ == "__main__":
    unittest.main()
