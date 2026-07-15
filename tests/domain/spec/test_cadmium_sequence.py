"""Tests for the CADmium CAD-sequence representation."""

import unittest

from harnesscad.domain.spec.cadmium_sequence import (
    CadmiumError,
    CadSequence,
    detokenize,
    normalise,
    parse,
    score,
    tokenize,
)


SRC = """
# a simple part
sketch  plane=XY
circle  cx=0 cy=0 r=5
extrude dist=10 op=new
fillet  edges=all r=1
"""


class TestParse(unittest.TestCase):
    def test_parses_operations_in_order(self):
        seq = parse(SRC)
        self.assertEqual(seq.commands(), ("sketch", "circle", "extrude", "fillet"))
        self.assertEqual(len(seq), 4)

    def test_numeric_and_categorical_values(self):
        seq = parse("extrude dist=10 op=new")
        op = seq.operations[0]
        self.assertEqual(op.get("dist"), 10)
        self.assertEqual(op.get("op"), "new")

    def test_float_value(self):
        seq = parse("fillet edges=all r=1.5")
        self.assertEqual(seq.operations[0].get("r"), 1.5)

    def test_comments_and_blank_lines_ignored(self):
        seq = parse("\n\n# hi\nshell thickness=2  # trailing\n")
        self.assertEqual(seq.commands(), ("shell",))

    def test_unknown_command_rejected(self):
        with self.assertRaises(CadmiumError):
            parse("frobnicate x=1")

    def test_unknown_parameter_rejected(self):
        with self.assertRaises(CadmiumError):
            parse("circle cx=0 cy=0 radius=5")

    def test_duplicate_parameter_rejected(self):
        with self.assertRaises(CadmiumError):
            parse("circle cx=0 cx=1 r=5")

    def test_missing_equals_rejected(self):
        with self.assertRaises(CadmiumError):
            parse("circle cx 0")

    def test_non_strict_keeps_unknown_command(self):
        seq = parse("wibble a=1", strict=False)
        self.assertEqual(seq.commands(), ("wibble",))


class TestNormalise(unittest.TestCase):
    def test_parameter_order_canonicalised(self):
        a = normalise(parse("circle r=5 cy=0 cx=0"))
        b = normalise(parse("circle cx=0 cy=0 r=5"))
        self.assertEqual(a, b)

    def test_numeric_spelling_canonicalised(self):
        a = normalise(parse("extrude dist=10 op=new"))
        b = normalise(parse("extrude dist=10.0 op=new"))
        self.assertEqual(a, b)

    def test_normal_form_is_reparseable(self):
        seq = parse(SRC)
        again = parse(normalise(seq))
        self.assertEqual(normalise(seq), normalise(again))

    def test_deterministic(self):
        self.assertEqual(normalise(parse(SRC)), normalise(parse(SRC)))


class TestTokenizer(unittest.TestCase):
    def test_token_stream_shape(self):
        toks = tokenize(parse("circle cx=0 cy=0 r=5"))
        self.assertEqual(toks[0], "<bos>")
        self.assertEqual(toks[-1], "<eos>")
        self.assertIn("CMD_circle", toks)
        self.assertIn("P_r", toks)
        self.assertIn("N_5", toks)
        self.assertIn("<eop>", toks)

    def test_categorical_uses_c_token(self):
        toks = tokenize(parse("extrude dist=10 op=new"))
        self.assertIn("C_new", toks)
        self.assertNotIn("N_new", toks)

    def test_round_trip_detokenize(self):
        seq = parse(SRC)
        self.assertEqual(normalise(detokenize(tokenize(seq))), normalise(seq))

    def test_quantisation_buckets_numbers(self):
        toks = tokenize(parse("fillet edges=all r=1.2"), quantum=0.5)
        self.assertIn("N_1", toks)  # 1.2 snaps to nearest 0.5 -> 1.0 -> canonical 1
        toks2 = tokenize(parse("fillet edges=all r=1.4"), quantum=0.5)
        self.assertIn("N_1.5", toks2)  # 1.4 snaps to 1.5

    def test_bad_token_stream_rejected(self):
        with self.assertRaises(CadmiumError):
            detokenize(["<bos>", "N_5", "<eos>"])  # value before any parameter


class TestScore(unittest.TestCase):
    def test_identical_sequences_perfect(self):
        seq = parse(SRC)
        s = score(seq, seq)
        self.assertEqual(s.command_f1, 1.0)
        self.assertEqual(s.parameter_error, 0.0)
        self.assertEqual(s.edit_distance, 0)
        self.assertTrue(s.exact_match)

    def test_missing_command_lowers_recall(self):
        ref = parse(SRC)
        pred = parse("sketch plane=XY\ncircle cx=0 cy=0 r=5\nextrude dist=10 op=new")
        s = score(pred, ref)
        self.assertLess(s.command_recall, 1.0)
        self.assertEqual(s.command_precision, 1.0)  # every predicted cmd is in ref
        self.assertFalse(s.exact_match)
        self.assertEqual(s.edit_distance, 1)

    def test_parameter_error_reported(self):
        ref = parse("circle cx=0 cy=0 r=10")
        pred = parse("circle cx=0 cy=0 r=5")
        s = score(pred, ref)
        self.assertEqual(s.command_f1, 1.0)          # command matches
        self.assertAlmostEqual(s.parameter_error, 0.5 / 3.0, places=6)
        self.assertFalse(s.exact_match)

    def test_wrong_categorical_counts_as_error(self):
        ref = parse("extrude dist=10 op=new")
        pred = parse("extrude dist=10 op=cut")
        s = score(pred, ref)
        self.assertGreater(s.parameter_error, 0.0)

    def test_edit_distance_symmetric_magnitude(self):
        a = parse("sketch plane=XY\ncircle cx=0 cy=0 r=5")
        b = parse("sketch plane=XY")
        self.assertEqual(score(a, b).edit_distance, 1)

    def test_deterministic(self):
        ref, pred = parse(SRC), parse(SRC)
        self.assertEqual(score(pred, ref).as_dict(), score(pred, ref).as_dict())


if __name__ == "__main__":
    unittest.main()
