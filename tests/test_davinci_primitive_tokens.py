import unittest

from harnesscad.io.ingest.davinci_primitive_tokens import (
    N_TOKENS, PARAM_COUNT, TYPE_TOKENS, decode_primitive, dequantize,
    encode_primitive, is_empty_slot, quantize, token_issues,
)


class TestQuantize(unittest.TestCase):
    def test_range_clamped(self):
        self.assertEqual(quantize(0.0), 1)
        self.assertEqual(quantize(0.999), 64)
        self.assertEqual(quantize(-5.0), 1)
        self.assertEqual(quantize(5.0), 64)

    def test_roundtrip_close(self):
        for token in (1, 20, 33, 64):
            self.assertEqual(quantize(dequantize(token)), token)

    def test_dequantize_bounds(self):
        with self.assertRaises(ValueError):
            dequantize(0)
        with self.assertRaises(ValueError):
            dequantize(65)


class TestEncodeDecode(unittest.TestCase):
    def test_line_roundtrip(self):
        tok = encode_primitive("line", (0.1, 0.2, 0.8, 0.9), construction=1)
        self.assertEqual(len(tok), N_TOKENS)
        prim = decode_primitive(tok)
        self.assertEqual(prim.type, "line")
        self.assertEqual(prim.construction, 1)
        self.assertEqual(len(prim.params), PARAM_COUNT["line"])

    def test_all_types_encode_len8(self):
        for t, n in PARAM_COUNT.items():
            if t == "none":
                continue
            tok = encode_primitive(t, tuple([0.5] * n))
            self.assertEqual(len(tok), 8)
            self.assertEqual(tok[0], TYPE_TOKENS[t])

    def test_padding_of_unused_slots(self):
        tok = encode_primitive("point", (0.5, 0.5))
        # point uses 2 param slots; remaining 4 are padded to token 1.
        self.assertEqual(tok[3:7], (1, 1, 1, 1))

    def test_bad_param_count(self):
        with self.assertRaises(ValueError):
            encode_primitive("circle", (0.1, 0.2))

    def test_bad_construction(self):
        with self.assertRaises(ValueError):
            encode_primitive("point", (0.1, 0.2), construction=2)

    def test_unknown_type(self):
        with self.assertRaises(KeyError):
            encode_primitive("spline", (0.1,))


class TestTokenIssues(unittest.TestCase):
    def test_valid(self):
        tok = encode_primitive("arc", (0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(token_issues(tok), ())

    def test_bad_length(self):
        self.assertTrue(token_issues((1, 2, 3)))

    def test_out_of_range_param(self):
        bad = (1, 0, 2, 3, 4, 5, 6, 0)
        issues = token_issues(bad)
        self.assertTrue(any("param-out-of-range" in x for x in issues))

    def test_construction_not_binary(self):
        bad = (1, 2, 3, 4, 5, 6, 7, 9)
        self.assertTrue(any("construction" in x for x in token_issues(bad)))

    def test_empty_slot(self):
        tok = encode_primitive("none", ())
        self.assertTrue(is_empty_slot(tok))


if __name__ == "__main__":
    unittest.main()
