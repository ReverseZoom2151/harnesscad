import unittest

from harnesscad.domain.spec.kcl_grammar import (
    BINARY_OPERATORS,
    KEYWORDS,
    SYNTAX_KINDS,
    Token,
    keyword_or_word,
    lex,
    lex_significant,
)


def kinds(src):
    return [t.kind for t in lex_significant(src)]


class TestKeywords(unittest.TestCase):
    def test_reserved_word_maps_to_keyword_kind(self):
        self.assertEqual(keyword_or_word("fn"), "FnKw")
        self.assertEqual(keyword_or_word("return"), "ReturnKw")
        self.assertEqual(keyword_or_word("import"), "ImportKw")

    def test_plain_identifier_is_word(self):
        self.assertEqual(keyword_or_word("myVar"), "Word")
        self.assertEqual(keyword_or_word("fnord"), "Word")  # not the fn keyword

    def test_all_keyword_kinds_are_declared(self):
        for kind in KEYWORDS.values():
            self.assertIn(kind, SYNTAX_KINDS)


class TestLexer(unittest.TestCase):
    def test_lossless_reconstruction(self):
        src = "x = 5mm |> line(%)  // comment\nfn foo() { return 1 }"
        toks = lex(src)
        self.assertEqual("".join(t.text for t in toks), src)

    def test_byte_ranges_are_contiguous(self):
        src = 'a = "hi" + 3.5deg'
        toks = lex(src)
        self.assertEqual(toks[0].start, 0)
        for a, b in zip(toks, toks[1:]):
            self.assertEqual(a.end, b.start)
        self.assertEqual(toks[-1].end, len(src.encode("utf-8")))

    def test_pipe_and_operators(self):
        self.assertEqual(
            kinds("a |> b"), ["Word", "PipeGt", "Word"]
        )
        self.assertEqual(kinds(">= <= == => != .. ..< ::"),
                         ["GtEq", "LtEq", "EqEq", "FatArrow", "BangEq",
                          "DoublePeriod", "DoublePeriodLessThan", "DoubleColon"])

    def test_number_with_unit_suffix(self):
        toks = lex_significant("10mm 3.5deg .5 42")
        self.assertEqual([t.kind for t in toks], ["Number"] * 4)
        self.assertEqual(toks[0].text, "10mm")
        self.assertEqual(toks[1].text, "3.5deg")
        self.assertEqual(toks[2].text, ".5")

    def test_dot_vs_range(self):
        # `a.b` is member access (Period), `a..b` is a range (DoublePeriod).
        self.assertEqual(kinds("a.b"), ["Word", "Period", "Word"])
        self.assertEqual(kinds("a..b"), ["Word", "DoublePeriod", "Word"])
        self.assertEqual(kinds("1..<5"),
                         ["Number", "DoublePeriodLessThan", "Number"])

    def test_string_multiline_and_escape(self):
        toks = lex_significant('"line1\nline2" \'a\\\'b\'')
        self.assertEqual(toks[0].kind, "String")
        self.assertIn("\n", toks[0].text)
        self.assertEqual(toks[1].kind, "String")

    def test_unterminated_string_recovers_at_line_end(self):
        toks = lex('"oops\nok')
        self.assertEqual(toks[0].kind, "UnterminatedString")
        self.assertEqual(toks[0].text, '"oops')

    def test_block_comment_terminated_and_not(self):
        self.assertEqual(lex("/* hi */")[0].kind, "BlockComment")
        self.assertEqual(lex("/* hi")[0].kind, "UnterminatedBlockComment")

    def test_unknown_recovery(self):
        toks = lex_significant("a ` b")
        self.assertIn("Unknown", [t.kind for t in toks])

    def test_keyword_in_stream(self):
        self.assertEqual(kinds("fn foo"), ["FnKw", "Word"])

    def test_determinism(self):
        src = "sketch = startSketchOn(XY) |> circle(radius = 5mm)"
        self.assertEqual([t.__dict__ for t in lex(src)],
                         [t.__dict__ for t in lex(src)])

    def test_unicode_byte_offsets(self):
        # A multibyte identifier char advances byte offsets by >1 per code point.
        src = "é = 1"  # e-acute
        toks = lex(src)
        self.assertEqual(toks[0].kind, "Word")
        self.assertEqual(toks[0].end, 2)  # 2 UTF-8 bytes


class TestOperators(unittest.TestCase):
    def test_binary_operator_spellings(self):
        self.assertEqual(BINARY_OPERATORS["Add"], "+")
        self.assertEqual(BINARY_OPERATORS["Gte"], ">=")
        self.assertEqual(BINARY_OPERATORS["Pow"], "^")


if __name__ == "__main__":
    unittest.main()
