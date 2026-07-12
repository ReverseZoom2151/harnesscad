"""Tests for generation.cqask_api_reference (CadQuery API parse + retrieval)."""

import unittest

from generation.cqask_api_reference import (
    ApiCard,
    parse_api_line,
    parse_reference,
    parse_signature_args,
    retrieve,
    build_prompt_context,
    render_card,
    card_tokens,
)


# A slice of CQAsk's actual system-prompt API block.
_REF = """
    cq.Workplane.center(x, y)- Shift local coordinates to the specified location.
    cq.Workplane.lineTo(x, y[, forConstruction])- Make a line from the current point to the provided point
    cq.Workplane.rect(xLen, yLen[, centered, ...])- Make a rectangle for each item on the stack.
    cq.Workplane.circle(radius[, forConstruction])- Make a circle for each item on the stack.
    cq.Workplane.polarArray(radius, startAngle, ...)- Creates a polar array of points and pushes them onto the stack.
    this is not a signature line and must be skipped
    cq.Workplane.close()- End construction, and attempt to build a closed wire.
"""


class TestParseSignatureArgs(unittest.TestCase):
    def test_all_required(self):
        req, opt, var = parse_signature_args("x, y")
        self.assertEqual(req, ("x", "y"))
        self.assertEqual(opt, ())
        self.assertFalse(var)

    def test_optional_bracket(self):
        req, opt, var = parse_signature_args("x, y[, forConstruction]")
        self.assertEqual(req, ("x", "y"))
        self.assertEqual(opt, ("forConstruction",))
        self.assertFalse(var)

    def test_variadic_tail(self):
        req, opt, var = parse_signature_args("xLen, yLen[, centered, ...]")
        self.assertEqual(req, ("xLen", "yLen"))
        self.assertEqual(opt, ("centered",))
        self.assertTrue(var)

    def test_bare_variadic(self):
        req, opt, var = parse_signature_args("radius, startAngle, ...")
        self.assertEqual(req, ("radius", "startAngle"))
        self.assertTrue(var)

    def test_empty(self):
        self.assertEqual(parse_signature_args(""), ((), (), False))

    def test_default_and_annotation_stripped(self):
        req, opt, var = parse_signature_args("naca_string, chord_length[, angle_units=\"rad\"]")
        self.assertEqual(req, ("naca_string", "chord_length"))
        self.assertEqual(opt, ("angle_units",))


class TestParseApiLine(unittest.TestCase):
    def test_basic(self):
        card = parse_api_line("cq.Workplane.center(x, y)- Shift local coordinates.")
        self.assertIsNotNone(card)
        self.assertEqual(card.qualname, "cq.Workplane.center")
        self.assertEqual(card.method, "center")
        self.assertEqual(card.required, ("x", "y"))
        self.assertEqual(card.description, "Shift local coordinates.")

    def test_no_args(self):
        card = parse_api_line("cq.Workplane.close()- End construction.")
        self.assertEqual(card.required, ())
        self.assertEqual(card.optional, ())

    def test_non_signature_returns_none(self):
        self.assertIsNone(parse_api_line("just some prose"))
        self.assertIsNone(parse_api_line(""))

    def test_arity(self):
        card = parse_api_line("cq.Workplane.rect(xLen, yLen[, centered, ...])- Make a rectangle.")
        lo, hi = card.arity()
        self.assertEqual(lo, 2)
        self.assertEqual(hi, float("inf"))

    def test_arity_bounded(self):
        card = parse_api_line("cq.Workplane.lineTo(x, y[, forConstruction])- line")
        self.assertEqual(card.arity(), (2, 3.0))


class TestParseReference(unittest.TestCase):
    def test_count_and_skip(self):
        cards = parse_reference(_REF)
        names = [c.method for c in cards]
        self.assertIn("center", names)
        self.assertIn("polarArray", names)
        self.assertNotIn("this", names)
        self.assertEqual(len(cards), 6)

    def test_dedup(self):
        dup = "cq.Workplane.circle(r)- a\ncq.Workplane.circle(r)- b"
        cards = parse_reference(dup)
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].description, "a")

    def test_deterministic_order(self):
        self.assertEqual(parse_reference(_REF), parse_reference(_REF))


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.cards = parse_reference(_REF)

    def test_method_name_hit_ranks_first(self):
        hits = retrieve("draw a circle", self.cards, top_k=3)
        self.assertTrue(len(hits) >= 1)
        self.assertEqual(hits[0].card.method, "circle")

    def test_description_token_match(self):
        hits = retrieve("make a rectangle", self.cards, top_k=3)
        methods = [h.card.method for h in hits]
        self.assertIn("rect", methods)

    def test_no_match_dropped(self):
        hits = retrieve("xyzzy quux nonsense", self.cards)
        self.assertEqual(hits, ())

    def test_deterministic(self):
        a = retrieve("line to point", self.cards)
        b = retrieve("line to point", self.cards)
        self.assertEqual(a, b)

    def test_top_k_limit(self):
        hits = retrieve("make a line point wire array", self.cards, top_k=2)
        self.assertLessEqual(len(hits), 2)


class TestPromptContext(unittest.TestCase):
    def setUp(self):
        self.cards = parse_reference(_REF)

    def test_header_only_on_no_match(self):
        ctx = build_prompt_context("xyzzy", self.cards)
        self.assertEqual(ctx, "Relevant CadQuery API:")

    def test_includes_relevant(self):
        ctx = build_prompt_context("draw a circle", self.cards, top_k=2)
        self.assertIn("cq.Workplane.circle", ctx)
        self.assertTrue(ctx.startswith("Relevant CadQuery API:"))

    def test_render_roundtrip_shape(self):
        card = parse_api_line("cq.Workplane.rect(xLen, yLen[, centered, ...])- Make a rectangle.")
        rendered = render_card(card)
        self.assertIn("cq.Workplane.rect(", rendered)
        self.assertIn("xLen, yLen[, centered, ...]", rendered)
        self.assertIn("Make a rectangle.", rendered)

    def test_render_no_args(self):
        card = parse_api_line("cq.Workplane.close()- End construction.")
        self.assertEqual(render_card(card), "cq.Workplane.close() - End construction.")


class TestCardTokens(unittest.TestCase):
    def test_camel_split(self):
        card = parse_api_line("cq.Workplane.lineTo(x, y)- Make a line.")
        toks = card_tokens(card)
        self.assertIn("line", toks)
        self.assertIn("to", toks)


if __name__ == "__main__":
    unittest.main()
