import unittest

from harnesscad.domain.geometry.cq_selector_algebra import (
    AndSelector,
    CenterNthSelector,
    DirectionMinMaxSelector,
    DirectionNthSelector,
    DirectionSelector,
    InverseSelector,
    ParallelDirSelector,
    PerpendicularDirSelector,
    Shape,
    SubtractSelector,
    SumSelector,
    TypeSelector,
)
from harnesscad.domain.geometry.cq_selector_grammar import (
    GRAMMAR_FINDINGS,
    parse_selector,
    tokenize,
)


def cube_faces(size=2.0):
    h = size / 2.0
    return [
        Shape("Face", (0, 0, h), (0, 0, 1), "PLANE", name="top"),
        Shape("Face", (0, 0, -h), (0, 0, -1), "PLANE", name="bot"),
        Shape("Face", (h, 0, 0), (1, 0, 0), "PLANE", name="right"),
        Shape("Face", (-h, 0, 0), (-1, 0, 0), "PLANE", name="left"),
        Shape("Face", (0, h, 0), (0, 1, 0), "PLANE", name="front"),
        Shape("Face", (0, -h, 0), (0, -1, 0), "PLANE", name="back"),
    ]


class TestTokenize(unittest.TestCase):
    def test_digraph(self):
        self.assertEqual(tokenize(">>Z"), [">>", "Z"])
        self.assertEqual(tokenize("<<X[1]"), ["<<", "X", "[", "1", "]"])
        self.assertEqual(tokenize(">Z"), [">", "Z"])


class TestAtoms(unittest.TestCase):
    def test_max_min(self):
        self.assertIsInstance(parse_selector(">Z"), DirectionMinMaxSelector)
        self.assertIsInstance(parse_selector("<Z"), DirectionMinMaxSelector)

    def test_indexed_dir_is_direction_nth(self):
        self.assertIsInstance(parse_selector(">Z[1]"), DirectionNthSelector)

    def test_center_nth_digraph(self):
        s = parse_selector(">>Z")
        self.assertIsInstance(s, CenterNthSelector)
        s = parse_selector("<<Z[0]")
        self.assertIsInstance(s, CenterNthSelector)

    def test_parallel_perp(self):
        self.assertIsInstance(parse_selector("|Z"), ParallelDirSelector)
        self.assertIsInstance(parse_selector("#Z"), PerpendicularDirSelector)

    def test_plus_minus(self):
        self.assertIsInstance(parse_selector("+Z"), DirectionSelector)
        self.assertIsInstance(parse_selector("-Z"), DirectionSelector)

    def test_type(self):
        self.assertIsInstance(parse_selector("%PLANE"), TypeSelector)


class TestGrammarCorners(unittest.TestCase):
    """The corners that cqcontrib_selector_dsl omitted or got wrong."""

    def test_bare_direction(self):
        # a lone axis == DirectionSelector (same sense)
        s = parse_selector("Z")
        self.assertIsInstance(s, DirectionSelector)
        out = s.filter(cube_faces())
        self.assertEqual([f.name for f in out], ["top"])

    def test_compound_axis(self):
        s = parse_selector("|XY")
        self.assertIsInstance(s, ParallelDirSelector)
        self.assertEqual(s.direction, (1.0, 1.0, 0.0))

    def test_named_views(self):
        for name in ("front", "back", "left", "right", "top", "bottom"):
            s = parse_selector(name)
            self.assertIsInstance(s, DirectionMinMaxSelector)
        # 'top' view maximises +Y
        out = parse_selector("top").filter(cube_faces())
        self.assertEqual([f.name for f in out], ["front"])

    def test_except_spelling(self):
        a = parse_selector("|Z exc >Z")
        b = parse_selector("|Z except >Z")
        self.assertIsInstance(a, SubtractSelector)
        self.assertIsInstance(b, SubtractSelector)
        self.assertEqual(
            [f.name for f in a.filter(cube_faces())],
            [f.name for f in b.filter(cube_faces())],
        )

    def test_not_is_loosest_precedence(self):
        # 'not >Z and |Z' must parse as not(>Z and |Z), NOT (not >Z) and |Z.
        s = parse_selector("not >Z and |Z")
        self.assertIsInstance(s, InverseSelector)
        inner = s.selector
        self.assertIsInstance(inner, AndSelector)
        # not(>Z and |Z): >Z and |Z selects {top}; complement is the other five.
        out = {f.name for f in s.filter(cube_faces())}
        self.assertEqual(out, {"bot", "right", "left", "front", "back"})

    def test_and_not_requires_parens(self):
        # bare 'A and not B' is a parse error in CadQuery
        with self.assertRaises(Exception):
            parse_selector(">Z and not |Z")
        # parenthesised form is fine
        s = parse_selector(">Z and (not <Z)")
        self.assertIsInstance(s, AndSelector)


class TestPrecedenceEnd2End(unittest.TestCase):
    def test_or_union(self):
        s = parse_selector(">Z or >X")
        self.assertIsInstance(s, SumSelector)
        out = {f.name for f in s.filter(cube_faces())}
        self.assertEqual(out, {"top", "right"})

    def test_nested(self):
        s = parse_selector("(>Z or >X) and |Z")
        out = {f.name for f in s.filter(cube_faces())}
        self.assertEqual(out, {"top"})


class TestFindings(unittest.TestCase):
    def test_findings_present(self):
        self.assertTrue(any("not-precedence" in f for f in GRAMMAR_FINDINGS))
        self.assertGreaterEqual(len(GRAMMAR_FINDINGS), 6)


if __name__ == "__main__":
    unittest.main()
