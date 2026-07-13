import unittest

from harnesscad.domain.geometry.cqcontrib_selector_dsl import (
    And,
    DirMinMax,
    Entity,
    Not,
    Or,
    SelectorError,
    parse,
    select,
    tokenize,
)


def box_faces(size=2.0):
    """Six faces of an axis-aligned cube centred at the origin."""
    h = size / 2.0
    return [
        Entity((0, 0, h), (0, 0, 1), "PLANE", "top"),
        Entity((0, 0, -h), (0, 0, -1), "PLANE", "bottom"),
        Entity((h, 0, 0), (1, 0, 0), "PLANE", "xmax"),
        Entity((-h, 0, 0), (-1, 0, 0), "PLANE", "xmin"),
        Entity((0, h, 0), (0, 1, 0), "PLANE", "ymax"),
        Entity((0, -h, 0), (0, -1, 0), "PLANE", "ymin"),
    ]


def names(ents):
    return [e.name for e in ents]


class TestTokenize(unittest.TestCase):
    def test_punct_split(self):
        self.assertEqual(tokenize(">Z[1]"), [">", "Z", "[", "1", "]"])

    def test_words(self):
        self.assertEqual(tokenize("not (<X or >X)"),
                         ["not", "(", "<", "X", "or", ">", "X", ")"])


class TestDirectionMinMax(unittest.TestCase):
    def setUp(self):
        self.faces = box_faces()

    def test_max_z(self):
        self.assertEqual(names(select(">Z", self.faces)), ["top"])

    def test_min_z(self):
        self.assertEqual(names(select("<Z", self.faces)), ["bottom"])

    def test_max_x(self):
        self.assertEqual(names(select(">X", self.faces)), ["xmax"])

    def test_ties_all_returned(self):
        ents = [Entity((0, 0, 1), name="a"), Entity((5, 0, 1), name="b"),
                Entity((0, 0, 0), name="c")]
        self.assertEqual(names(select(">Z", ents)), ["a", "b"])

    def test_index_groups(self):
        ents = [Entity((0, 0, 0), name="z0"), Entity((0, 0, 1), name="z1"),
                Entity((0, 0, 2), name="z2")]
        self.assertEqual(names(select(">Z[0]", ents)), ["z2"])
        self.assertEqual(names(select(">Z[1]", ents)), ["z1"])
        self.assertEqual(names(select("<Z[1]", ents)), ["z1"])
        self.assertEqual(names(select("<Z[-1]", ents)), ["z2"])

    def test_index_out_of_range(self):
        self.assertEqual(select(">Z[9]", box_faces()), [])


class TestAxisSelectors(unittest.TestCase):
    def setUp(self):
        self.faces = box_faces()

    def test_parallel_z(self):
        self.assertEqual(sorted(names(select("|Z", self.faces))),
                         ["bottom", "top"])

    def test_perpendicular_z(self):
        self.assertEqual(sorted(names(select("#Z", self.faces))),
                         ["xmax", "xmin", "ymax", "ymin"])

    def test_directional_sense(self):
        self.assertEqual(names(select("+Z", self.faces)), ["top"])
        self.assertEqual(names(select("-Z", self.faces)), ["bottom"])

    def test_vector_axis(self):
        self.assertEqual(names(select("|(0,0,1)", self.faces)), ["top", "bottom"])

    def test_no_axis_entities_excluded(self):
        ents = [Entity((0, 0, 0), (0, 0, 0), "VERTEX", "v")]
        self.assertEqual(select("|Z", ents), [])


class TestTypeSelector(unittest.TestCase):
    def test_type(self):
        ents = [Entity((0, 0, 0), (0, 0, 1), "CIRCLE", "c"),
                Entity((0, 0, 0), (0, 0, 1), "PLANE", "p")]
        self.assertEqual(names(select("%CIRCLE", ents)), ["c"])
        self.assertEqual(names(select("%circle", ents)), ["c"])


class TestLogicalOperators(unittest.TestCase):
    def setUp(self):
        self.faces = box_faces()

    def test_not(self):
        self.assertEqual(len(select("not >Z", self.faces)), 5)

    def test_or(self):
        self.assertEqual(sorted(names(select(">Z or <Z", self.faces))),
                         ["bottom", "top"])

    def test_contrib_shelled_cube_selector(self):
        # from Shelled_Cube..._Logical_Selector_Operators.py
        got = names(select("not(<X or >X or <Y or >Y)", self.faces))
        self.assertEqual(sorted(got), ["bottom", "top"])

    def test_and(self):
        got = names(select("#Z and >X", self.faces))
        self.assertEqual(got, ["xmax"])

    def test_exc(self):
        got = names(select("|Z exc >Z", self.faces))
        self.assertEqual(got, ["bottom"])

    def test_and_binds_tighter_than_or(self):
        node = parse("#Z and >X or >Z")
        self.assertIsInstance(node, Or)
        self.assertIsInstance(node.left, And)

    def test_order_preserved(self):
        got = names(select("not >X", self.faces))
        self.assertEqual(got, ["top", "bottom", "xmin", "ymax", "ymin"])


class TestParseErrors(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(SelectorError):
            parse("")

    def test_unknown_axis(self):
        with self.assertRaises(SelectorError):
            parse(">Q")

    def test_trailing(self):
        with self.assertRaises(SelectorError):
            parse(">Z )")

    def test_unbalanced(self):
        with self.assertRaises(SelectorError):
            parse("(>Z")

    def test_parse_tree_shape(self):
        node = parse("not >Z")
        self.assertIsInstance(node, Not)
        self.assertIsInstance(node.child, DirMinMax)
        self.assertTrue(node.child.maximize)


if __name__ == "__main__":
    unittest.main()
