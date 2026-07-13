import unittest

from harnesscad.domain.geometry.seekcad_captype import (
    CAP_TYPES,
    END,
    START,
    SWEPT,
    CapTypeError,
    CapTypeResolver,
    build_refinement_entities,
)


class TestCapTypeResolver(unittest.TestCase):
    def setUp(self):
        self.r = CapTypeResolver(["e0", "e1", "c0"], "extrude#1")

    def test_primitive_set(self):
        self.assertEqual(self.r.sketch_primitives(), ("e0", "e1", "c0"))

    def test_resolve_categories(self):
        self.assertEqual(self.r.resolve_id("e0", START), "extrude#1/START/e0")
        self.assertEqual(self.r.resolve_id("e0", END), "extrude#1/END/e0")
        self.assertEqual(self.r.resolve_id("c0", SWEPT), "extrude#1/SWEPT/c0")

    def test_resolve_returns_primitive(self):
        b = self.r.resolve("e1", END)
        self.assertEqual(b.source_tag, "e1")
        self.assertEqual(b.cap_type, END)

    def test_unknown_primitive_rejected(self):
        with self.assertRaises(CapTypeError):
            self.r.resolve("intersection_edge", START)

    def test_unknown_cap_rejected(self):
        with self.assertRaises(CapTypeError):
            self.r.resolve("e0", "MIDDLE")

    def test_swept_set_size_and_order(self):
        b = self.r.swept_set()
        # 3 primitives x 3 cap types
        self.assertEqual(len(b), 9)
        self.assertEqual(b[0].reference_id, "extrude#1/START/e0")
        self.assertEqual(b[1].cap_type, END)
        self.assertEqual(b[2].cap_type, SWEPT)

    def test_feature_id_scopes_references(self):
        other = CapTypeResolver(["e0"], "extrude#2")
        self.assertNotEqual(
            other.resolve_id("e0", START), self.r.resolve_id("e0", START)
        )

    def test_empty_sketch(self):
        with self.assertRaises(ValueError):
            CapTypeResolver([], "f")

    def test_duplicate_tags(self):
        with self.assertRaises(ValueError):
            CapTypeResolver(["e0", "e0"], "f")

    def test_cap_types_constant(self):
        self.assertEqual(set(CAP_TYPES), {START, END, SWEPT})


class TestBuildRefinementEntities(unittest.TestCase):
    def setUp(self):
        self.r = CapTypeResolver(["e0", "e1"], "revolve#1")

    def test_dedup_and_order(self):
        ents = build_refinement_entities(
            self.r, [("e1", END), ("e0", START), ("e1", END)]
        )
        self.assertEqual(
            ents, ["revolve#1/END/e1", "revolve#1/START/e0"]
        )

    def test_unreferenceable_raises(self):
        with self.assertRaises(CapTypeError):
            build_refinement_entities(self.r, [("ghost", START)])


if __name__ == "__main__":
    unittest.main()
