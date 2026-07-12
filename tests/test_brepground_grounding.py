"""Tests for reconstruction.brepground_grounding."""

import unittest

from reconstruction.brepground_grounding import (
    BRepPrimitive,
    ParsedQuery,
    ground,
    ground_all,
    ground_one,
    index_set,
    parse_query,
)


def _sample():
    # A block with a top planar face (max z), a bottom face, two cylindrical
    # holes (one large, one small), and some edges.
    return [
        BRepPrimitive(0, "face", "planar", (0.0, 0.0, 5.0), size=100.0),   # top
        BRepPrimitive(1, "face", "planar", (0.0, 0.0, 0.0), size=100.0),   # bottom
        BRepPrimitive(2, "face", "planar", (5.0, 0.0, 2.5), size=50.0),    # right
        BRepPrimitive(3, "face", "cylindrical", (1.0, 1.0, 2.5), size=8.0,
                      is_hole=True),                                        # small hole
        BRepPrimitive(4, "face", "cylindrical", (-1.0, -1.0, 2.5), size=20.0,
                      is_hole=True),                                        # large hole
        BRepPrimitive(5, "edge", "circle", (1.0, 1.0, 5.0), size=6.0,
                      is_hole=True),
        BRepPrimitive(6, "edge", "line", (0.0, 0.0, 5.0), size=10.0),
    ]


class TestParseQuery(unittest.TestCase):
    def test_top_face(self):
        q = parse_query("the top face")
        self.assertEqual(q.kind, "face")
        self.assertEqual(q.position, (2, +1))
        self.assertFalse(q.wants_all)

    def test_largest_hole(self):
        q = parse_query("the largest hole")
        self.assertTrue(q.require_hole)
        self.assertEqual(q.kind, "face")
        self.assertEqual(q.size_dir, +1)

    def test_all_circular_holes(self):
        q = parse_query("All circular holes on the surface.")
        self.assertTrue(q.require_hole)
        self.assertTrue(q.wants_all)
        self.assertIn("cylindrical", q.subtypes)

    def test_edges_plural(self):
        q = parse_query("all straight edges")
        self.assertEqual(q.kind, "edge")
        self.assertTrue(q.wants_all)
        self.assertEqual(q.subtypes, frozenset({"line"}))

    def test_unknown_words_ignored(self):
        q = parse_query("the wibble face")
        self.assertEqual(q.kind, "face")
        self.assertIsNone(q.position)

    def test_case_insensitive(self):
        self.assertEqual(parse_query("TOP FACE").position, (2, +1))


class TestGround(unittest.TestCase):
    def setUp(self):
        self.prims = _sample()

    def test_ground_top_face(self):
        best = ground_one("the top face", self.prims)
        self.assertEqual(best.index, 0)

    def test_ground_bottom_face(self):
        best = ground_one("the bottom face", self.prims)
        self.assertEqual(best.index, 1)

    def test_ground_right_face(self):
        best = ground_one("the rightmost face", self.prims)
        self.assertEqual(best.index, 2)

    def test_largest_hole(self):
        best = ground_one("the largest hole", self.prims)
        self.assertEqual(best.index, 4)

    def test_smallest_hole(self):
        best = ground_one("the smallest hole", self.prims)
        self.assertEqual(best.index, 3)

    def test_all_holes_faces(self):
        got = ground_all("all holes", self.prims)
        # holes grounded on cylindrical faces: indices 3 and 4.
        self.assertEqual(index_set(got), (3, 4))

    def test_all_circular_edges(self):
        got = ground_all("all circular edges", self.prims)
        self.assertEqual(index_set(got), (5,))

    def test_kind_filter_excludes_edges(self):
        got = ground("faces", self.prims)
        self.assertTrue(all(p.kind == "face" for p in got))
        self.assertEqual(len(got), 5)

    def test_no_match_returns_empty(self):
        self.assertEqual(ground("spherical face", self.prims), [])
        self.assertIsNone(ground_one("spherical face", self.prims))

    def test_singular_no_cue_returns_all_matches(self):
        # "the planar face" with no ranking cue: all planar faces returned.
        got = ground_all("the planar face", self.prims)
        self.assertEqual(index_set(got), (0, 1, 2))

    def test_singular_with_cue_returns_one(self):
        got = ground_all("the top planar face", self.prims)
        self.assertEqual([p.index for p in got], [0])

    def test_deterministic_tie_break(self):
        # Two equal-size top-less faces -> index order is the tie-break.
        prims = [
            BRepPrimitive(7, "face", "planar", (0.0, 0.0, 0.0), size=10.0),
            BRepPrimitive(3, "face", "planar", (0.0, 0.0, 0.0), size=10.0),
        ]
        got = ground("faces", prims)
        self.assertEqual([p.index for p in got], [3, 7])

    def test_ranking_is_stable(self):
        r1 = [p.index for p in ground("all holes", self.prims)]
        r2 = [p.index for p in ground("all holes", self.prims)]
        self.assertEqual(r1, r2)


class TestValidation(unittest.TestCase):
    def test_bad_kind(self):
        with self.assertRaises(ValueError):
            BRepPrimitive(0, "vertex")

    def test_negative_size(self):
        with self.assertRaises(ValueError):
            BRepPrimitive(0, "face", size=-1.0)


if __name__ == "__main__":
    unittest.main()
