"""Tests for reconstruction.brepground_describe."""

import unittest

from reconstruction.brepground_grounding import BRepPrimitive, ground_one
from reconstruction.brepground_describe import (
    describe,
    describe_all,
    describe_detailed,
    round_trips,
)


def _sample():
    return [
        BRepPrimitive(0, "face", "planar", (0.0, 0.0, 5.0), size=100.0),   # top
        BRepPrimitive(1, "face", "planar", (0.0, 0.0, 0.0), size=90.0),    # bottom
        BRepPrimitive(2, "face", "cylindrical", (1.0, 1.0, 2.5), size=8.0,
                      is_hole=True),                                        # small hole
        BRepPrimitive(3, "face", "cylindrical", (-1.0, -1.0, 2.5), size=20.0,
                      is_hole=True),                                        # large hole
        BRepPrimitive(4, "edge", "line", (0.0, 0.0, 5.0), size=10.0),
    ]


class TestDescribe(unittest.TestCase):
    def setUp(self):
        self.prims = _sample()

    def test_type_phrase_face(self):
        self.assertIn("planar face", describe(self.prims[0]))

    def test_type_phrase_hole(self):
        self.assertIn("cylindrical hole", describe(self.prims[2]))

    def test_type_phrase_edge(self):
        self.assertIn("straight edge", describe(self.prims[4]))

    def test_top_modifier(self):
        self.assertEqual(describe(self.prims[0], self.prims), "the top planar face")

    def test_bottom_modifier(self):
        self.assertEqual(describe(self.prims[1], self.prims), "the bottom planar face")

    def test_size_modifier_for_holes(self):
        # Two co-located holes: no position extreme, so size discriminates.
        holes = [
            BRepPrimitive(0, "face", "cylindrical", (0.0, 0.0, 0.0), size=8.0,
                          is_hole=True),
            BRepPrimitive(1, "face", "cylindrical", (0.0, 0.0, 0.0), size=20.0,
                          is_hole=True),
        ]
        self.assertIn("largest", describe(holes[1], holes))
        self.assertIn("smallest", describe(holes[0], holes))

    def test_intrinsic_has_no_modifier(self):
        # Without brep context there is no discriminating modifier.
        self.assertEqual(describe(self.prims[0]), "the planar face")


class TestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.prims = _sample()

    def test_each_primitive_round_trips(self):
        for p in self.prims:
            with self.subTest(index=p.index):
                self.assertTrue(round_trips(p, self.prims))

    def test_round_trip_grounds_to_same(self):
        for p in self.prims:
            phrase = describe(p, self.prims)
            got = ground_one(phrase, self.prims)
            self.assertIsNotNone(got)
            self.assertEqual(got.index, p.index)


class TestDetailed(unittest.TestCase):
    def setUp(self):
        self.prims = _sample()

    def test_detailed_mentions_size_and_position(self):
        d = describe_detailed(self.prims[0])
        self.assertIn("area", d)
        self.assertIn("100", d)
        self.assertIn("5", d)

    def test_detailed_edge_uses_length(self):
        self.assertIn("length", describe_detailed(self.prims[4]))

    def test_article_vowel(self):
        p = BRepPrimitive(9, "face", "toroidal", (0.0, 0.0, 0.0), size=1.0)
        # "toroidal" -> consonant "a"; ensure no crash and starts with article.
        self.assertTrue(describe_detailed(p).startswith(("a ", "an ")))

    def test_deterministic(self):
        self.assertEqual(
            describe_detailed(self.prims[3]), describe_detailed(self.prims[3])
        )


class TestDescribeAll(unittest.TestCase):
    def test_describe_all_length(self):
        prims = _sample()
        self.assertEqual(len(describe_all(prims)), len(prims))


if __name__ == "__main__":
    unittest.main()
