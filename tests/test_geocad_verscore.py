"""Tests for GeoCAD Ver-score (vertex-based text-to-CAD consistency)."""

import unittest

from harnesscad.domain.reconstruction.evaluate import geocad_verscore as vs


class NormaliseTest(unittest.TestCase):
    def test_strip_article(self):
        self.assertEqual(vs.normalise_instruction("A Square"), "square")
        self.assertEqual(vs.normalise_instruction("an isosceles right triangle"),
                         "isosceles right triangle")

    def test_strip_dimensions(self):
        self.assertEqual(vs.normalise_instruction("a square with side 10"), "square")


class ScorePolygonTest(unittest.TestCase):
    def test_match(self):
        r = vs.score_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], "a square")
        self.assertTrue(r.matched)

    def test_match_with_dimensions(self):
        r = vs.score_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], "a square with side 2")
        self.assertTrue(r.matched)

    def test_mismatch(self):
        r = vs.score_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], "a square")
        self.assertFalse(r.matched)
        self.assertEqual(r.recovered, "a rectangle")

    def test_triangle_match(self):
        r = vs.score_polygon([(0, 0), (4, 0), (0, 4)],
                             "An Isosceles Right Triangle")
        self.assertTrue(r.matched)


class ScoreArcTest(unittest.TestCase):
    def test_semicircle_match(self):
        self.assertTrue(vs.score_arc_loop(180, "a semicircle").matched)

    def test_arc_mismatch(self):
        self.assertFalse(vs.score_arc_loop(90, "a semicircle").matched)


class AggregateTest(unittest.TestCase):
    def test_ver_score(self):
        results = [
            vs.score_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], "a square"),   # match
            vs.score_polygon([(0, 0), (4, 0), (4, 2), (0, 2)], "a square"),   # miss
        ]
        self.assertAlmostEqual(vs.ver_score(results), 0.5)

    def test_empty(self):
        self.assertEqual(vs.ver_score([]), 0.0)


if __name__ == "__main__":
    unittest.main()
