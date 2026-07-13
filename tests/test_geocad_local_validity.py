"""Tests for GeoCAD local-part validity (closed + non-self-intersecting)."""

import unittest

from harnesscad.domain.geometry.sketch import geocad_local_validity as lv


class SegmentTest(unittest.TestCase):
    def test_crossing(self):
        self.assertTrue(lv.segments_properly_intersect(
            (0, 0), (2, 2), (0, 2), (2, 0)))

    def test_non_crossing(self):
        self.assertFalse(lv.segments_properly_intersect(
            (0, 0), (1, 0), (0, 1), (1, 1)))

    def test_collinear_overlap(self):
        self.assertTrue(lv.segments_properly_intersect(
            (0, 0), (2, 0), (1, 0), (3, 0)))


class ClosedTest(unittest.TestCase):
    def test_square_closed(self):
        self.assertTrue(lv.is_closed([(0, 0), (1, 0), (1, 1), (0, 1)]))

    def test_explicit_closure_tolerated(self):
        self.assertTrue(lv.is_closed([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]))

    def test_two_points_not_closed(self):
        self.assertFalse(lv.is_closed([(0, 0), (1, 1)]))

    def test_zero_length_edge_rejected(self):
        self.assertFalse(lv.is_closed([(0, 0), (0, 0), (1, 1)]))


class SimplePolygonTest(unittest.TestCase):
    def test_square_simple(self):
        self.assertTrue(lv.is_simple_polygon([(0, 0), (2, 0), (2, 2), (0, 2)]))

    def test_bowtie_not_simple(self):
        # Self-intersecting "bowtie" quadrilateral.
        self.assertFalse(lv.is_simple_polygon([(0, 0), (2, 2), (2, 0), (0, 2)]))


class CheckLoopTest(unittest.TestCase):
    def test_valid_square(self):
        rep = lv.check_loop([(0, 0), (2, 0), (2, 2), (0, 2)])
        self.assertTrue(rep.valid)
        self.assertTrue(rep.closed and rep.simple)

    def test_invalid_open(self):
        rep = lv.check_loop([(0, 0), (1, 1)])
        self.assertFalse(rep.valid)
        self.assertFalse(rep.closed)

    def test_invalid_self_intersect(self):
        rep = lv.check_loop([(0, 0), (2, 2), (2, 0), (0, 2)])
        self.assertFalse(rep.valid)
        self.assertTrue(rep.closed)
        self.assertFalse(rep.simple)

    def test_pv_rate(self):
        loops = [
            [(0, 0), (2, 0), (2, 2), (0, 2)],   # valid
            [(0, 0), (2, 2), (2, 0), (0, 2)],   # invalid (bowtie)
        ]
        self.assertAlmostEqual(lv.prediction_validity_rate(loops), 0.5)

    def test_pv_empty(self):
        self.assertEqual(lv.prediction_validity_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
