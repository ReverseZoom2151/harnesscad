"""Tests for the BrepGPT Voronoi Half-Patch face decomposition."""

import unittest

from harnesscad.domain.reconstruction.brep import voronoi_half_patch as vhp


class DistanceTest(unittest.TestCase):
    def test_distance_to_polyline(self):
        # vertical segment from (0,0) to (0,1); point (2, 0.5) distance 2.
        self.assertAlmostEqual(vhp.distance_to_polyline((2.0, 0.5), [(0, 0), (0, 1)]), 2.0)

    def test_single_vertex_polyline(self):
        self.assertAlmostEqual(vhp.distance_to_polyline((3.0, 4.0), [(0, 0)]), 5.0)


class NearestCurveTest(unittest.TestCase):
    def test_picks_closer_curve(self):
        left = [(0.0, 0.0), (0.0, 1.0)]
        right = [(1.0, 0.0), (1.0, 1.0)]
        self.assertEqual(vhp.nearest_curve((0.1, 0.5), [left, right]), 0)
        self.assertEqual(vhp.nearest_curve((0.9, 0.5), [left, right]), 1)

    def test_no_curves_raises(self):
        with self.assertRaises(ValueError):
            vhp.nearest_curve((0, 0), [])


class DecomposeTest(unittest.TestCase):
    def setUp(self):
        # Two vertical boundary curves partition the unit square into left/right halves.
        self.left = [(0.0, 0.0), (0.0, 1.0)]
        self.right = [(1.0, 0.0), (1.0, 1.0)]

    def test_two_patches(self):
        d = vhp.decompose_face([self.left, self.right], resolution=8)
        self.assertEqual(len(d.patches), 2)

    def test_symmetric_split_equal_area(self):
        d = vhp.decompose_face([self.left, self.right], resolution=8)
        areas = sorted(p.cell_area for p in d.patches)
        self.assertAlmostEqual(areas[0], areas[1], places=6)

    def test_adjacency_between_the_two_cells(self):
        d = vhp.decompose_face([self.left, self.right], resolution=8)
        self.assertIn((0, 1), d.adjacency)

    def test_dominant_curve_when_one_owns_more(self):
        # A single curve owns the whole domain.
        d = vhp.decompose_face([self.left], resolution=4)
        self.assertEqual(d.dominant_curve(), 0)

    def test_labels_shape(self):
        d = vhp.decompose_face([self.left, self.right], resolution=5)
        self.assertEqual(len(d.labels), 5)
        self.assertEqual(len(d.labels[0]), 5)

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            vhp.decompose_face([self.left], resolution=1)


if __name__ == "__main__":
    unittest.main()
