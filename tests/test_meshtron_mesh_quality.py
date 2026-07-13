"""Tests for Meshtron artist-like mesh quality metrics."""

import math
import unittest

from harnesscad.eval.bench.meshtron_mesh_quality import (
    distribution_summary,
    edge_ratio,
    face_area,
    face_areas,
    face_count,
    histogram,
    mesh_quality_report,
    quad_count,
    quad_ratio,
    radius_ratio,
    triangle_area,
    triangle_aspect_ratios,
    triangle_count,
    valence_regularity,
    vertex_valences,
)


SQ = math.sqrt(3) / 2.0
EQUI = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, SQ, 0.0)]

# A unit square split into two right triangles.
SQUARE_VERTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
SQUARE_TRIS = [(0, 1, 2), (0, 2, 3)]


class CountsTest(unittest.TestCase):
    def test_counts(self):
        faces = [(0, 1, 2), (0, 1, 2, 3), (4, 5, 6)]
        self.assertEqual(face_count(faces), 3)
        self.assertEqual(triangle_count(faces), 2)
        self.assertEqual(quad_count(faces), 1)
        self.assertAlmostEqual(quad_ratio(faces), 1 / 3)

    def test_quad_ratio_empty(self):
        self.assertEqual(quad_ratio([]), 0.0)


class AreaTest(unittest.TestCase):
    def test_triangle_area_equilateral(self):
        # area of unit equilateral triangle = sqrt(3)/4
        self.assertAlmostEqual(triangle_area(*EQUI), math.sqrt(3) / 4.0)

    def test_face_area_square(self):
        self.assertAlmostEqual(face_area(SQUARE_VERTS, (0, 1, 2, 3)), 1.0)

    def test_face_areas_list(self):
        areas = face_areas(SQUARE_VERTS, SQUARE_TRIS)
        self.assertEqual(len(areas), 2)
        self.assertAlmostEqual(sum(areas), 1.0)

    def test_bad_face(self):
        with self.assertRaises(ValueError):
            face_area(SQUARE_VERTS, (0, 1))


class AspectTest(unittest.TestCase):
    def test_equilateral_ratios_are_one(self):
        self.assertAlmostEqual(radius_ratio(*EQUI), 1.0)
        self.assertAlmostEqual(edge_ratio(*EQUI), 1.0)

    def test_sliver_is_large(self):
        sliver = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.001, 0.0)]
        self.assertGreater(radius_ratio(*sliver), 10.0)
        self.assertGreater(edge_ratio(*sliver), 10.0)

    def test_degenerate_is_inf(self):
        degen = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        self.assertEqual(radius_ratio(*degen), math.inf)
        zero = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        self.assertEqual(edge_ratio(*zero), math.inf)

    def test_aspect_list_skips_non_triangles(self):
        faces = [(0, 1, 2), (0, 1, 2, 3)]
        ratios = triangle_aspect_ratios(SQUARE_VERTS, faces, metric="radius")
        self.assertEqual(len(ratios), 1)

    def test_bad_metric(self):
        with self.assertRaises(ValueError):
            triangle_aspect_ratios(SQUARE_VERTS, SQUARE_TRIS, metric="xyz")


class ValenceTest(unittest.TestCase):
    def test_square_valences(self):
        # each corner of the split square touches the diagonal or not
        vals = vertex_valences(4, SQUARE_TRIS)
        self.assertEqual(len(vals), 4)
        # vertices 0 and 2 (diagonal) connect to all others -> valence 3
        self.assertEqual(vals[0], 3)
        self.assertEqual(vals[2], 3)
        self.assertEqual(vals[1], 2)

    def test_regularity_fraction(self):
        # no vertex has valence 6 here
        self.assertEqual(valence_regularity(4, SQUARE_TRIS), 0.0)
        # ideal=3 matches the two diagonal vertices -> 2/4
        self.assertEqual(valence_regularity(4, SQUARE_TRIS, ideal=3), 0.5)

    def test_regularity_empty(self):
        self.assertEqual(valence_regularity(0, []), 0.0)


class SummaryTest(unittest.TestCase):
    def test_summary_stats(self):
        s = distribution_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(s["min"], 1.0)
        self.assertEqual(s["max"], 4.0)
        self.assertAlmostEqual(s["mean"], 2.5)
        self.assertAlmostEqual(s["median"], 2.5)

    def test_summary_counts_infinite(self):
        s = distribution_summary([1.0, math.inf, 3.0])
        self.assertEqual(s["n_infinite"], 1.0)
        self.assertEqual(s["count"], 3.0)
        self.assertAlmostEqual(s["mean"], 2.0)

    def test_summary_all_infinite(self):
        s = distribution_summary([math.inf, math.inf])
        self.assertNotIn("mean", s)
        self.assertEqual(s["n_infinite"], 2.0)

    def test_histogram(self):
        counts, edges = histogram([0.0, 1.0, 2.0, 3.0], bins=2)
        self.assertEqual(sum(counts), 4)
        self.assertEqual(len(edges), 3)

    def test_histogram_bad_bins(self):
        with self.assertRaises(ValueError):
            histogram([1.0], bins=0)

    def test_histogram_empty(self):
        counts, edges = histogram([math.inf], bins=3)
        self.assertEqual(counts, [0, 0, 0])


class ReportTest(unittest.TestCase):
    def test_report_keys(self):
        rep = mesh_quality_report(SQUARE_VERTS, SQUARE_TRIS)
        for key in ("face_count", "triangle_count", "quad_ratio",
                    "valence_regularity", "face_area", "aspect_ratio"):
            self.assertIn(key, rep)
        self.assertEqual(rep["triangle_count"], 2)

    def test_report_deterministic(self):
        a = mesh_quality_report(SQUARE_VERTS, SQUARE_TRIS)
        b = mesh_quality_report(SQUARE_VERTS, SQUARE_TRIS)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
