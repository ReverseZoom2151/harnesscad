"""Tests for geometry.proccad_serpentine (procedural MEMS serpentine spring)."""

import unittest

from harnesscad.domain.geometry.features.proccad_serpentine import (
    beam_endpoints,
    bounding_box,
    cross_braces,
    mirror_x,
    serpentine_polyline,
    serpentine_segments,
    wire_length,
)


class SerpentineShapeTest(unittest.TestCase):
    def test_vertex_count(self):
        # n_turns=2 -> 3 beams -> 3 horizontal + 2 riser segments = 5 segments,
        # 6 vertices.
        poly = serpentine_polyline(2, 10.0, 2.0)
        self.assertEqual(len(poly), 6)
        self.assertEqual(len(serpentine_segments(poly)), 5)

    def test_beam_count_scales_with_turns(self):
        for n in range(1, 6):
            poly = serpentine_polyline(n, 5.0, 1.0)
            # beams = n+1, risers = n -> vertices = 1 + 2*(n+1) - 1
            self.assertEqual(len(poly), 2 * (n + 1))

    def test_wire_length_formula(self):
        poly = serpentine_polyline(3, 10.0, 2.0)
        expected = (3 + 1) * 10.0 + 3 * 2.0  # beams*length + turns*pitch
        self.assertAlmostEqual(wire_length(poly), expected)

    def test_boustrophedon_alternates_direction(self):
        poly = serpentine_polyline(3, 10.0, 2.0, origin=(0.0, 0.0))
        # first beam goes +x to x=10, second (after riser) goes back to x=0
        self.assertAlmostEqual(poly[1][0], 10.0)
        self.assertAlmostEqual(poly[3][0], 0.0)

    def test_bounding_box(self):
        poly = serpentine_polyline(2, 10.0, 2.0, origin=(0.0, 0.0))
        (xmin, ymin), (xmax, ymax) = bounding_box(poly)
        self.assertAlmostEqual(xmin, 0.0)
        self.assertAlmostEqual(xmax, 10.0)
        self.assertAlmostEqual(ymin, 0.0)
        self.assertAlmostEqual(ymax, 4.0)  # 2 turns * pitch 2

    def test_start_dir_negative(self):
        poly = serpentine_polyline(1, 4.0, 1.0, start_dir=-1)
        self.assertAlmostEqual(poly[1][0], -4.0)

    def test_deterministic(self):
        self.assertEqual(
            serpentine_polyline(4, 3.0, 0.5), serpentine_polyline(4, 3.0, 0.5)
        )

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            serpentine_polyline(0, 10.0, 2.0)
        with self.assertRaises(ValueError):
            serpentine_polyline(2, -1.0, 2.0)


class SymmetricPairTest(unittest.TestCase):
    def test_mirror_x_reflects(self):
        poly = serpentine_polyline(2, 10.0, 2.0, origin=(0.0, 0.0))
        mirrored = mirror_x(poly, axis_x=20.0)
        for (x, _), (mx, _) in zip(poly, mirrored):
            self.assertAlmostEqual(mx, 40.0 - x)

    def test_mirror_preserves_y(self):
        poly = serpentine_polyline(2, 10.0, 2.0)
        mirrored = mirror_x(poly, axis_x=5.0)
        for p, m in zip(poly, mirrored):
            self.assertAlmostEqual(p[1], m[1])

    def test_cross_braces_connect_corresponding_vertices(self):
        left = serpentine_polyline(2, 10.0, 2.0)
        right = mirror_x(left, axis_x=20.0)
        braces = cross_braces(left, right, indices=[0, 2, 4])
        self.assertEqual(len(braces), 3)
        # each brace is horizontal (same y) because mirror preserves y
        for (a, b) in braces:
            self.assertAlmostEqual(a[1], b[1])

    def test_cross_braces_mismatched_counts(self):
        left = serpentine_polyline(2, 10.0, 2.0)
        right = serpentine_polyline(3, 10.0, 2.0)
        with self.assertRaises(ValueError):
            cross_braces(left, right, indices=[0])

    def test_beam_endpoints_are_extremes(self):
        poly = serpentine_polyline(3, 10.0, 2.0)
        ends = beam_endpoints(poly)
        for x, _ in ends:
            self.assertTrue(abs(x - 0.0) < 1e-6 or abs(x - 10.0) < 1e-6)


if __name__ == "__main__":
    unittest.main()
