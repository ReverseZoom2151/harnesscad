"""Tests for geometry.solidpy_catmull_rom."""

import math
import unittest
from collections import Counter

from harnesscad.domain.geometry.solidpy_catmull_rom import (
    affine_combination,
    catmull_rom_patch,
    catmull_rom_points,
    catmull_rom_polygon,
    catmull_rom_prism,
    catmull_rom_prism_scad,
    catmull_rom_segment,
    centroid,
)
from harnesscad.domain.programs.scadlm_ast import parse
from harnesscad.domain.programs.solidpy_scad_emit import scad_render


def edge_counts(faces):
    c = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            c[(min(a, b), max(a, b))] += 1
    return c


def ring(radius, n, z=0.0):
    return [(radius * math.cos(2 * math.pi * i / n),
             radius * math.sin(2 * math.pi * i / n), z) for i in range(n)]


class TestHelpers(unittest.TestCase):
    def test_affine_combination(self):
        self.assertEqual(affine_combination((0, 0), (10, 20), 0.5), (5.0, 10.0, 0.0))
        self.assertEqual(affine_combination((0, 0, 0), (2, 2, 2), 0.0), (0.0, 0.0, 0.0))

    def test_centroid(self):
        self.assertEqual(centroid([(0, 0), (2, 0), (2, 2), (0, 2)]), (1.0, 1.0, 0.0))

    def test_centroid_empty(self):
        with self.assertRaises(ValueError):
            centroid([])


class TestSegment(unittest.TestCase):
    def test_segment_endpoints(self):
        controls = [(0, 0), (1, 0), (2, 0), (3, 0)]
        pts = catmull_rom_segment(controls, 4, include_last=True)
        self.assertEqual(len(pts), 5)
        self.assertAlmostEqual(pts[0][0], 1.0)
        self.assertAlmostEqual(pts[-1][0], 2.0)

    def test_segment_excludes_last(self):
        controls = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(len(catmull_rom_segment(controls, 4)), 4)

    def test_colinear_controls_stay_straight(self):
        controls = [(0, 0), (1, 0), (2, 0), (3, 0)]
        for p in catmull_rom_segment(controls, 8, include_last=True):
            self.assertAlmostEqual(p[1], 0.0)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            catmull_rom_segment([(0, 0), (1, 0)], 4)
        with self.assertRaises(ValueError):
            catmull_rom_segment([(0, 0)] * 4, 0)


class TestPoints(unittest.TestCase):
    def test_interpolates_control_points(self):
        controls = [(0, 0), (1, 3), (2, -1), (4, 2)]
        subdivisions = 5
        curve = catmull_rom_points(controls, subdivisions)
        self.assertEqual(len(curve), (len(controls) - 1) * subdivisions + 1)
        for i, c in enumerate(controls):
            p = curve[i * subdivisions]
            self.assertAlmostEqual(p[0], c[0], places=9)
            self.assertAlmostEqual(p[1], c[1], places=9)

    def test_closed_loop_length_and_no_duplicate(self):
        controls = [(0, 0), (10, 0), (10, 10), (0, 10)]
        curve = catmull_rom_points(controls, 4, close_loop=True)
        self.assertEqual(len(curve), 4 * 4)
        self.assertNotEqual(curve[0], curve[-1])
        for i, c in enumerate(controls):
            self.assertAlmostEqual(curve[i * 4][0], c[0], places=9)
            self.assertAlmostEqual(curve[i * 4][1], c[1], places=9)

    def test_closed_loop_is_symmetric(self):
        # a square's CR ring must be symmetric about the diagonal
        controls = [(0, 0), (10, 0), (10, 10), (0, 10)]
        curve = catmull_rom_points(controls, 6, close_loop=True)
        for p in curve:
            mirrored = (p[1], p[0], p[2])
            self.assertTrue(
                any(math.dist(mirrored, q) < 1e-9 for q in curve))

    def test_explicit_tangents(self):
        controls = [(0, 0), (1, 0), (2, 0)]
        a = catmull_rom_points(controls, 4, start_tangent=(0, 5, 0))
        b = catmull_rom_points(controls, 4)
        self.assertNotAlmostEqual(a[1][1], b[1][1])
        # endpoints are unaffected
        self.assertAlmostEqual(a[0][0], 0.0)
        self.assertAlmostEqual(a[-1][0], 2.0)

    def test_determinism(self):
        controls = [(0, 0), (1, 3), (2, -1)]
        self.assertEqual(catmull_rom_points(controls, 7),
                         catmull_rom_points(controls, 7))

    def test_validation(self):
        with self.assertRaises(ValueError):
            catmull_rom_points([(0, 0)], 4)
        with self.assertRaises(ValueError):
            catmull_rom_points([(0, 0), (1, 1)], 4, close_loop=True)

    def test_polygon_node(self):
        node = catmull_rom_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], 3)
        src = scad_render(node)
        self.assertTrue(src.startswith("polygon("))
        self.assertEqual(len(parse(src)), 1)


class TestPatch(unittest.TestCase):
    def test_patch_shape(self):
        a = [(0, 0, 0), (5, 0, 2), (10, 0, 0)]
        b = [(0, 10, 0), (5, 10, 2), (10, 10, 0)]
        subdivisions = 4
        verts, faces = catmull_rom_patch(a, b, subdivisions)
        strip = (len(a) - 1) * subdivisions + 1
        self.assertEqual(len(verts), strip * (subdivisions + 1))
        self.assertEqual(len(faces), 2 * (strip - 1) * subdivisions)

    def test_patch_rows_span_the_curves(self):
        a = [(0, 0, 0), (5, 0, 0), (10, 0, 0)]
        b = [(0, 10, 0), (5, 10, 0), (10, 10, 0)]
        verts, _ = catmull_rom_patch(a, b, 2)
        self.assertAlmostEqual(verts[0][1], 0.0)
        self.assertAlmostEqual(verts[-1][1], 10.0)

    def test_mismatched_curves(self):
        with self.assertRaises(ValueError):
            catmull_rom_patch([(0, 0), (1, 0)], [(0, 1), (1, 1), (2, 1)], 3)


class TestPrism(unittest.TestCase):
    def _curves(self, n=4):
        # n vertical control curves arranged around a ring
        curves = []
        for i in range(n):
            theta = 2 * math.pi * i / n
            x, y = math.cos(theta) * 5, math.sin(theta) * 5
            curves.append([(x, y, 0), (x * 1.5, y * 1.5, 5), (x, y, 10)])
        return curves

    def test_capped_prism_is_edge_manifold(self):
        verts, faces = catmull_rom_prism(self._curves(), subdivisions=3)
        counts = edge_counts(faces)
        self.assertTrue(all(c == 2 for c in counts.values()))

    def test_vertex_count(self):
        subdivisions = 3
        curves = self._curves(4)
        verts, _ = catmull_rom_prism(curves, subdivisions=subdivisions)
        height = (len(curves[0]) - 1) * subdivisions + 1
        width = len(curves) * subdivisions
        self.assertEqual(len(verts), height * width + 2)  # + 2 cap centroids

    def test_open_ring_has_no_caps(self):
        verts, faces = catmull_rom_prism(self._curves(), subdivisions=2,
                                         closed_ring=False)
        counts = edge_counts(faces)
        # an open sheet has boundary edges used only once
        self.assertTrue(any(c == 1 for c in counts.values()))

    def test_no_caps_option(self):
        curves = self._curves()
        verts, _ = catmull_rom_prism(curves, subdivisions=2, add_caps=False)
        height = (len(curves[0]) - 1) * 2 + 1
        self.assertEqual(len(verts), height * len(curves) * 2)

    def test_smooth_edges_is_manifold_and_rounder(self):
        curves = self._curves(4)
        verts, faces = catmull_rom_prism(curves, subdivisions=4, smooth_edges=True)
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))
        # smoothing around the ring pushes the mid-edge points outward compared
        # to the straight (linear) interpolation of the same controls
        linear, _ = catmull_rom_prism(curves, subdivisions=4, smooth_edges=False)
        r_smooth = max(math.hypot(p[0], p[1]) for p in verts)
        r_linear = max(math.hypot(p[0], p[1]) for p in linear)
        self.assertGreaterEqual(r_smooth, r_linear - 1e-9)

    def test_control_curve_validation(self):
        with self.assertRaises(ValueError):
            catmull_rom_prism([[(0, 0, 0), (0, 0, 1)]], subdivisions=2)
        with self.assertRaises(ValueError):
            catmull_rom_prism([[(0, 0, 0), (0, 0, 1)],
                               [(1, 0, 0), (1, 0, 1), (1, 0, 2)]], subdivisions=2)

    def test_scad_node_parses(self):
        node = catmull_rom_prism_scad(self._curves(), subdivisions=2)
        src = scad_render(node)
        self.assertIn("polyhedron(", src)
        self.assertEqual(len(parse(src)), 1)

    def test_determinism(self):
        curves = self._curves()
        self.assertEqual(catmull_rom_prism(curves, subdivisions=3),
                         catmull_rom_prism(curves, subdivisions=3))


if __name__ == "__main__":
    unittest.main()
