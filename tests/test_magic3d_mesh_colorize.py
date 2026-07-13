"""Tests for geometry.magic3d_mesh_colorize (Magic3DSketch colorization scheme)."""

import unittest

from harnesscad.domain.geometry.magic3d_mesh_colorize import (
    barycentric,
    interpolate_color,
    sample_surface_color,
    face_centroid_color,
    mesh_average_color,
    color_from_prompt,
    nearest_named_color,
)

TRI = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
COLORS = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]


class BarycentricTest(unittest.TestCase):
    def test_vertices(self):
        u, v, w = barycentric(TRI[0], TRI)
        self.assertAlmostEqual(u, 1.0)
        self.assertAlmostEqual(v, 0.0)
        self.assertAlmostEqual(w, 0.0)
        _, v2, _ = barycentric(TRI[1], TRI)
        self.assertAlmostEqual(v2, 1.0)

    def test_centroid(self):
        cen = (1.0 / 3.0, 1.0 / 3.0, 0.0)
        u, v, w = barycentric(cen, TRI)
        for coord in (u, v, w):
            self.assertAlmostEqual(coord, 1.0 / 3.0)

    def test_sum_to_one(self):
        u, v, w = barycentric((0.2, 0.3, 0.0), TRI)
        self.assertAlmostEqual(u + v + w, 1.0)

    def test_outside_has_negative(self):
        u, v, w = barycentric((-0.5, -0.5, 0.0), TRI)
        self.assertTrue(min(u, v, w) < 0.0)

    def test_degenerate_raises(self):
        with self.assertRaises(ValueError):
            barycentric((0, 0, 0), [(0, 0, 0), (1, 0, 0), (2, 0, 0)])


class InterpolateTest(unittest.TestCase):
    def test_at_vertex(self):
        self.assertEqual(sample_surface_color(TRI[0], TRI, COLORS), (1.0, 0.0, 0.0))
        self.assertEqual(sample_surface_color(TRI[2], TRI, COLORS), (0.0, 0.0, 1.0))

    def test_centroid_blend(self):
        cen = (1.0 / 3.0, 1.0 / 3.0, 0.0)
        col = sample_surface_color(cen, TRI, COLORS)
        for c in col:
            self.assertAlmostEqual(c, 1.0 / 3.0)

    def test_edge_midpoint(self):
        mid = (0.5, 0.0, 0.0)  # between v0 (red) and v1 (green)
        col = sample_surface_color(mid, TRI, COLORS)
        self.assertAlmostEqual(col[0], 0.5)
        self.assertAlmostEqual(col[1], 0.5)
        self.assertAlmostEqual(col[2], 0.0)

    def test_interpolate_bad_length(self):
        with self.assertRaises(ValueError):
            interpolate_color([0.5, 0.5], COLORS)


class AggregateTest(unittest.TestCase):
    def test_centroid_color(self):
        self.assertEqual(
            face_centroid_color(COLORS), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        )

    def test_centroid_bad_length(self):
        with self.assertRaises(ValueError):
            face_centroid_color([(1, 0, 0), (0, 1, 0)])

    def test_mesh_average(self):
        # two faces; all vertices grey -> average grey
        vc = [(0.5, 0.5, 0.5)] * 4
        faces = [(0, 1, 2), (0, 2, 3)]
        self.assertEqual(mesh_average_color(faces, vc), (0.5, 0.5, 0.5))

    def test_mesh_average_mixed(self):
        vc = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
              (0.0, 0.0, 0.0)]
        faces = [(0, 1, 2)]  # all red -> red
        self.assertEqual(mesh_average_color(faces, vc), (1.0, 0.0, 0.0))

    def test_empty_mesh_raises(self):
        with self.assertRaises(ValueError):
            mesh_average_color([], [])


class PaletteTest(unittest.TestCase):
    def test_known_word(self):
        self.assertEqual(color_from_prompt("red"), (1.0, 0.0, 0.0))
        self.assertEqual(color_from_prompt("  BLUE "), (0.0, 0.0, 1.0))

    def test_grey_gray_alias(self):
        self.assertEqual(color_from_prompt("gray"), color_from_prompt("grey"))

    def test_unknown_default(self):
        self.assertEqual(color_from_prompt("chartreuse"), (0.5, 0.5, 0.5))
        self.assertEqual(
            color_from_prompt("chartreuse", default=(0.1, 0.2, 0.3)),
            (0.1, 0.2, 0.3),
        )

    def test_nearest_exact(self):
        self.assertEqual(nearest_named_color((1.0, 0.0, 0.0)), "red")

    def test_nearest_close(self):
        self.assertEqual(nearest_named_color((0.95, 0.05, 0.02)), "red")

    def test_nearest_deterministic_tie(self):
        # grey and gray are identical points; alphabetical first -> "gray"
        self.assertEqual(nearest_named_color((0.5, 0.5, 0.5)), "gray")


if __name__ == "__main__":
    unittest.main()
