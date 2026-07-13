import math
import unittest

from harnesscad.domain.drawings.creft_projection import Box
from harnesscad.domain.drawings.t2cadtd_isometric_projection import (
    PAPER_AZIMUTH_DEG, PAPER_ELEVATION_DEG, TRUE_ISO_ELEVATION_DEG,
    project_point, projected_axis_vectors, axis_foreshortening,
    project_box, convex_hull, isometric_outline, bounding_box_2d,
    visible_faces, face_spanning_axes, normalize_longest_edge,
)


class TestProjectPoint(unittest.TestCase):
    def test_origin_maps_to_origin(self):
        self.assertEqual(project_point((0.0, 0.0, 0.0)), (0.0, 0.0))

    def test_linear(self):
        # Projection is linear: proj(2p) == 2*proj(p).
        p = (1.3, -2.1, 0.7)
        u1, v1 = project_point(p)
        u2, v2 = project_point((2 * p[0], 2 * p[1], 2 * p[2]))
        self.assertAlmostEqual(u2, 2 * u1)
        self.assertAlmostEqual(v2, 2 * v1)

    def test_vertical_axis_has_no_horizontal_component(self):
        # A point purely along +Z projects onto the vertical screen axis only.
        u, v = project_point((0.0, 0.0, 1.0))
        self.assertAlmostEqual(u, 0.0)
        self.assertGreater(v, 0.0)


class TestForeshortening(unittest.TestCase):
    def test_true_iso_equal_foreshortening(self):
        f = axis_foreshortening(45.0, TRUE_ISO_ELEVATION_DEG)
        self.assertAlmostEqual(f["x"], f["y"])
        self.assertAlmostEqual(f["y"], f["z"])

    def test_paper_viewpoint_not_all_equal(self):
        # At 45/45 the vertical axis foreshortens differently from x/y.
        f = axis_foreshortening(PAPER_AZIMUTH_DEG, PAPER_ELEVATION_DEG)
        self.assertAlmostEqual(f["x"], f["y"])
        self.assertNotAlmostEqual(f["z"], f["x"])

    def test_axis_vectors_match_foreshortening(self):
        vecs = projected_axis_vectors()
        f = axis_foreshortening()
        for k in ("x", "y", "z"):
            self.assertAlmostEqual(math.hypot(*vecs[k]), f[k])


class TestBoxProjection(unittest.TestCase):
    def test_eight_corners(self):
        self.assertEqual(len(project_box(Box(0, 0, 0, 2, 3, 4))), 8)

    def test_outline_is_hexagon_for_generic_box(self):
        hull = isometric_outline(Box(0, 0, 0, 2, 3, 4))
        self.assertEqual(len(hull), 6)

    def test_convex_hull_ccw(self):
        hull = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
        self.assertEqual(len(hull), 4)  # interior point dropped

    def test_bounding_box(self):
        pts = project_box(Box(0, 0, 0, 1, 1, 1))
        umin, vmin, umax, vmax = bounding_box_2d(pts)
        self.assertLess(umin, umax)
        self.assertLess(vmin, vmax)


class TestVisibleFaces(unittest.TestCase):
    def test_three_visible_faces_at_paper_viewpoint(self):
        faces = visible_faces()
        self.assertEqual(len(faces), 3)
        self.assertEqual(set(faces), {("x", 1), ("y", 1), ("z", 1)})

    def test_faces_cover_all_three_planes(self):
        spans = {face_spanning_axes(axis) for axis, _ in visible_faces()}
        self.assertEqual(spans, {("y", "z"), ("x", "z"), ("x", "y")})

    def test_face_spanning_axes(self):
        self.assertEqual(face_spanning_axes("z"), ("x", "y"))


class TestNormalizeLongestEdge(unittest.TestCase):
    def test_longest_edge_becomes_target(self):
        b = normalize_longest_edge(Box(0, 0, 0, 1, 5, 3), target=2.0)
        self.assertAlmostEqual(max(b.dx, b.dy, b.dz), 2.0)

    def test_proportions_preserved(self):
        b = normalize_longest_edge(Box(0, 0, 0, 1, 5, 3), target=2.0)
        self.assertAlmostEqual(b.dx / b.dy, 1.0 / 5.0)
        self.assertAlmostEqual(b.dz / b.dy, 3.0 / 5.0)

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            normalize_longest_edge(Box(0, 0, 0, 1, 1, 1), target=0.0)


if __name__ == "__main__":
    unittest.main()
