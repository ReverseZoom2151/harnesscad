"""Tests for geometry.solidpy_extrude_along_path."""

import math
import unittest
from collections import Counter

from harnesscad.domain.geometry.solidpy_extrude_along_path import (
    centroid_endcap,
    extrude_along_path,
    extrude_along_path_scad,
    face_strip_list,
    fan_endcap_list,
    look_at_frame,
    transform_points_to_frame,
)
from harnesscad.domain.programs.scadlm_ast import parse


def square_shape(r=1.0, n=4):
    return [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def edge_counts(faces):
    c = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            c[(min(a, b), max(a, b))] += 1
    return c


class TestFrame(unittest.TestCase):
    def test_axes_orthonormal(self):
        x, y, z, o = look_at_frame((1, 2, 3), (0, 1, 0))
        for v in (x, y, z):
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in v)), 1.0)
        self.assertAlmostEqual(sum(a * b for a, b in zip(x, y)), 0.0)
        self.assertAlmostEqual(sum(a * b for a, b in zip(y, z)), 0.0)
        self.assertEqual(o, (1.0, 2.0, 3.0))

    def test_z_axis_is_reverse_normal(self):
        _, _, z, _ = look_at_frame((0, 0, 0), (0, 2, 0))
        self.assertAlmostEqual(z[1], -1.0)

    def test_parallel_to_up_does_not_collapse(self):
        # normal parallel to the default up (0,0,1): frame must stay valid
        x, y, z, _ = look_at_frame((0, 0, 0), (0, 0, 5))
        self.assertAlmostEqual(math.sqrt(sum(c * c for c in x)), 1.0)
        self.assertAlmostEqual(z[2], -1.0)

    def test_shape_plane_is_perpendicular_to_normal(self):
        pts = transform_points_to_frame([(1, 0), (0, 1)], (0, 0, 0), (1, 0, 0))
        for p in pts:
            self.assertAlmostEqual(p[0], 0.0)  # normal is +x, shape lies in yz


class TestFaceHelpers(unittest.TestCase):
    def test_face_strip_list(self):
        self.assertEqual(
            face_strip_list(0, 3, 3),
            [(0, 4, 3), (0, 1, 4), (1, 5, 4), (1, 2, 5)],
        )

    def test_face_strip_close_loop(self):
        faces = face_strip_list(0, 3, 3, close_loop=True)
        self.assertEqual(len(faces), 6)

    def test_fan_endcap_list(self):
        self.assertEqual(fan_endcap_list(6), [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)])

    def test_centroid_endcap(self):
        pts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
        center, faces = centroid_endcap(pts, [0, 1, 2, 3])
        self.assertEqual(center, (1.0, 1.0, 0.0))
        self.assertEqual(len(faces), 4)
        self.assertEqual(faces[0], (4, 0, 1))

    def test_centroid_endcap_invert(self):
        pts = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
        _, faces = centroid_endcap(pts, [0, 1, 2, 3], invert=True)
        self.assertEqual(faces[0], (1, 0, 4))


class TestExtrude(unittest.TestCase):
    def test_point_and_face_counts(self):
        shape = square_shape(n=4)
        path = [(0, 0, z) for z in range(5)]
        pts, faces = extrude_along_path(shape, path, cap_ends=False)
        self.assertEqual(len(pts), 4 * 5)
        self.assertEqual(len(faces), 2 * 4 * 4)

    def test_capped_mesh_is_edge_manifold(self):
        shape = square_shape(n=5)
        path = [(0, 0, z) for z in range(4)]
        pts, faces = extrude_along_path(shape, path, cap_ends=True)
        self.assertEqual(len(pts), 5 * 4 + 2)
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))

    def test_closed_loop_is_edge_manifold(self):
        shape = square_shape(r=1, n=6)
        path = [(4 * math.cos(t), 4 * math.sin(t), 0)
                for t in [2 * math.pi * i / 12 for i in range(12)]]
        pts, faces = extrude_along_path(shape, path, connect_ends=True)
        self.assertEqual(len(pts), 6 * 12)
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))

    def test_coincident_ends_auto_close(self):
        shape = square_shape(n=4)
        path = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0), (0, 0, 0)]
        pts, faces = extrude_along_path(shape, path)
        self.assertEqual(len(pts), 16)  # last path point dropped, no caps
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))

    def test_straight_extrusion_geometry(self):
        # A shape swept up +z keeps its cross-section size at every loop
        shape = square_shape(r=2, n=4)
        path = [(0, 0, 0), (0, 0, 10)]
        pts, _ = extrude_along_path(shape, path, cap_ends=False)
        for p in pts[:4]:
            self.assertAlmostEqual(p[2], 0.0)
        for p in pts[4:8]:
            self.assertAlmostEqual(p[2], 10.0)
        radii = [math.hypot(p[0], p[1]) for p in pts]
        for r in radii:
            self.assertAlmostEqual(r, 2.0)

    def test_scales_taper(self):
        shape = square_shape(r=2, n=4)
        path = [(0, 0, 0), (0, 0, 10)]
        pts, _ = extrude_along_path(shape, path, scales=[1.0, 0.5], cap_ends=False)
        self.assertAlmostEqual(math.hypot(pts[0][0], pts[0][1]), 2.0)
        self.assertAlmostEqual(math.hypot(pts[4][0], pts[4][1]), 1.0)

    def test_differential_scale(self):
        shape = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        path = [(0, 0, 0), (0, 0, 1)]
        pts, _ = extrude_along_path(shape, path, scales=[(2, 3), (2, 3)],
                                    cap_ends=False)
        xs = sorted(abs(p[0]) for p in pts[:4])
        ys = sorted(abs(p[1]) for p in pts[:4])
        self.assertAlmostEqual(max(xs), 2.0)
        self.assertAlmostEqual(max(ys), 3.0)

    def test_single_rotation_sweeps_smoothly(self):
        # A single rotation is spread linearly over the path: 0 at the start,
        # half at the middle, all of it at the end. Measure against the
        # unrotated sweep, since the look-at frame itself reorients the shape.
        shape = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        path = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
        base, _ = extrude_along_path(shape, path, cap_ends=False)
        pts, _ = extrude_along_path(shape, path, rotations=[90], cap_ends=False)

        def angle(p):
            return math.degrees(math.atan2(p[1], p[0]))

        for index, expected in ((0, 0.0), (4, 45.0), (8, 90.0)):
            delta = abs(angle(pts[index]) - angle(base[index])) % 360
            self.assertAlmostEqual(min(delta, 360 - delta), expected, places=5)

    def test_per_loop_rotations(self):
        shape = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        path = [(0, 0, 0), (0, 0, 1)]
        pts, _ = extrude_along_path(shape, path, rotations=[0, 90], cap_ends=False)
        self.assertAlmostEqual(pts[4][0], 0.0, places=6)
        self.assertAlmostEqual(pts[4][1], 1.0, places=6)

    def test_transform_callable(self):
        seen = []

        def bulge(p, path_fraction, loop_fraction):
            seen.append((path_fraction, loop_fraction))
            return (p[0] * (1 + path_fraction), p[1], p[2])

        shape = square_shape(r=1, n=4)
        path = [(0, 0, 0), (0, 0, 1)]
        pts, _ = extrude_along_path(shape, path, transforms=[bulge], cap_ends=False)
        self.assertEqual(seen[0][0], 0.0)
        self.assertEqual(seen[-1][0], 1.0)
        self.assertAlmostEqual(abs(pts[4][0]), 2.0)

    def test_determinism(self):
        shape = square_shape(n=5)
        path = [(0, 0, 0), (1, 1, 1), (2, 3, 1)]
        a = extrude_along_path(shape, path)
        b = extrude_along_path(shape, path)
        self.assertEqual(a, b)

    def test_validation(self):
        with self.assertRaises(ValueError):
            extrude_along_path([(0, 0), (1, 1)], [(0, 0, 0), (0, 0, 1)])
        with self.assertRaises(ValueError):
            extrude_along_path(square_shape(), [(0, 0, 0)])
        with self.assertRaises(ValueError):
            extrude_along_path(square_shape(), [(0, 0, 0), (0, 0, 1)], scales=[1])

    def test_scad_node_parses(self):
        node = extrude_along_path_scad(square_shape(n=4),
                                       [(0, 0, 0), (0, 0, 5)])
        from harnesscad.domain.programs.solidpy_scad_emit import scad_render
        src = scad_render(node)
        self.assertIn("polyhedron(", src)
        self.assertIn("convexity = 2", src)
        self.assertEqual(len(parse(src)), 1)


if __name__ == "__main__":
    unittest.main()
