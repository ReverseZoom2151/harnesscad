"""Tests for geometry.solidpy_screw_thread."""

import math
import unittest
from collections import Counter

from harnesscad.domain.geometry.features.screw_thread import (
    default_thread_section,
    map_segment,
    thread,
    thread_scad,
)
from harnesscad.domain.programs.ast.openscad import parse
from harnesscad.domain.programs.emit.openscad_emit import scad_render

SECTION = default_thread_section(tooth_height=2.0, tooth_depth=1.0)


def edge_counts(faces):
    c = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            c[(min(a, b), max(a, b))] += 1
    return c


class TestMapSegment(unittest.TestCase):
    def test_linear_remap(self):
        self.assertAlmostEqual(map_segment(5, 0, 10, 0, 100), 50.0)
        self.assertAlmostEqual(map_segment(0, 0, 10, 3, 7), 3.0)
        self.assertAlmostEqual(map_segment(10, 0, 10, 3, 7), 7.0)

    def test_degenerate_domain(self):
        self.assertEqual(map_segment(5, 2, 2, 1, 9), 1)
        self.assertEqual(map_segment(5, 0, 10, 4, 4), 4)


class TestSection(unittest.TestCase):
    def test_default_section(self):
        self.assertEqual(default_thread_section(2, 1),
                         [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0)])


class TestThread(unittest.TestCase):
    def test_point_and_face_counts(self):
        pts, faces = thread(SECTION, inner_rad=5, pitch=2, length=4,
                            segments_per_rot=8)
        rotations = 4 / 2
        steps = math.ceil(rotations * 8) + 1
        self.assertEqual(len(pts), steps * 3)
        self.assertEqual(len(faces), 2 * 3 * (steps - 1) + 2 * (3 - 2))

    def test_mesh_is_edge_manifold(self):
        pts, faces = thread(SECTION, inner_rad=5, pitch=2, length=4,
                            segments_per_rot=12)
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))

    def test_square_section_manifold(self):
        section = [(0, -1), (1, -1), (1, 1), (0, 1)]
        _, faces = thread(section, inner_rad=4, pitch=3, length=6,
                          segments_per_rot=10)
        self.assertTrue(all(c == 2 for c in edge_counts(faces).values()))

    def test_helix_climbs_and_wraps(self):
        pts, _ = thread(SECTION, inner_rad=5, pitch=2, length=4,
                        segments_per_rot=8)
        self.assertAlmostEqual(min(p[2] for p in pts), -1.0)  # tooth half-height
        self.assertAlmostEqual(max(p[2] for p in pts), 5.0)   # length + half height
        # after one full revolution (8 segments) the profile is back at angle 0,
        # one pitch higher
        first = pts[1]           # tip of the tooth, step 0
        after_one_rot = pts[8 * 3 + 1]
        self.assertAlmostEqual(after_one_rot[0], first[0], places=6)
        self.assertAlmostEqual(after_one_rot[1], first[1], places=6)
        self.assertAlmostEqual(after_one_rot[2] - first[2], 2.0, places=6)

    def test_external_tooth_points_outward(self):
        pts, _ = thread(SECTION, inner_rad=10, pitch=2, length=2,
                        segments_per_rot=8)
        tip = pts[1]
        self.assertGreater(math.hypot(tip[0], tip[1]), 10.0)

    def test_internal_tooth_points_inward(self):
        pts, _ = thread(SECTION, inner_rad=10, pitch=2, length=2,
                        segments_per_rot=8, external=False)
        tip = pts[1]
        self.assertLess(math.hypot(tip[0], tip[1]), 10.0)

    def test_neck_in_ramps_radius(self):
        pts, _ = thread(SECTION, inner_rad=10, pitch=4, length=8,
                        segments_per_rot=16, neck_in_degrees=90,
                        neck_out_degrees=90)
        first_tip = pts[1]
        mid_tip = pts[(len(pts) // 3 // 2) * 3 + 1]
        # the thread starts sunk into the shaft and reaches full depth later
        self.assertLess(math.hypot(first_tip[0], first_tip[1]),
                        math.hypot(mid_tip[0], mid_tip[1]))

    def test_conical_thread_radius_grows(self):
        pts, _ = thread(SECTION, inner_rad=5, pitch=2, length=6,
                        segments_per_rot=12, rad_2=10)
        r_first = math.hypot(pts[1][0], pts[1][1])
        r_last = math.hypot(pts[-2][0], pts[-2][1])
        self.assertGreater(r_last, r_first + 3.0)

    def test_left_handed_reverses_sweep_and_winding(self):
        right, r_faces = thread(SECTION, inner_rad=5, pitch=2, length=4,
                                segments_per_rot=8)
        left, l_faces = thread(SECTION, inner_rad=5, pitch=2, length=4,
                               segments_per_rot=8, inverse_thread_direction=True)
        # mirrored about the XZ plane
        for a, b in zip(right, left):
            self.assertAlmostEqual(a[0], b[0], places=9)
            self.assertAlmostEqual(a[1], -b[1], places=9)
            self.assertAlmostEqual(a[2], b[2], places=9)
        self.assertEqual(l_faces[0], tuple(reversed(r_faces[0])))
        self.assertTrue(all(c == 2 for c in edge_counts(l_faces).values()))

    def test_validation(self):
        with self.assertRaises(ValueError):
            thread([(0, 0), (1, 1)], 5, 2, 4)
        with self.assertRaises(ValueError):
            thread(SECTION, 5, 0, 4)
        with self.assertRaises(ValueError):
            thread(SECTION, 5, 2, 4, segments_per_rot=2)
        with self.assertRaises(ValueError):
            thread(SECTION, 5, 2, 4, neck_in_degrees=400, neck_out_degrees=400)

    def test_determinism(self):
        a = thread(SECTION, 5, 2, 4, segments_per_rot=9)
        b = thread(SECTION, 5, 2, 4, segments_per_rot=9)
        self.assertEqual(a, b)


class TestThreadScad(unittest.TestCase):
    def test_external_is_intersected_with_tube(self):
        node = thread_scad(SECTION, inner_rad=5, pitch=2, length=4,
                           segments_per_rot=8)
        src = scad_render(node)
        self.assertTrue(src.startswith("intersection() {"))
        self.assertIn("polyhedron(", src)
        self.assertIn("difference()", src)
        self.assertEqual(src.count("cylinder("), 2)
        self.assertEqual(len(parse(src)), 1)

    def test_internal_is_intersected_with_solid_cylinder(self):
        node = thread_scad(SECTION, inner_rad=5, pitch=2, length=4,
                           segments_per_rot=8, external=False)
        src = scad_render(node)
        self.assertNotIn("difference()", src)
        self.assertEqual(src.count("cylinder("), 1)
        self.assertEqual(len(parse(src)), 1)


if __name__ == "__main__":
    unittest.main()
