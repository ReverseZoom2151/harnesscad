"""Tests for geometry.scadclj_facets."""

import math
import unittest

from harnesscad.domain.geometry.scadclj_facets import (
    DEFAULT_FA,
    DEFAULT_FS,
    GRID_FINE,
    chord_error,
    circle_fragment_points,
    fragments_for_chord_error,
    fragments_for_node,
    get_fragments_from_r,
    sphere_rings,
)
from harnesscad.domain.programs.scadclj_data_ir import circle, cylinder, sphere, with_fn


class GetFragmentsTest(unittest.TestCase):
    def test_tiny_radius_is_three(self):
        self.assertEqual(get_fragments_from_r(GRID_FINE / 2), 3)
        self.assertEqual(get_fragments_from_r(0.0), 3)

    def test_fn_override(self):
        self.assertEqual(get_fragments_from_r(100, fn=64), 64)

    def test_fn_override_floor_of_three(self):
        # $fn of 1 or 2 still renders at least a triangle.
        self.assertEqual(get_fragments_from_r(100, fn=2), 3)
        self.assertEqual(get_fragments_from_r(100, fn=1), 3)

    def test_default_small_radius_fa_dominates(self):
        # r=1, defaults: min(360/12, 2*pi*1/2) = min(30, 3.14) = 3.14 -> 4
        # but max(...,5) floor applies -> ceil(max(3.14,5)) = 5
        self.assertEqual(get_fragments_from_r(1.0), 5)

    def test_default_medium_radius(self):
        # r=10: min(30, 2*pi*10/2=31.4)=30 -> ceil(max(30,5))=30
        self.assertEqual(get_fragments_from_r(10.0), 30)

    def test_default_large_radius_fs_dominates(self):
        # r=100: 2*pi*100/2 = 314.15; min(30, 314)=30 -> 30 (fa caps it)
        self.assertEqual(get_fragments_from_r(100.0), 30)

    def test_fine_fs_raises_count(self):
        # small $fs makes fragment size small -> more fragments, capped by 360/fa
        n = get_fragments_from_r(10.0, fs=0.1)
        self.assertEqual(n, 30)  # 360/12 = 30 caps

    def test_fine_fa_and_fs(self):
        # loosen both caps: fa=1 -> 360 max; fs=0.5 -> 2*pi*10/0.5=125.6
        n = get_fragments_from_r(10.0, fs=0.5, fa=1.0)
        self.assertEqual(n, 126)

    def test_inf_nan_fn(self):
        self.assertEqual(get_fragments_from_r(10.0, fn=float("inf")), 3)
        self.assertEqual(get_fragments_from_r(10.0, fn=float("nan")), 3)

    def test_deterministic(self):
        self.assertEqual(get_fragments_from_r(7.3), get_fragments_from_r(7.3))


class FragmentsForNodeTest(unittest.TestCase):
    def test_circle_default(self):
        self.assertEqual(fragments_for_node(circle(10.0)), 30)

    def test_circle_with_fn_binding(self):
        with with_fn(48):
            c = circle(10.0)
        self.assertEqual(fragments_for_node(c), 48)

    def test_sphere(self):
        with with_fn(16):
            s = sphere(5.0)
        self.assertEqual(fragments_for_node(s), 16)

    def test_cylinder_cone_uses_max_radius(self):
        c = cylinder([2.0, 20.0], 10.0, center=False)
        # uses r2=20 -> min(30, 2*pi*20/2=62.8)=30
        self.assertEqual(fragments_for_node(c), 30)

    def test_bad_node(self):
        with self.assertRaises(TypeError):
            fragments_for_node(("not", "a", "node"))


class CirclePointsTest(unittest.TestCase):
    def test_count_matches_fragments(self):
        pts = circle_fragment_points(10.0)
        self.assertEqual(len(pts), 30)

    def test_first_point_on_axis(self):
        pts = circle_fragment_points(5.0, fn=8)
        self.assertAlmostEqual(pts[0][0], 5.0)
        self.assertAlmostEqual(pts[0][1], 0.0)

    def test_ccw_order(self):
        pts = circle_fragment_points(1.0, fn=4)
        # square: (1,0),(0,1),(-1,0),(0,-1)
        self.assertAlmostEqual(pts[1][0], 0.0)
        self.assertAlmostEqual(pts[1][1], 1.0)

    def test_all_on_radius(self):
        pts = circle_fragment_points(3.0, fn=12)
        for x, y in pts:
            self.assertAlmostEqual(math.hypot(x, y), 3.0)


class SphereRingsTest(unittest.TestCase):
    def test_rings(self):
        frags, rings = sphere_rings(5.0, fn=16)
        self.assertEqual(frags, 16)
        self.assertEqual(rings, 8)  # (16+1)//2

    def test_odd_fragments(self):
        frags, rings = sphere_rings(5.0, fn=15)
        self.assertEqual(rings, 8)  # (15+1)//2


class ChordErrorTest(unittest.TestCase):
    def test_error_decreases_with_fragments(self):
        self.assertGreater(chord_error(10.0, 6), chord_error(10.0, 60))

    def test_known_value(self):
        # hexagon inscribed in r=1: sagitta = 1 - cos(180/6 deg) = 1 - cos(30deg)
        self.assertAlmostEqual(chord_error(1.0, 6), 1.0 - math.cos(math.pi / 6))

    def test_too_few(self):
        with self.assertRaises(ValueError):
            chord_error(1.0, 2)

    def test_roundtrip_inverse(self):
        r = 10.0
        n = fragments_for_chord_error(r, 0.05)
        # the chosen n must actually meet the tolerance...
        self.assertLessEqual(chord_error(r, n), 0.05)
        # ...and n-1 must not (n is the smallest sufficient count)
        self.assertGreater(chord_error(r, n - 1), 0.05)

    def test_huge_tolerance(self):
        self.assertEqual(fragments_for_chord_error(1.0, 5.0), 3)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            fragments_for_chord_error(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
