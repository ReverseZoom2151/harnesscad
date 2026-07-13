"""Tests for datagen.datacon_spoke_augment (deterministic spoke augmentation)."""

import math
import random
import unittest

from harnesscad.data.datagen.contour_augment import (
    augment_spoke_design,
    design_is_balanced,
    few_shot_expand,
    jitter_polygon,
    mirror_polygon,
    polygon_area,
    replicate_rotational,
    rotate_polygon,
    scale_polygon,
)


BASE_SPOKE = [(1.0, 0.0), (3.0, 0.2), (3.0, -0.2)]


class TestReplicateRotational(unittest.TestCase):
    def test_n4_gives_four_polygons(self):
        polys = replicate_rotational(BASE_SPOKE, 4)
        self.assertEqual(len(polys), 4)

    def test_second_is_base_rotated_half_pi(self):
        polys = replicate_rotational(BASE_SPOKE, 4)
        expected = rotate_polygon(BASE_SPOKE, math.pi / 2)
        # Check first vertex of the 2nd polygon.
        self.assertAlmostEqual(polys[1][0][0], expected[0][0])
        self.assertAlmostEqual(polys[1][0][1], expected[0][1])
        # (1,0) rotated by pi/2 -> (0,1)
        self.assertAlmostEqual(polys[1][0][0], 0.0)
        self.assertAlmostEqual(polys[1][0][1], 1.0)

    def test_bad_order_raises(self):
        with self.assertRaises(ValueError):
            replicate_rotational(BASE_SPOKE, 0)


class TestRotate(unittest.TestCase):
    def test_full_turn_returns_original(self):
        rotated = rotate_polygon(BASE_SPOKE, 2.0 * math.pi)
        for (px, py), (ox, oy) in zip(rotated, BASE_SPOKE):
            self.assertAlmostEqual(px, ox)
            self.assertAlmostEqual(py, oy)


class TestScale(unittest.TestCase):
    def test_factor_two_doubles_distance(self):
        scaled = scale_polygon(BASE_SPOKE, 2.0)
        for (sx, sy), (ox, oy) in zip(scaled, BASE_SPOKE):
            self.assertAlmostEqual(sx, ox * 2.0)
            self.assertAlmostEqual(sy, oy * 2.0)

    def test_nonpositive_factor_raises(self):
        with self.assertRaises(ValueError):
            scale_polygon(BASE_SPOKE, 0.0)
        with self.assertRaises(ValueError):
            scale_polygon(BASE_SPOKE, -1.0)


class TestMirror(unittest.TestCase):
    def test_mirror_x_flips_y(self):
        m = mirror_polygon(BASE_SPOKE, "x")
        for (mx, my), (ox, oy) in zip(m, BASE_SPOKE):
            self.assertAlmostEqual(mx, ox)
            self.assertAlmostEqual(my, -oy)


class TestAugmentDesign(unittest.TestCase):
    def test_determinism(self):
        a = augment_spoke_design(BASE_SPOKE, 5, seed=42, n_variants=6)
        b = augment_spoke_design(BASE_SPOKE, 5, seed=42, n_variants=6)
        self.assertEqual(len(a), len(b))
        for da, db in zip(a, b):
            self.assertEqual(da["source"], db["source"])
            self.assertEqual(da["symmetry_order"], db["symmetry_order"])
            for pa, pb in zip(da["spokes"], db["spokes"]):
                for (xa, ya), (xb, yb) in zip(pa, pb):
                    self.assertAlmostEqual(xa, xb)
                    self.assertAlmostEqual(ya, yb)

    def test_first_is_base_with_order_spokes(self):
        designs = augment_spoke_design(BASE_SPOKE, 5, seed=1, n_variants=4)
        self.assertEqual(designs[0]["source"], "base")
        self.assertEqual(len(designs[0]["spokes"]), 5)
        self.assertEqual(designs[0]["symmetry_order"], 5)

    def test_all_variants_preserve_order(self):
        designs = augment_spoke_design(BASE_SPOKE, 7, seed=3, n_variants=5)
        for d in designs:
            self.assertEqual(d["symmetry_order"], 7)
            self.assertEqual(len(d["spokes"]), 7)


class TestFewShotExpand(unittest.TestCase):
    def test_exact_target_size(self):
        bases = [(BASE_SPOKE, 5), (scale_polygon(BASE_SPOKE, 1.5), 6)]
        out = few_shot_expand(bases, seed=7, target_size=13)
        self.assertEqual(len(out), 13)

    def test_deterministic(self):
        bases = [(BASE_SPOKE, 5), (scale_polygon(BASE_SPOKE, 1.5), 6)]
        a = few_shot_expand(bases, seed=7, target_size=13)
        b = few_shot_expand(bases, seed=7, target_size=13)
        self.assertEqual(len(a), len(b))
        for da, db in zip(a, b):
            self.assertEqual(da["source"], db["source"])
            for pa, pb in zip(da["spokes"], db["spokes"]):
                for (xa, ya), (xb, yb) in zip(pa, pb):
                    self.assertAlmostEqual(xa, xb)
                    self.assertAlmostEqual(ya, yb)


class TestBalance(unittest.TestCase):
    def test_replicated_design_is_balanced(self):
        design = {
            "spokes": replicate_rotational(BASE_SPOKE, 6),
            "symmetry_order": 6,
            "source": "base",
            "transforms": [],
        }
        self.assertTrue(design_is_balanced(design))

    def test_augmented_design_is_balanced(self):
        designs = augment_spoke_design(BASE_SPOKE, 5, seed=9, n_variants=5)
        for d in designs:
            self.assertTrue(design_is_balanced(d))

    def test_shoelace_area_positive(self):
        self.assertGreater(polygon_area(BASE_SPOKE), 0.0)


class TestJitter(unittest.TestCase):
    def test_same_seed_reproducible(self):
        r1 = random.Random(123)
        r2 = random.Random(123)
        j1 = jitter_polygon(BASE_SPOKE, r1, 0.05)
        j2 = jitter_polygon(BASE_SPOKE, r2, 0.05)
        for (x1, y1), (x2, y2) in zip(j1, j2):
            self.assertAlmostEqual(x1, x2)
            self.assertAlmostEqual(y1, y2)


if __name__ == "__main__":
    unittest.main()
