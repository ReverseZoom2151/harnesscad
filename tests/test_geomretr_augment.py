"""Tests for the seeded VICReg augmentation pipeline."""

from __future__ import annotations

import math
import random
import unittest

from harnesscad.domain.reconstruction.recognize.pointcloud_augment import (
    normalize_unit_sphere,
    subsample,
    rotate,
    anisotropic_scale,
    isotropic_scale,
    jitter,
    translate,
    augment,
    positive_pair,
)


def _cloud(n, seed):
    rng = random.Random(seed)
    return [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))
            for _ in range(n)]


class NormalizeTest(unittest.TestCase):
    def test_unit_sphere(self):
        norm = normalize_unit_sphere(_cloud(50, 1))
        maxr = max(math.sqrt(x * x + y * y + z * z) for x, y, z in norm)
        self.assertAlmostEqual(maxr, 1.0, places=9)
        # centroid near origin
        cx = sum(p[0] for p in norm) / len(norm)
        self.assertAlmostEqual(cx, 0.0, places=6)

    def test_empty(self):
        self.assertEqual(normalize_unit_sphere([]), [])


class SubsampleTest(unittest.TestCase):
    def test_count(self):
        rng = random.Random(0)
        out = subsample(_cloud(100, 2), 30, rng)
        self.assertEqual(len(out), 30)

    def test_no_upsample(self):
        rng = random.Random(0)
        cloud = _cloud(10, 3)
        self.assertEqual(len(subsample(cloud, 50, rng)), 10)


class TransformTest(unittest.TestCase):
    def test_rotation_preserves_norm(self):
        cloud = normalize_unit_sphere(_cloud(40, 4))
        rng = random.Random(1)
        rot = rotate(cloud, rng)
        for a, b in zip(cloud, rot):
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            self.assertAlmostEqual(na, nb, places=9)

    def test_single_axis_keeps_z(self):
        cloud = _cloud(30, 5)
        rng = random.Random(2)
        rot = rotate(cloud, rng, single_axis=True)
        for a, b in zip(cloud, rot):
            self.assertAlmostEqual(a[2], b[2], places=9)

    def test_isotropic_scale_ratio(self):
        cloud = [(1.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
        rng = random.Random(7)
        out = isotropic_scale(cloud, rng)
        # ratio of coordinates preserved
        self.assertAlmostEqual(out[1][1] / out[0][0], 2.0, places=9)

    def test_anisotropic_scale_within_bounds(self):
        cloud = [(1.0, 1.0, 1.0)]
        rng = random.Random(8)
        out = anisotropic_scale(cloud, rng)
        for v in out[0]:
            self.assertGreaterEqual(v, 0.7)
            self.assertLessEqual(v, 1.25)

    def test_jitter_clamped(self):
        cloud = [(0.0, 0.0, 0.0)] * 200
        rng = random.Random(9)
        out = jitter(cloud, rng)
        for p in out:
            for v in p:
                self.assertLessEqual(abs(v), 0.05)

    def test_translate_shifts_all_equally(self):
        cloud = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        rng = random.Random(11)
        out = translate(cloud, rng)
        d0 = tuple(out[0][d] - cloud[0][d] for d in range(3))
        d1 = tuple(out[1][d] - cloud[1][d] for d in range(3))
        for a, b in zip(d0, d1):
            self.assertAlmostEqual(a, b, places=9)


class PositivePairTest(unittest.TestCase):
    def test_deterministic(self):
        cloud = _cloud(60, 12)
        a1, b1 = positive_pair(cloud, seed=42)
        a2, b2 = positive_pair(cloud, seed=42)
        self.assertEqual(a1, a2)
        self.assertEqual(b1, b2)

    def test_views_differ(self):
        cloud = _cloud(60, 13)
        a, b = positive_pair(cloud, seed=1)
        self.assertNotEqual(a, b)

    def test_augment_no_rotation(self):
        cloud = normalize_unit_sphere(_cloud(20, 14))
        rng = random.Random(3)
        out = augment(cloud, rng, rotate_enabled=False)
        self.assertEqual(len(out), len(cloud))


if __name__ == "__main__":
    unittest.main()
