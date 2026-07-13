"""Tests for rotation-invariant retrieval descriptors."""

from __future__ import annotations

import math
import random
import unittest

from harnesscad.domain.reconstruction.recognize.shape_descriptors import (
    d2_shape_distribution,
    radial_shell_signature,
    pca_extents,
    bounding_volume_signature,
    descriptor_vector,
)


def _rotate_z(points, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(c * x - s * y, s * x + c * y, z) for x, y, z in points]


def _random_cloud(n, seed):
    rng = random.Random(seed)
    return [(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
            for _ in range(n)]


class D2Test(unittest.TestCase):
    def test_sums_to_one(self):
        cloud = _random_cloud(200, 1)
        hist = d2_shape_distribution(cloud, bins=16, samples=500, seed=3)
        self.assertEqual(len(hist), 16)
        self.assertAlmostEqual(sum(hist), 1.0, places=9)

    def test_deterministic(self):
        cloud = _random_cloud(100, 2)
        a = d2_shape_distribution(cloud, seed=7)
        b = d2_shape_distribution(cloud, seed=7)
        self.assertEqual(a, b)

    def test_rotation_invariant(self):
        cloud = _random_cloud(150, 5)
        rot = _rotate_z(cloud, 0.9)
        a = d2_shape_distribution(cloud, samples=800, seed=4)
        b = d2_shape_distribution(rot, samples=800, seed=4)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)

    def test_degenerate(self):
        self.assertEqual(d2_shape_distribution([(0.0, 0.0, 0.0)], bins=8), [0.0] * 8)


class RadialShellTest(unittest.TestCase):
    def test_sphere_concentrates(self):
        # points on a sphere -> all radii equal -> mass in the last bin
        pts = [(math.cos(t), math.sin(t), 0.0)
               for t in [i * 0.1 for i in range(60)]]
        hist = radial_shell_signature(pts, bins=10)
        self.assertAlmostEqual(sum(hist), 1.0, places=9)
        self.assertAlmostEqual(hist[-1], 1.0, places=9)

    def test_rotation_invariant(self):
        cloud = _random_cloud(120, 8)
        rot = _rotate_z(cloud, 1.3)
        a = radial_shell_signature(cloud, bins=12)
        b = radial_shell_signature(rot, bins=12)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=9)


class BoundingVolumeTest(unittest.TestCase):
    def test_extents_descending(self):
        pts = [(x, y, z) for x in (-2, 2) for y in (-1, 1) for z in (-0.5, 0.5)]
        e1, e2, e3 = pca_extents(pts)
        self.assertGreaterEqual(e1, e2)
        self.assertGreaterEqual(e2, e3)
        self.assertAlmostEqual(e1, 4.0, places=6)
        self.assertAlmostEqual(e2, 2.0, places=6)
        self.assertAlmostEqual(e3, 1.0, places=6)

    def test_rotation_invariant_signature(self):
        pts = _random_cloud(200, 11)
        rot = _rotate_z(pts, 0.7)
        sa = bounding_volume_signature(pts)
        sb = bounding_volume_signature(rot)
        self.assertAlmostEqual(sa["elongation"], sb["elongation"], places=6)
        self.assertAlmostEqual(sa["flatness"], sb["flatness"], places=6)
        self.assertAlmostEqual(sa["anisotropy"], sb["anisotropy"], places=6)

    def test_flat_object(self):
        rng = random.Random(3)
        flat = [(rng.uniform(-1, 1), rng.uniform(-1, 1), 0.0) for _ in range(100)]
        sig = bounding_volume_signature(flat)
        self.assertLess(sig["flatness"], 0.05)


class DescriptorVectorTest(unittest.TestCase):
    def test_length_and_rotation(self):
        cloud = _random_cloud(120, 21)
        rot = _rotate_z(cloud, 2.1)
        a = descriptor_vector(cloud, d2_bins=16, shell_bins=16, samples=600, seed=9)
        b = descriptor_vector(rot, d2_bins=16, shell_bins=16, samples=600, seed=9)
        self.assertEqual(len(a), 16 + 16 + 4)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=6)


if __name__ == "__main__":
    unittest.main()
