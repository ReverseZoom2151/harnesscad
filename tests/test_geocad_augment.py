"""Tests for GeoCAD caption-invariant geometric augmentation."""

import random
import unittest

from harnesscad.data.dataengine.augment import geocad_augment as aug
from harnesscad.domain.reconstruction.recognize.geocad_part_classifier import BRANCH_VERTEX, BRANCH_VLLM
from harnesscad.domain.geometry.sketch.geocad_vertex_caption import caption_triangle, caption_quadrilateral


class TransformTest(unittest.TestCase):
    def test_translate(self):
        self.assertEqual(aug.translate([(0, 0), (1, 1)], 2, 3), [(2, 3), (3, 4)])

    def test_scale(self):
        self.assertEqual(aug.scale([(1, 1)], 2), [(2, 2)])

    def test_scale_zero_rejected(self):
        with self.assertRaises(ValueError):
            aug.scale([(1, 1)], 0)

    def test_rotate_90(self):
        out = aug.rotate([(1, 0)], 90)
        self.assertAlmostEqual(out[0][0], 0.0)
        self.assertAlmostEqual(out[0][1], 1.0)

    def test_reflect_x(self):
        self.assertEqual(aug.reflect([(1, 2)], "x"), [(1, -2)])

    def test_reflect_diag(self):
        self.assertEqual(aug.reflect([(1, 2)], "diag"), [(2, 1)])


class PolicyTest(unittest.TestCase):
    def test_simple_policy(self):
        p = aug.policy_for_branch(BRANCH_VERTEX)
        self.assertTrue(p.rotation and p.reflection)

    def test_complex_policy(self):
        p = aug.policy_for_branch(BRANCH_VLLM)
        self.assertFalse(p.rotation or p.reflection)
        self.assertTrue(p.translation and p.scaling)


class InvarianceTest(unittest.TestCase):
    def test_triangle_caption_invariant(self):
        tri = [(0, 0), (4, 0), (0, 4)]  # isosceles right triangle
        base = caption_triangle(tri)
        rng = random.Random(7)
        for _ in range(20):
            a = aug.augment_once(tri, rng, aug.POLICY_SIMPLE)
            self.assertEqual(caption_triangle(a), base)

    def test_quad_caption_invariant(self):
        sq = [(0, 0), (2, 0), (2, 2), (0, 2)]  # square
        base = caption_quadrilateral(sq)
        rng = random.Random(11)
        for _ in range(20):
            a = aug.augment_once(sq, rng, aug.POLICY_SIMPLE)
            self.assertEqual(caption_quadrilateral(a), base)

    def test_deterministic(self):
        tri = [(0, 0), (3, 0), (0, 4)]
        a = aug.augment_batch(tri, random.Random(1), 5)
        b = aug.augment_batch(tri, random.Random(1), 5)
        self.assertEqual(a, b)

    def test_batch_count(self):
        out = aug.augment_batch([(0, 0), (1, 0), (0, 1)], random.Random(0), 4)
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
