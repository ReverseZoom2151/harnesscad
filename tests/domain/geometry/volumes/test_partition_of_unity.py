"""Tests for geometry.octfusion_mpu."""

import unittest

from harnesscad.domain.geometry.volumes.octree import Octree
from harnesscad.domain.geometry.volumes.partition_of_unity import (
    bspline_weight,
    leaf_records_from_octree,
    local_coords,
    mpu_blend,
    mpu_weights,
)


class TestLocalCoords(unittest.TestCase):
    def test_transform(self):
        x = local_coords((0.6, 0.5, 0.5), (0.5, 0.5, 0.5), 0.25)
        self.assertAlmostEqual(x[0], 0.4)
        self.assertAlmostEqual(x[1], 0.0)
        self.assertAlmostEqual(x[2], 0.0)

    def test_invalid_r(self):
        with self.assertRaises(ValueError):
            local_coords((0, 0, 0), (0, 0, 0), 0.0)


class TestBspline(unittest.TestCase):
    def test_center_is_one(self):
        self.assertEqual(bspline_weight((0.0, 0.0, 0.0)), 1.0)

    def test_tent_decays(self):
        self.assertAlmostEqual(bspline_weight((0.5, 0.0, 0.0)), 0.5)

    def test_zero_outside_support(self):
        self.assertEqual(bspline_weight((1.0, 0.0, 0.0)), 0.0)
        self.assertEqual(bspline_weight((0.0, 1.5, 0.0)), 0.0)

    def test_separable(self):
        self.assertAlmostEqual(bspline_weight((0.5, 0.5, 0.0)), 0.25)


class TestMpuBlend(unittest.TestCase):
    def _grid(self):
        # regular 1D grid of leaves along x at 0,1,2 (r=1), y=z=0
        return [((float(i), 0.0, 0.0), 1.0, i) for i in range(3)]

    def test_constant_reproduction(self):
        leaves = self._grid()
        val = mpu_blend((0.3, 0.0, 0.0), leaves, lambda x, p: 7.0)
        self.assertAlmostEqual(val, 7.0)

    def test_linear_reproduction(self):
        # linear field f(x) = 2x + 1; each node's local field returns f(center)
        leaves = [((float(i), 0.0, 0.0), 1.0, 2.0 * i + 1.0) for i in range(3)]
        val = mpu_blend((0.3, 0.0, 0.0), leaves, lambda x, p: p)
        self.assertAlmostEqual(val, 2.0 * 0.3 + 1.0)  # = 1.6

    def test_default_when_uncovered(self):
        leaves = self._grid()
        val = mpu_blend((10.0, 0.0, 0.0), leaves, lambda x, p: 1.0, default=-99.0)
        self.assertEqual(val, -99.0)

    def test_phi_receives_local_coords(self):
        leaves = [((0.0, 0.0, 0.0), 1.0, None)]
        seen = {}

        def phi(x, p):
            seen["x"] = x
            return 0.0

        mpu_blend((0.2, 0.0, 0.0), leaves, phi)
        self.assertAlmostEqual(seen["x"][0], 0.2)


class TestMpuWeights(unittest.TestCase):
    def test_partition_of_unity(self):
        leaves = [((float(i), 0.0, 0.0), 1.0, i) for i in range(3)]
        w = mpu_weights((0.3, 0.0, 0.0), leaves)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertAlmostEqual(w[0], 0.7)
        self.assertAlmostEqual(w[1], 0.3)
        self.assertAlmostEqual(w[2], 0.0)

    def test_all_zero_when_uncovered(self):
        leaves = [((0.0, 0.0, 0.0), 1.0, 0)]
        self.assertEqual(mpu_weights((5.0, 0.0, 0.0), leaves), [0.0])


class TestFromOctree(unittest.TestCase):
    def test_records_and_constant_blend(self):
        t = Octree.from_points([(0.1, 0.1, 0.1), (0.9, 0.9, 0.9)], max_depth=2)
        recs = leaf_records_from_octree(t, payload_of=lambda leaf: leaf.key())
        self.assertEqual(len(recs), t.occupied_leaf_count())
        # blend a constant field at an occupied leaf center reproduces the constant
        center = recs[0][0]
        val = mpu_blend(center, recs, lambda x, p: 3.5)
        self.assertAlmostEqual(val, 3.5)

    def test_all_leaves_option(self):
        t = Octree.from_points([(0.1, 0.1, 0.1)], max_depth=1)
        recs = leaf_records_from_octree(t, payload_of=lambda leaf: 0, occupied_only=False)
        self.assertEqual(len(recs), t.leaf_count())


if __name__ == "__main__":
    unittest.main()
