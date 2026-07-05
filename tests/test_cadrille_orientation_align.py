"""Tests for orientation-invariant (discrete-ICP) alignment."""

import unittest

from bench.cadrille_orientation_align import (
    proper_axis_rotations,
    centroid,
    apply_rotation,
    align_orientation,
    aligned_chamfer,
    _det3,
)


class RotationSetTest(unittest.TestCase):
    def test_count_is_24(self):
        self.assertEqual(len(proper_axis_rotations()), 24)

    def test_all_proper(self):
        for m in proper_axis_rotations():
            self.assertEqual(_det3(m), 1)

    def test_unique(self):
        mats = proper_axis_rotations()
        self.assertEqual(len(set(mats)), 24)


class GeometryTest(unittest.TestCase):
    def test_centroid(self):
        self.assertEqual(centroid([(0.0, 0.0, 0.0), (2.0, 4.0, 6.0)]),
                         (1.0, 2.0, 3.0))

    def test_apply_identity(self):
        ident = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        pts = [(1.0, 2.0, 3.0)]
        self.assertEqual(apply_rotation(pts, ident), pts)

    def test_apply_z_rotation(self):
        # 90deg about z: (x,y,z) -> (-y, x, z)
        rot = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
        self.assertEqual(apply_rotation([(1.0, 0.0, 5.0)], rot),
                         [(0.0, 1.0, 5.0)])


class AlignTest(unittest.TestCase):
    def _shape(self):
        # an L-shaped, orientation-distinguishable point set
        return [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)]

    def test_rotated_copy_aligns_to_zero(self):
        src = self._shape()
        rot = ((0, -1, 0), (1, 0, 0), (0, 0, 1))  # 90deg z
        target = apply_rotation(src, rot)
        result = align_orientation(src, target)
        self.assertAlmostEqual(result["chamfer"], 0.0, places=9)

    def test_translation_invariant(self):
        src = self._shape()
        target = [(x + 100.0, y - 50.0, z + 7.0) for (x, y, z) in src]
        self.assertAlmostEqual(aligned_chamfer(src, target), 0.0, places=9)

    def test_empty(self):
        with self.assertRaises(ValueError):
            align_orientation([], [(0.0, 0.0, 0.0)])


if __name__ == "__main__":
    unittest.main()
