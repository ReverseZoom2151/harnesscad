"""Tests for editing.voxhammer_mask."""
import unittest

from harnesscad.domain.editing.voxhammer_mask import (
    binary_mask,
    dilate,
    edit_voxels_in_box,
    edit_voxels_in_sphere,
    preserved_voxels,
    soft_mask,
)


class TestBoxSelection(unittest.TestCase):
    def setUp(self):
        self.coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (5, 5, 5)]

    def test_inclusive_box(self):
        edit = edit_voxels_in_box(self.coords, (0, 0, 0), (1, 0, 0))
        self.assertEqual(edit, frozenset({(0, 0, 0), (1, 0, 0)}))

    def test_outside_excluded(self):
        edit = edit_voxels_in_box(self.coords, (0, 0, 0), (2, 0, 0))
        self.assertNotIn((5, 5, 5), edit)

    def test_bad_box_raises(self):
        with self.assertRaises(ValueError):
            edit_voxels_in_box(self.coords, (3, 0, 0), (1, 0, 0))


class TestSphereSelection(unittest.TestCase):
    def test_radius(self):
        coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        edit = edit_voxels_in_sphere(coords, (0, 0, 0), 1.0)
        self.assertEqual(edit, frozenset({(0, 0, 0), (1, 0, 0)}))

    def test_negative_radius(self):
        with self.assertRaises(ValueError):
            edit_voxels_in_sphere([(0, 0, 0)], (0, 0, 0), -1.0)


class TestPreservedAndBinary(unittest.TestCase):
    def setUp(self):
        self.coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        self.edit = {(0, 0, 0)}

    def test_preserved_is_complement(self):
        keep = preserved_voxels(self.coords, self.edit)
        self.assertEqual(keep, frozenset({(1, 0, 0), (2, 0, 0)}))

    def test_partition(self):
        keep = preserved_voxels(self.coords, self.edit)
        edit = frozenset(self.edit)
        self.assertEqual(keep | edit, frozenset(self.coords))
        self.assertEqual(keep & edit, frozenset())

    def test_binary_values(self):
        m = binary_mask(self.coords, self.edit)
        self.assertEqual(m[(0, 0, 0)], 1.0)
        self.assertEqual(m[(1, 0, 0)], 0.0)
        self.assertEqual(m[(2, 0, 0)], 0.0)


class TestDilate(unittest.TestCase):
    def test_dilate_zero_is_identity(self):
        e = {(0, 0, 0)}
        self.assertEqual(dilate(e, 0), frozenset(e))

    def test_dilate_6connectivity(self):
        e = {(0, 0, 0)}
        d = dilate(e, 1, connectivity=6)
        # 6 face neighbours + self
        self.assertEqual(len(d), 7)
        self.assertIn((1, 0, 0), d)
        self.assertNotIn((1, 1, 0), d)

    def test_dilate_26connectivity(self):
        d = dilate({(0, 0, 0)}, 1, connectivity=26)
        self.assertEqual(len(d), 27)
        self.assertIn((1, 1, 1), d)

    def test_bad_connectivity(self):
        with self.assertRaises(ValueError):
            dilate({(0, 0, 0)}, 1, connectivity=7)


class TestSoftMask(unittest.TestCase):
    def setUp(self):
        self.coords = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        self.edit = {(0, 0, 0)}

    def test_edit_voxel_is_one(self):
        m = soft_mask(self.coords, self.edit, dilation=0, sigma=1.0)
        self.assertEqual(m[(0, 0, 0)], 1.0)

    def test_plateau_within_dilation(self):
        m = soft_mask(self.coords, self.edit, dilation=1, sigma=1.0)
        self.assertEqual(m[(1, 0, 0)], 1.0)  # distance 1, within plateau

    def test_falloff_monotone(self):
        m = soft_mask(self.coords, self.edit, dilation=0, sigma=1.0)
        # weights strictly decreasing with distance
        self.assertGreater(m[(1, 0, 0)], m[(2, 0, 0)])
        self.assertGreater(m[(2, 0, 0)], m[(3, 0, 0)])

    def test_range_bounded(self):
        m = soft_mask(self.coords, self.edit, dilation=1, sigma=0.7)
        for w in m.values():
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)

    def test_bad_sigma(self):
        with self.assertRaises(ValueError):
            soft_mask(self.coords, self.edit, sigma=0.0)

    def test_deterministic(self):
        a = soft_mask(self.coords, self.edit, dilation=1, sigma=1.3)
        b = soft_mask(self.coords, self.edit, dilation=1, sigma=1.3)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
