"""Tests for procedural.proccad_symmetry."""

import unittest
from math import isclose, tau

from harnesscad.domain.procedural.proccad_symmetry import (
    SymmetryReducer,
    bilateral,
    dihedral,
    mirror_point,
    nfold,
    rotate_point,
    symmetry_consistency,
)


class ReplicationTest(unittest.TestCase):
    def test_rotate_quarter_turn(self):
        x, y = rotate_point((1.0, 0.0), tau / 4)
        self.assertTrue(isclose(x, 0.0, abs_tol=1e-12))
        self.assertTrue(isclose(y, 1.0, abs_tol=1e-12))

    def test_rotate_about_center(self):
        p = rotate_point((2.0, 1.0), tau / 2, center=(1.0, 1.0))
        self.assertAlmostEqual(p[0], 0.0)
        self.assertAlmostEqual(p[1], 1.0)

    def test_mirror_y_axis(self):
        self.assertEqual(mirror_point((3.0, 5.0), "y"), (-3.0, 5.0))

    def test_mirror_x_axis(self):
        self.assertEqual(mirror_point((3.0, 5.0), "x"), (3.0, -5.0))

    def test_nfold_count(self):
        copies = nfold([(1.0, 0.0)], 4)
        self.assertEqual(len(copies), 4)
        # copy 0 is original
        self.assertAlmostEqual(copies[0][0][0], 1.0)

    def test_nfold_places_on_circle(self):
        copies = nfold([(1.0, 0.0)], 4)
        # the four single points should be at the 4 axis directions
        xs = sorted(round(c[0][0], 6) for c in copies)
        self.assertEqual(xs, [-1.0, 0.0, 0.0, 1.0])

    def test_bilateral(self):
        orig, mir = bilateral([(2.0, 1.0)], "y")
        self.assertEqual(orig[0], (2.0, 1.0))
        self.assertEqual(mir[0], (-2.0, 1.0))

    def test_dihedral_count(self):
        copies = dihedral([(1.0, 0.0), (2.0, 0.0)], 4)
        self.assertEqual(len(copies), 8)  # 2*order


class ParameterReductionTest(unittest.TestCase):
    def test_full_size(self):
        r = SymmetryReducer(order=4, base_size=3)
        self.assertEqual(r.full_size, 12)

    def test_reduced_count_factor(self):
        r = SymmetryReducer(order=4, base_size=3)
        # 4-fold symmetry reduces 12 params to 3 (factor of four)
        self.assertEqual(r.reduced_count(12), 3)

    def test_expand_replicates(self):
        r = SymmetryReducer(order=4, base_size=2)
        self.assertEqual(r.expand([1.0, 2.0]), (1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0))

    def test_reduce_roundtrip(self):
        r = SymmetryReducer(order=3, base_size=2)
        base = [7.0, -1.5]
        self.assertEqual(r.reduce(r.expand(base)), tuple(base))

    def test_reduce_rejects_asymmetric(self):
        r = SymmetryReducer(order=2, base_size=2)
        with self.assertRaises(ValueError):
            r.reduce([1.0, 2.0, 1.0, 9.0])  # second block differs

    def test_is_symmetric(self):
        r = SymmetryReducer(order=2, base_size=2)
        self.assertTrue(r.is_symmetric([1.0, 2.0, 1.0, 2.0]))
        self.assertFalse(r.is_symmetric([1.0, 2.0, 1.0, 3.0]))


class ConsistencyTest(unittest.TestCase):
    def test_rotated_copies_are_consistent(self):
        motif = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0)]
        copies = nfold(motif, 4)
        self.assertTrue(symmetry_consistency(copies))

    def test_mismatched_shapes_flagged(self):
        good = [(0.0, 0.0), (1.0, 0.0)]
        bad = [(0.0, 0.0), (5.0, 0.0)]
        self.assertFalse(symmetry_consistency([good, bad]))

    def test_empty_is_consistent(self):
        self.assertTrue(symmetry_consistency([]))


if __name__ == "__main__":
    unittest.main()
