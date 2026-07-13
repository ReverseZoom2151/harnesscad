"""Tests for Meshtron mesh-sequence ordering enforcement."""

import unittest

from harnesscad.io.formats.meshtron_order_enforcement import (
    eos_allowed,
    invalid_category_count,
    invalid_category_mask,
    invalid_fraction,
    is_face_vertices_ordered,
    is_stream_ordered,
)


ORDERED_FACES = [
    [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
    [(0, 0, 0), (0, 1, 0), (1, 0, 0)],
    [(1, 0, 0), (1, 0, 1), (1, 1, 0)],
]


class FaceOrderTest(unittest.TestCase):
    def test_ordered_face(self):
        self.assertTrue(is_face_vertices_ordered([(0, 0, 0), (0, 0, 1), (0, 1, 0)]))

    def test_unordered_face(self):
        self.assertFalse(is_face_vertices_ordered([(0, 0, 1), (0, 0, 0)]))

    def test_stream_ordered(self):
        self.assertTrue(is_stream_ordered(ORDERED_FACES))

    def test_stream_unordered_faces(self):
        bad = [
            [(1, 0, 0), (1, 0, 1), (1, 1, 0)],
            [(0, 0, 0), (0, 1, 0), (1, 0, 0)],  # face key drops
        ]
        self.assertFalse(is_stream_ordered(bad))

    def test_bad_face_size(self):
        with self.assertRaises(ValueError):
            is_stream_ordered([[(0, 0, 0), (0, 0, 1)]])


class InvalidCategoryTest(unittest.TestCase):
    def test_first_coord_forbids_below_bound(self):
        # lower bound (2,3,4): first coord must be >= 2 -> 2 invalid categories
        self.assertEqual(invalid_category_count((2, 3, 4), (), 10), 2)

    def test_matching_prefix_forbids_next_bound(self):
        self.assertEqual(invalid_category_count((2, 3, 4), (2,), 10), 3)
        self.assertEqual(invalid_category_count((2, 3, 4), (2, 3), 10), 4)

    def test_greater_prefix_unconstrained(self):
        self.assertEqual(invalid_category_count((2, 3, 4), (5,), 10), 0)
        self.assertEqual(invalid_category_count((2, 3, 4), (2, 9), 10), 0)

    def test_below_prefix_raises(self):
        with self.assertRaises(ValueError):
            invalid_category_count((2, 3, 4), (1,), 10)

    def test_mask_matches_count(self):
        mask = invalid_category_mask((2, 3, 4), (), 10)
        self.assertEqual(sum(mask), 2)
        self.assertEqual(mask[:3], [True, True, False])

    def test_bad_num_bins(self):
        with self.assertRaises(ValueError):
            invalid_category_count((0, 0, 0), (), 0)

    def test_prefix_too_long(self):
        with self.assertRaises(ValueError):
            invalid_category_count((0, 0, 0), (0, 0, 0), 10)


class EosTest(unittest.TestCase):
    def test_eos_at_face_start(self):
        self.assertTrue(eos_allowed(0))
        self.assertTrue(eos_allowed(9))
        self.assertTrue(eos_allowed(18))

    def test_eos_mid_face(self):
        self.assertFalse(eos_allowed(3))
        self.assertFalse(eos_allowed(8))

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            eos_allowed(-1)


class InvalidFractionTest(unittest.TestCase):
    def test_fraction_in_unit_interval(self):
        frac = invalid_fraction(ORDERED_FACES, num_bins=8)
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)

    def test_fraction_positive_when_constrained(self):
        # bounds above zero must forbid some categories
        self.assertGreater(invalid_fraction(ORDERED_FACES, num_bins=8), 0.0)

    def test_requires_ordered_input(self):
        bad = [
            [(1, 0, 0), (1, 0, 1), (1, 1, 0)],
            [(0, 0, 0), (0, 1, 0), (1, 0, 0)],
        ]
        with self.assertRaises(ValueError):
            invalid_fraction(bad, num_bins=8)

    def test_deterministic(self):
        a = invalid_fraction(ORDERED_FACES, num_bins=16)
        b = invalid_fraction(ORDERED_FACES, num_bins=16)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
