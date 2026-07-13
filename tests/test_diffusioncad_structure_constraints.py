import unittest

from harnesscad.domain.geometry.sketch.diffusioncad_structure_constraints import (
    available_constraints,
    enforce,
    line_line_parallel,
    line_line_perpendicular,
    point_point_coincidence,
    symmetry,
)


class TestCoincidence(unittest.TestCase):
    def test_snaps_b_to_a(self):
        r = point_point_coincidence((3, 4), (9, 1))
        self.assertEqual(r.points[0], (3, 4))
        self.assertEqual(r.points[1], (3, 4))
        self.assertTrue(r.valid)
        self.assertFalse(r.satisfied)

    def test_already_satisfied(self):
        r = point_point_coincidence((2, 2), (2, 2))
        self.assertTrue(r.satisfied)


class TestParallel(unittest.TestCase):
    def test_makes_both_vertical(self):
        r = line_line_parallel((1, 0), (5, 3), (10, 0), (14, 8))
        (a, b, c, d) = r.points
        self.assertEqual(a[0], b[0])  # AB vertical
        self.assertEqual(c[0], d[0])  # CD vertical
        self.assertTrue(r.valid)

    def test_preserves_y(self):
        r = line_line_parallel((1, 7), (5, 3), (10, 2), (14, 8))
        self.assertEqual(r.points[1], (1, 3))
        self.assertEqual(r.points[3], (10, 8))

    def test_satisfied_detection(self):
        r = line_line_parallel((1, 0), (1, 9), (4, 0), (4, 9))
        self.assertTrue(r.satisfied)


class TestPerpendicular(unittest.TestCase):
    def test_ab_vertical_cd_horizontal(self):
        r = line_line_perpendicular((2, 1), (7, 6), (3, 4), (9, 8))
        (a, b, c, d) = r.points
        self.assertEqual(a[0], b[0])  # AB vertical
        self.assertEqual(c[1], d[1])  # CD horizontal
        self.assertTrue(r.valid)

    def test_satisfied(self):
        r = line_line_perpendicular((2, 1), (2, 6), (3, 4), (9, 4))
        self.assertTrue(r.satisfied)


class TestSymmetry(unittest.TestCase):
    def test_even_sum_valid(self):
        r = symmetry((2, 5), (8, 9), (0, 1), (0, 7))
        (a, b, c, d) = r.points
        self.assertEqual(a[1], b[1])  # y1 = y2
        self.assertEqual(c[0], d[0])  # axis shared
        self.assertEqual(c[0], (2 + 8) // 2)  # axis at midpoint 5
        self.assertTrue(r.valid)

    def test_odd_sum_invalid(self):
        r = symmetry((2, 5), (7, 9), (0, 1), (0, 7))  # 2+7 = 9 odd
        self.assertFalse(r.valid)
        self.assertIn("non-integer", r.note)
        # axis still integral (rounded)
        self.assertIsInstance(r.points[2][0], int)

    def test_satisfied_when_prealigned(self):
        r = symmetry((2, 5), (8, 5), (5, 1), (5, 7))
        self.assertTrue(r.satisfied)
        self.assertTrue(r.valid)


class TestDispatch(unittest.TestCase):
    def test_enforce_names(self):
        self.assertEqual(
            set(available_constraints()),
            {"coincidence", "parallel", "perpendicular", "symmetry"},
        )

    def test_enforce_coincidence(self):
        r = enforce("coincidence", [(1, 1), (2, 2)])
        self.assertEqual(r.points[1], (1, 1))

    def test_enforce_four_point(self):
        r = enforce("perpendicular", [(2, 1), (7, 6), (3, 4), (9, 8)])
        self.assertTrue(r.valid)

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            enforce("parallel", [(1, 1), (2, 2)])
        with self.assertRaises(ValueError):
            enforce("coincidence", [(1, 1)])

    def test_unknown_constraint(self):
        with self.assertRaises(KeyError):
            enforce("tangent", [(0, 0), (1, 1), (2, 2), (3, 3)])


if __name__ == "__main__":
    unittest.main()
