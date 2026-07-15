"""Tests for the CanonicalVAE template-matched canonical point ordering."""

import unittest

from harnesscad.domain.geometry.transforms import canonical_point_order as cpo


class TemplateTest(unittest.TestCase):
    def test_fibonacci_sphere_count_and_norm(self):
        pts = cpo.fibonacci_sphere(50)
        self.assertEqual(len(pts), 50)
        for p in pts:
            self.assertAlmostEqual(sum(c * c for c in p), 1.0, places=6)

    def test_fibonacci_deterministic(self):
        self.assertEqual(cpo.fibonacci_sphere(20), cpo.fibonacci_sphere(20))

    def test_grid_template(self):
        g = cpo.grid_template(2, 3)
        self.assertEqual(len(g), 6)
        self.assertTrue(all(p[2] == 0.0 for p in g))


class NearestOrderTest(unittest.TestCase):
    def setUp(self):
        # a 4-slot square template
        self.template = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]

    def test_nearest_index(self):
        pts = [(0.9, 0.9, 0), (-0.9, -0.9, 0)]
        idx = cpo.nearest_template_index(pts, self.template)
        self.assertEqual(idx, [2, 0])

    def test_canonical_order_sorts_by_slot(self):
        # points near slots 2, 0, 3 -> order should walk slots ascending: 0,2,3
        pts = [(0.9, 0.9, 0), (-0.9, -0.9, 0), (-0.9, 0.9, 0)]
        order = cpo.canonical_order(pts, self.template)
        reordered = cpo.reorder(pts, order)
        self.assertEqual(reordered[0], (-0.9, -0.9, 0))  # slot 0 first
        self.assertEqual(reordered[-1], (-0.9, 0.9, 0))  # slot 3 last

    def test_empty_template_raises(self):
        with self.assertRaises(ValueError):
            cpo.nearest_template_index([(0, 0, 0)], [])


class BijectiveTest(unittest.TestCase):
    def test_is_permutation(self):
        template = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        pts = [(1.1, 1.1, 0), (-1.1, -1.1, 0), (-1.1, 1.1, 0), (1.1, -1.1, 0)]
        assign = cpo.bijective_assignment(pts, template)
        self.assertEqual(sorted(assign), [0, 1, 2, 3])
        # point 0 nearest slot 2, point 1 nearest slot 0, etc.
        self.assertEqual(assign[0], 2)
        self.assertEqual(assign[1], 0)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cpo.bijective_assignment([(0, 0, 0)], [(0, 0, 0), (1, 1, 1)])


class CanonicalDistanceTest(unittest.TestCase):
    def setUp(self):
        self.template = cpo.fibonacci_sphere(16)

    def test_zero_for_identical_shapes(self):
        shape = [(0.5 * i, 0.1 * i, -0.2 * i) for i in range(8)]
        self.assertAlmostEqual(cpo.canonical_distance(shape, shape, self.template), 0.0)

    def test_invariant_to_input_permutation(self):
        shape = [(0.5, 0.1, 0.2), (-0.3, 0.4, -0.1), (0.9, -0.5, 0.0), (0.0, 0.0, 0.7)]
        shuffled = [shape[2], shape[0], shape[3], shape[1]]
        d = cpo.canonical_distance(shape, shuffled, self.template)
        self.assertAlmostEqual(d, 0.0)

    def test_positive_for_different_shapes(self):
        a = [(0.5, 0.5, 0.5), (-0.5, -0.5, -0.5)]
        b = [(0.9, 0.1, 0.2), (-0.1, -0.9, -0.2)]
        self.assertGreater(cpo.canonical_distance(a, b, self.template), 0.0)


if __name__ == "__main__":
    unittest.main()
