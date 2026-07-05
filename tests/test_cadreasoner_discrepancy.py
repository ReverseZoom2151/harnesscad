"""Tests for editing/cadreasoner_discrepancy.py (cross-shape offset encoding)."""

import unittest

from editing.cadreasoner_discrepancy import (
    DiscrepancyEncoding,
    encode_discrepancy,
    encode_null_init,
    nearest,
)


class TestNearest(unittest.TestCase):
    def test_finds_nearest_and_distance(self):
        cloud = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        i, d = nearest((1.2, 0.0, 0.0), cloud)
        self.assertEqual(i, 2)
        self.assertAlmostEqual(d, 0.2)

    def test_ties_resolve_to_lowest_index(self):
        cloud = [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
        i, _ = nearest((0.0, 0.0, 0.0), cloud)
        self.assertEqual(i, 0)

    def test_empty_cloud_raises(self):
        with self.assertRaises(ValueError):
            nearest((0.0, 0.0, 0.0), [])


class TestEncodeDiscrepancy(unittest.TestCase):
    def test_offset_points_toward_nearest_opposite(self):
        target = [(0.0, 0.0, 0.0)]
        render = [(2.0, 0.0, 0.0)]
        enc = encode_discrepancy(target, render, k=0)
        # single target point: offset = render - target = (2,0,0)
        p = enc.target_offsets[0]
        self.assertEqual(p[:3], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(p[3], 2.0)
        self.assertAlmostEqual(enc.max_discrepancy, 2.0)

    def test_zero_when_shapes_coincide(self):
        pts = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        enc = encode_discrepancy(pts, pts, k=0)
        self.assertAlmostEqual(enc.max_discrepancy, 0.0)
        self.assertAlmostEqual(enc.symmetric_discrepancy, 0.0)

    def test_farthest_point_selection_keeps_k(self):
        target = [(float(i), 0.0, 0.0) for i in range(10)]
        render = [(0.0, 0.0, 0.0)]
        enc = encode_discrepancy(target, render, k=3)
        self.assertEqual(len(enc.target_offsets), 3)
        # The three farthest target points are x=9,8,7 (largest distance first).
        xs = [p[0] for p in enc.target_offsets]
        self.assertEqual(xs, [9.0, 8.0, 7.0])

    def test_symmetric_discrepancy_is_mean_of_directed_means(self):
        target = [(0.0, 0.0, 0.0)]
        render = [(4.0, 0.0, 0.0)]
        enc = encode_discrepancy(target, render, k=0)
        # both directed means = 4.0
        self.assertAlmostEqual(enc.symmetric_discrepancy, 4.0)

    def test_points_union(self):
        enc = encode_discrepancy([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], k=0)
        self.assertEqual(len(enc.points),
                         len(enc.target_offsets) + len(enc.render_offsets))

    def test_deterministic(self):
        t = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        r = [(0.0, 0.0, 0.0), (2.0, 2.0, 2.0)]
        self.assertEqual(encode_discrepancy(t, r).to_dict(),
                         encode_discrepancy(t, r).to_dict())

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            encode_discrepancy([], [(0.0, 0.0, 0.0)])
        with self.assertRaises(ValueError):
            encode_discrepancy([(0.0, 0.0, 0.0)], [])


class TestNullInit(unittest.TestCase):
    def test_target_offsets_point_to_origin(self):
        target = [(3.0, 4.0, 0.0)]  # distance 5 from origin
        enc = encode_null_init(target, k=0, seed=0)
        self.assertTrue(enc.t1)
        p = enc.target_offsets[0]
        self.assertEqual(p[:3], (3.0, 4.0, 0.0))
        self.assertEqual(p[3:], (-3.0, -4.0, -0.0))
        self.assertAlmostEqual(enc.max_discrepancy, 5.0)

    def test_render_offsets_are_origin_paired_to_permuted_target(self):
        target = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        enc = encode_null_init(target, k=0, seed=0)
        for q in enc.render_offsets:
            # position at origin, offset equals some target sample
            self.assertEqual(q[:3], (0.0, 0.0, 0.0))
            self.assertIn(q[3:], {(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)})

    def test_permutation_is_seed_deterministic(self):
        target = [(float(i), 0.0, 0.0) for i in range(20)]
        a = encode_null_init(target, k=0, seed=7).to_dict()
        b = encode_null_init(target, k=0, seed=7).to_dict()
        self.assertEqual(a, b)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            encode_null_init([])


if __name__ == "__main__":
    unittest.main()
