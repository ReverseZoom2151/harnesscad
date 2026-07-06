import unittest

from reconstruction.img2cadsvg_binding import (
    nearest_endpoint,
    bind_segment,
    bind_and_select,
    collinearity_quality,
)


class NearestTest(unittest.TestCase):
    def test_nearest(self):
        eps = [(0, 0), (5, 0), (0, 5)]
        i, d = nearest_endpoint((4.9, 0.1), eps)
        self.assertEqual(i, 1)
        self.assertAlmostEqual(d, 0.01 + 0.01, places=6)

    def test_tie_lowest_index(self):
        eps = [(1, 0), (-1, 0)]
        i, _ = nearest_endpoint((0, 0), eps)
        self.assertEqual(i, 0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            nearest_endpoint((0, 0), [])


class BindTest(unittest.TestCase):
    def test_bind_segment_snaps(self):
        endpoints = [(0, 0), (4, 0)]
        b = bind_segment(((0.1, 0.0), (3.9, 0.0)), endpoints)
        self.assertEqual((b.i1, b.i2), (0, 1))
        self.assertEqual(b.snapped(), ((0, 0), (4, 0)))
        self.assertAlmostEqual(b.delta, max(b.delta1, b.delta2))

    def test_select_by_epsilon(self):
        endpoints = [(0, 0), (4, 0)]
        segs = [
            ((0.05, 0.0), (3.95, 0.0)),  # low delta -> kept
            ((0.0, 0.0), (3.0, 3.0)),  # far from (4,0) -> high delta
        ]
        kept = bind_and_select(segs, endpoints, epsilon=0.1)
        self.assertEqual(len(kept), 1)

    def test_rejects_same_endpoint(self):
        endpoints = [(0, 0), (10, 10)]
        # both proposal endpoints snap to (0,0)
        kept = bind_and_select([((0.1, 0.1), (0.2, 0.0))], endpoints, epsilon=100.0)
        self.assertEqual(kept, [])

    def test_epsilon_must_be_positive(self):
        with self.assertRaises(ValueError):
            bind_and_select([], [(0, 0)], epsilon=0.0)


class QualityTest(unittest.TestCase):
    def test_quality_monotone(self):
        endpoints = [(0, 0), (4, 0)]
        near = bind_segment(((0.0, 0.0), (4.0, 0.0)), endpoints)
        far = bind_segment(((0.0, 1.0), (4.0, 1.0)), endpoints)
        self.assertGreater(collinearity_quality(near), collinearity_quality(far))
        self.assertAlmostEqual(collinearity_quality(near), 1.0)


if __name__ == "__main__":
    unittest.main()
