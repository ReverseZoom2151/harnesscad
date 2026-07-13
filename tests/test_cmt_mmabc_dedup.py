import unittest

from harnesscad.domain.reconstruction.recognize.cmt_mmabc_dedup import (
    quantize_point, model_hash, deduplicate, connected_bodies,
)


class TestQuantizePoint(unittest.TestCase):
    def test_six_bit(self):
        self.assertEqual(quantize_point((0.0, 0.5, 1.0), bits=6), (0, 32, 63))


class TestModelHash(unittest.TestCase):
    def test_order_independent(self):
        a = model_hash(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        b = model_hash(((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))
        self.assertEqual(a, b)

    def test_near_duplicates_collide_at_6bit(self):
        # points within one 6-bit level quantize identically
        a = model_hash(((0.0, 0.0, 0.0),))
        b = model_hash(((0.001, 0.001, 0.001),))
        self.assertEqual(a, b)

    def test_distinct_differ(self):
        a = model_hash(((0.0, 0.0, 0.0),))
        b = model_hash(((0.9, 0.9, 0.9),))
        self.assertNotEqual(a, b)

    def test_deterministic(self):
        pts = ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
        self.assertEqual(model_hash(pts), model_hash(pts))


class TestDeduplicate(unittest.TestCase):
    def test_keeps_first_drops_dupes(self):
        models = (
            ("a", ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            ("b", ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))),  # same set as a
            ("c", ((0.5, 0.5, 0.5),)),
        )
        result = deduplicate(models)
        self.assertEqual(result.kept, ("a", "c"))
        self.assertEqual(result.dropped, ("b",))
        self.assertEqual(result.groups, (("a", "b"),))

    def test_all_unique(self):
        models = (
            ("a", ((0.0, 0.0, 0.0),)),
            ("b", ((1.0, 1.0, 1.0),)),
        )
        result = deduplicate(models)
        self.assertEqual(result.kept, ("a", "b"))
        self.assertEqual(result.dropped, ())
        self.assertEqual(result.groups, ())


class TestConnectedBodies(unittest.TestCase):
    def test_single_body(self):
        bodies = connected_bodies(4, ((0, 1), (1, 2), (2, 3)))
        self.assertEqual(bodies, ((0, 1, 2, 3),))

    def test_two_bodies(self):
        bodies = connected_bodies(4, ((0, 1), (2, 3)))
        self.assertEqual(bodies, ((0, 1), (2, 3)))

    def test_isolated_surfaces(self):
        bodies = connected_bodies(3, ())
        self.assertEqual(bodies, ((0,), (1,), (2,)))

    def test_ordered_by_min_index(self):
        bodies = connected_bodies(4, ((3, 1), (0, 2)))
        self.assertEqual(bodies, ((0, 2), (1, 3)))

    def test_bad_index(self):
        with self.assertRaises(ValueError):
            connected_bodies(2, ((0, 5),))

    def test_negative(self):
        with self.assertRaises(ValueError):
            connected_bodies(-1, ())


if __name__ == "__main__":
    unittest.main()
