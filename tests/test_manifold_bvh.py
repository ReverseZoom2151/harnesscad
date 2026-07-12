"""Tests for geometry.manifold_bvh."""

import random
import unittest

from geometry.manifold_bvh import (
    AABB,
    BVH,
    morton3,
    spread_bits3,
    boxes_of_triangles,
    brute_force_pairs,
)


class TestAABB(unittest.TestCase):
    def test_overlap(self):
        a = AABB((0, 0, 0), (1, 1, 1))
        b = AABB((0.5, 0.5, 0.5), (2, 2, 2))
        c = AABB((2, 2, 2), (3, 3, 3))
        self.assertTrue(a.overlaps(b))
        self.assertFalse(a.overlaps(c))
        # touching counts as overlap (closed intervals)
        d = AABB((1, 1, 1), (2, 2, 2))
        self.assertTrue(a.overlaps(d))

    def test_union_and_center(self):
        a = AABB((0, 0, 0), (1, 1, 1))
        b = AABB((2, 2, 2), (3, 3, 3))
        u = a.union(b)
        self.assertEqual(u.min, (0, 0, 0))
        self.assertEqual(u.max, (3, 3, 3))
        self.assertEqual(a.center(), (0.5, 0.5, 0.5))

    def test_contains_point(self):
        a = AABB((0, 0, 0), (1, 1, 1))
        self.assertTrue(a.contains_point((0.5, 0.5, 0.5)))
        self.assertFalse(a.contains_point((1.5, 0.5, 0.5)))


class TestMorton(unittest.TestCase):
    def test_spread_bits(self):
        # bit 0 stays at 0, bit 1 -> position 3, etc.
        self.assertEqual(spread_bits3(1), 1)
        self.assertEqual(spread_bits3(2), 0b1000)
        self.assertEqual(spread_bits3(0b11), 0b1001)

    def test_morton_monotone_on_axis(self):
        box = AABB((0, 0, 0), (1, 1, 1))
        m_lo = morton3((0.1, 0.0, 0.0), box)
        m_hi = morton3((0.9, 0.0, 0.0), box)
        self.assertLess(m_lo, m_hi)


class TestBVHQuery(unittest.TestCase):
    def test_matches_brute_force_query(self):
        rng = random.Random(1234)
        boxes = []
        for _ in range(60):
            lo = (rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5))
            sz = (rng.uniform(0.1, 2), rng.uniform(0.1, 2), rng.uniform(0.1, 2))
            boxes.append(AABB(lo, (lo[0] + sz[0], lo[1] + sz[1], lo[2] + sz[2])))
        bvh = BVH(boxes)
        for qi in range(len(boxes)):
            q = boxes[qi]
            got = set(bvh.query(q))
            expected = {j for j in range(len(boxes)) if boxes[j].overlaps(q)}
            self.assertEqual(got, expected)

    def test_self_collisions_match_brute_force(self):
        rng = random.Random(99)
        boxes = []
        for _ in range(80):
            lo = (rng.uniform(0, 10), rng.uniform(0, 10), rng.uniform(0, 10))
            boxes.append(AABB(lo, (lo[0] + 1.0, lo[1] + 1.0, lo[2] + 1.0)))
        bvh = BVH(boxes)
        self.assertEqual(bvh.self_collisions(), brute_force_pairs(boxes))

    def test_no_self_report(self):
        boxes = [AABB((0, 0, 0), (1, 1, 1))]
        bvh = BVH(boxes)
        self.assertEqual(bvh.self_collisions(), [])

    def test_query_point(self):
        boxes = [
            AABB((0, 0, 0), (1, 1, 1)),
            AABB((2, 2, 2), (3, 3, 3)),
            AABB((0.5, 0.5, 0.5), (4, 4, 4)),
        ]
        bvh = BVH(boxes)
        self.assertEqual(set(bvh.query_point((0.6, 0.6, 0.6))), {0, 2})

    def test_bounding_box(self):
        boxes = [AABB((0, 0, 0), (1, 1, 1)), AABB((5, 5, 5), (6, 6, 6))]
        bvh = BVH(boxes)
        bb = bvh.bounding_box()
        self.assertEqual(bb.min, (0, 0, 0))
        self.assertEqual(bb.max, (6, 6, 6))


class TestTriangleBoxes(unittest.TestCase):
    def test_boxes_of_triangles(self):
        v = [(0, 0, 0), (2, 0, 0), (0, 3, 1)]
        boxes = boxes_of_triangles(v, [(0, 1, 2)])
        self.assertEqual(boxes[0].min, (0, 0, 0))
        self.assertEqual(boxes[0].max, (2, 3, 1))

    def test_empty_bvh(self):
        bvh = BVH([])
        self.assertEqual(bvh.query(AABB((0, 0, 0), (1, 1, 1))), [])
        self.assertTrue(bvh.bounding_box().is_empty())


if __name__ == "__main__":
    unittest.main()
